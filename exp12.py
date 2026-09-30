
import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DTYPE = torch.float64
CDTYPE = torch.complex128
MIN_SEEDS_FOR_TEST = 5
MIN_QUANT_BITS = 2
FREEZE_TOL = 1e-9
IMG_COLS = 4  # RY, RX, RZ, RY encoding angles per qubit (encoding block unchanged from v11)
# NEW in v12: per-qubit variational rotation is a full RZ-RY-RZ Euler triple (params
# index 1,2,3) instead of v11's single RY (index 1 only). Index 0 stays the ZZ
# entangler param. See module docstring, point 1.
ROT_ORDER = ("RZ", "RY", "RZ")

DATASETS = {
    "mnist": ("MNIST", 3, 6),
    "fashionmnist": ("FashionMNIST", 3, 6),
    "digits": (None, 3, 6),  # sklearn 8x8 digits: offline smoke-test dataset only
}


def pick_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            sys.exit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# Data (unchanged from v11)
# ----------------------------------------------------------------------
_RAW_CACHE = {}


def _load_raw(name, split, root):
    key = (name, split)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    if name == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        X = d.images.astype(np.float64) / 16.0
        y = np.asarray(d.target)
    else:
        import torchvision
        cls = getattr(torchvision.datasets, DATASETS[name][0])
        ds = cls(root=root, train=(split == "train"), download=True)
        X = np.asarray(ds.data).astype(np.float64) / 255.0
        X = X[:, 2:26, 2:26]
        y = np.asarray(ds.targets)
    out = (X.reshape(len(X), -1), y)
    _RAW_CACHE[key] = out
    return out


def load_binary_subset(name, n_qubits, n_train, n_val, seed, eval_split="val", root="./data"):
    from sklearn.decomposition import PCA
    _, ca, cb = DATASETS[name]
    n_feat = n_qubits * IMG_COLS
    X_all, y_all = _load_raw(name, "train", root)
    rng = np.random.RandomState(seed)
    ia = np.where(y_all == ca)[0]
    ib = np.where(y_all == cb)[0]
    rng.shuffle(ia)
    rng.shuffle(ib)
    nt, nv = n_train // 2, n_val // 2
    tr = np.concatenate([ia[:nt], ib[:nt]])
    if eval_split == "test" and name != "digits":
        X_te, y_te = _load_raw(name, "test", root)
        ja = np.where(y_te == ca)[0]
        jb = np.where(y_te == cb)[0]
        rng.shuffle(ja)
        rng.shuffle(jb)
        ev_idx = np.concatenate([ja[:nv], jb[:nv]])
        X_ev_raw, y_ev = X_te[ev_idx], (y_te[ev_idx] == cb).astype(np.int64)
    else:
        ev = np.concatenate([ia[nt:nt + nv], ib[nt:nt + nv]])
        X_ev_raw, y_ev = X_all[ev], (y_all[ev] == cb).astype(np.int64)
    rng.shuffle(tr)
    X_tr_raw, y_tr = X_all[tr], (y_all[tr] == cb).astype(np.int64)
    if len(X_tr_raw) < 2 or len(X_ev_raw) < 2:
        raise ValueError("not enough samples; lower --n-train/--n-val")

    pca = PCA(n_components=min(n_feat, X_tr_raw.shape[1], len(X_tr_raw) - 1), random_state=seed)
    Ftr = pca.fit_transform(X_tr_raw)
    Fev = pca.transform(X_ev_raw)
    if Ftr.shape[1] < n_feat:
        pad = n_feat - Ftr.shape[1]
        Ftr = np.pad(Ftr, ((0, 0), (0, pad)))
        Fev = np.pad(Fev, ((0, 0), (0, pad)))
    lo, hi = Ftr.min(0, keepdims=True), Ftr.max(0, keepdims=True)
    span = np.clip(hi - lo, 1e-8, None)
    Xtr = (Ftr - lo) / span
    Xev = np.clip((Fev - lo) / span, 0.0, 1.0)
    return {
        "X_train": Xtr.reshape(-1, n_qubits, IMG_COLS) * np.pi, "y_train": y_tr,
        "X_eval": Xev.reshape(-1, n_qubits, IMG_COLS) * np.pi, "y_eval": y_ev,
        "F_train": Xtr, "F_eval": Xev,
    }


def classical_sanity_baseline(F_train, y_train, F_eval, y_eval):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    clf = LogisticRegression(max_iter=3000).fit(F_train, y_train)
    return float(clf.score(F_eval, y_eval))


# ----------------------------------------------------------------------
# Exact batched PyTorch simulator (statevector + density matrix)
# CHANGED in v12: the variational layer's per-qubit gate is now the full
# RZ-RY-RZ Euler rotation (3 trainable angles, params[...,1:4]) instead of a
# single RY (params[...,1]). This is a universal (up to global phase)
# single-qubit unitary, unlike RY alone. The ZZ entangler (params[...,0])
# and the data-reuploading encoding block are unchanged from v11.
# ----------------------------------------------------------------------
def _gate_matrix(name, theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    z = torch.zeros_like(c)
    if name == "RY":
        re, im = [[c, -s], [s, c]], [[z, z], [z, z]]
    elif name == "RX":
        re, im = [[c, z], [z, c]], [[z, -s], [-s, z]]
    elif name == "RZ":
        re, im = [[c, z], [z, c]], [[-s, z], [z, s]]
    else:
        raise ValueError(name)

    def mat(a):
        return torch.stack([torch.stack(r, -1) for r in a], -2)
    return torch.complex(mat(re), mat(im))


def _apply_1q_sv(psi, U, q, n):
    B = psi.shape[0]
    s = psi.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
    if U.dim() == 2:
        out = torch.einsum("ab,xlbr->xlar", U, s)
    else:
        out = torch.einsum("xab,xlbr->xlar", U, s)
    return out.reshape(B, 1 << n)


def _apply_1q_dm(rho, U, q, n):
    B, dim = rho.shape[0], 1 << n
    r7 = rho.reshape(B, 1 << q, 2, 1 << (n - 1 - q), 1 << q, 2, 1 << (n - 1 - q))
    if U.dim() == 2:
        t = torch.einsum("ab,xlbrmcs->xlarmcs", U, r7)
        t = torch.einsum("cd,xlarmds->xlarmcs", U.conj(), t)
    else:
        t = torch.einsum("xab,xlbrmcs->xlarmcs", U, r7)
        t = torch.einsum("xcd,xlarmds->xlarmcs", U.conj(), t)
    return t.reshape(B, dim, dim)


def _depolarize_dm(rho, p, q, n):
    if p <= 0:
        return rho
    B, dim = rho.shape[0], 1 << n
    r7 = rho.reshape(B, 1 << q, 2, 1 << (n - 1 - q), 1 << q, 2, 1 << (n - 1 - q))
    tr = torch.einsum("blarmas->blrms", r7)
    eye2 = torch.eye(2, dtype=rho.dtype, device=rho.device)
    mixed = torch.einsum("blrms,ac->blarmcs", tr, eye2)
    out = (1 - 4.0 * p / 3.0) * r7 + (2.0 * p / 3.0) * mixed
    return out.reshape(B, dim, dim)


class QSim:
    """Circuit = [encode] + L x ([encode if reupload and not first] variational
    layer [IsingZZ(i,i+1) for i<n-1 ; RZ-RY-RZ(i) for all i]), measuring
    computational-basis statistics. Wire 0 is the most-significant bit
    (PennyLane convention). Encoding block = per-qubit RY,RX,RZ,RY with data
    angles; never pruned/quantized (data-dependent, not a trainable weight).
    Params shape: (n_layers, n_qubits, 4) -- index 0 = ZZ entangler angle
    (unused/False-live on the last qubit), indices 1,2,3 = the RZ,RY,RZ
    Euler angles of that qubit's variational rotation this layer (NEW in
    v12; v11 only had a single RY angle at index 1)."""

    def __init__(self, n_qubits, n_layers, feature_mode, device, reupload=True):
        n = n_qubits
        self.n, self.L, self.dim, self.device = n, n_layers, 1 << n, device
        self.reupload = reupload
        idx = np.arange(self.dim)
        bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
        Zs = 1.0 - 2.0 * bits
        cols = [Zs[:, q] for q in range(n)]
        if feature_mode == "z+zz":
            cols += [Zs[:, i] * Zs[:, i + 1] for i in range(n - 1)]
        elif feature_mode != "z":
            raise ValueError(feature_mode)
        self.feat_mat = torch.tensor(np.stack(cols, 1), dtype=DTYPE, device=device)
        self.n_features = len(cols)
        self.zz = [torch.tensor(Zs[:, i] * Zs[:, i + 1], dtype=DTYPE, device=device) for i in range(n - 1)]
        live = np.ones((n_layers, n, 4), dtype=bool)
        live[:, n - 1, 0] = False  # IsingZZ param of the last qubit row is never applied
        self.live_mask = live
        self.layer_ops_list = []
        for l in range(n_layers):
            ops = [("ZZ", (i, i + 1), (l, i, 0)) for i in range(n - 1)]
            for i in range(n):
                for k, kind in enumerate(ROT_ORDER):
                    ops.append((kind, (i,), (l, i, 1 + k)))
            self.layer_ops_list.append(ops)
        self.eye = torch.eye(self.dim, dtype=CDTYPE, device=device)

    def _zz_phase(self, i, theta):
        a = -0.5 * theta * self.zz[i]
        return torch.complex(torch.cos(a), torch.sin(a))

    def zero_state(self, B):
        psi = torch.zeros(B, self.dim, dtype=CDTYPE, device=self.device)
        psi[:, 0] = 1.0
        return psi

    # ---- encoding block (data-dependent, batched, repeatable) -------------
    def encode_block_sv(self, psi, X):
        for q in range(self.n):
            for k, g in enumerate(("RY", "RX", "RZ", "RY")):
                psi = _apply_1q_sv(psi, _gate_matrix(g, X[:, q, k]), q, self.n)
        return psi

    def encode_block_dm(self, rho, X, p):
        for q in range(self.n):
            for k, g in enumerate(("RY", "RX", "RZ", "RY")):
                rho = _apply_1q_dm(rho, _gate_matrix(g, X[:, q, k]), q, self.n)
                rho = _depolarize_dm(rho, p, q, self.n)
        return rho

    def encode_sv(self, X):
        """One-shot encoding of the |0> state; used to build the cached
        pre-layer-0 state (D['psi_train'] / D['psi_eval'])."""
        return self.encode_block_sv(self.zero_state(X.shape[0]), X)

    # ---- variational layer (parameter-only -> precomputable unitary) ------
    def var_unitary_layer(self, params, l):
        R = self.eye
        for kind, wires, (ll, i, j) in self.layer_ops_list[l]:
            th = params[ll, i, j]
            if kind == "ZZ":
                R = R * self._zz_phase(wires[0], th)[None, :]
            else:
                R = _apply_1q_sv(R, _gate_matrix(kind, th), wires[0], self.n)
        return R

    def var_dm_layer(self, rho, params_np, l, p, p2, tol=FREEZE_TOL):
        n = self.n
        for kind, wires, (ll, i, j) in self.layer_ops_list[l]:
            th = float(params_np[ll, i, j])
            if abs(th) <= tol:
                continue  # zeroed/frozen gate is not applied -> no gate, no noise
            tt = torch.tensor(th, dtype=DTYPE, device=self.device)
            if kind == "ZZ":
                ph = self._zz_phase(wires[0], tt)
                rho = rho * ph.view(1, -1, 1) * ph.conj().view(1, 1, -1)
                rho = _depolarize_dm(rho, p2, wires[0], n)
                rho = _depolarize_dm(rho, p2, wires[1], n)
            else:
                rho = _apply_1q_dm(rho, _gate_matrix(kind, tt), wires[0], n)
                rho = _depolarize_dm(rho, p, wires[0], n)
        return rho

    def features_from_psi(self, psi):
        p = psi.real ** 2 + psi.imag ** 2
        return p @ self.feat_mat

    # ---- full forward passes -----------------------------------------------
    def features_sv_forward(self, psi0, X, params):
        """psi0 = encode_sv(X) (cached once per seed). Re-applies the
        encoding block between layers iff self.reupload."""
        psi = psi0
        for l in range(self.L):
            psi = psi @ self.var_unitary_layer(params, l)
            if self.reupload and l < self.L - 1:
                psi = self.encode_block_sv(psi, X)
        return self.features_from_psi(psi)

    def features_noisy(self, X, params_np, p, p2_mult=1.0, tol=FREEZE_TOL):
        p2 = min(1.0, p * p2_mult)
        chunk = max(8, int(3e6 // (self.dim * self.dim)))
        out = []
        for s in range(0, X.shape[0], chunk):
            Xc = X[s:s + chunk]
            rho = torch.zeros(Xc.shape[0], self.dim, self.dim, dtype=CDTYPE, device=self.device)
            rho[:, 0, 0] = 1.0
            rho = self.encode_block_dm(rho, Xc, p)
            for l in range(self.L):
                rho = self.var_dm_layer(rho, params_np, l, p, p2, tol)
                if self.reupload and l < self.L - 1:
                    rho = self.encode_block_dm(rho, Xc, p)
            probs = torch.diagonal(rho, dim1=-2, dim2=-1).real
            out.append(probs @ self.feat_mat)
        return torch.cat(out, 0)


def circuit_stats(sim, params_np, tol=FREEZE_TOL):
    """Gate counts and depth of the deployed circuit (zeroed gates removed),
    correctly counting every re-uploaded encoding block and every one of the
    (now 3, was 1) single-qubit rotation gates per qubit per layer."""
    n = sim.n
    t = np.zeros(n)
    n1_var = n2 = n1_enc = 0

    def do_encoding():
        nonlocal n1_enc
        t[:] += 4.0
        n1_enc += 4 * n

    do_encoding()
    for l in range(sim.L):
        for kind, wires, (ll, i, j) in sim.layer_ops_list[l]:
            if abs(params_np[ll, i, j]) <= tol:
                continue
            if kind == "ZZ":
                n2 += 1
                a, b = wires
                t[a] = t[b] = max(t[a], t[b]) + 1
            else:
                n1_var += 1
                t[wires[0]] += 1
        if sim.reupload and l < sim.L - 1:
            do_encoding()
    return {"n_gates_1q_var": n1_var, "n_gates_1q_enc": n1_enc, "n_gates_2q": n2,
            "n_gates_total": n1_enc + n1_var + n2, "depth": int(t.max())}


# ----------------------------------------------------------------------
# Quantization (QAT) helpers -- unchanged from v11
# ----------------------------------------------------------------------
def wrap_angle(x):
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def fake_quantize_angles(params, bits):
    if bits is None:
        return params
    wrapped = wrap_angle(params)
    n_levels = (1 << (bits - 1)) - 1
    step = math.pi / max(n_levels, 1)
    with torch.no_grad():
        q = torch.clamp(torch.round(wrapped / step), -n_levels, n_levels) * step
    return wrapped + (q - wrapped).detach()


def quant_bits_at_step(t, start_step, end_step, initial_bits, final_bits, n_stages=4):
    if t <= start_step:
        return None
    if t >= end_step or n_stages <= 1:
        return final_bits
    progress = (t - start_step) / (end_step - start_step)
    stage = min(n_stages - 1, int(progress * n_stages))
    bits_now = initial_bits - round((initial_bits - final_bits) * stage / (n_stages - 1))
    return max(final_bits, int(bits_now))


# ----------------------------------------------------------------------
# Pruning helpers -- unchanged from v11 (all operate generically on flat
# boolean/float arrays, so they need no change for the 2->4 param-per-qubit
# ansatz expansion)
# ----------------------------------------------------------------------
def cubic_sparsity_at_step(t, start_step, end_step, final_sparsity, initial_sparsity=0.0):
    if t <= start_step:
        return initial_sparsity
    if t >= end_step:
        return final_sparsity
    progress = (t - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3


def rigl_cycle_fraction(t, start_step, end_step, zeta_init):
    if t <= start_step or t >= end_step or zeta_init <= 0:
        return 0.0
    progress = (t - start_step) / (end_step - start_step)
    return zeta_init * 0.5 * (1 + math.cos(math.pi * progress))


def entangling_protected_mask(n_layers, n_qubits, protect_frac, seed=0):
    """Only ever protects the ZZ entangler slot (index 0). The RZ-RY-RZ
    rotation slots (indices 1-3) are never protected -- unchanged intent
    from v11, just a bigger array."""
    mask = np.zeros((n_layers, n_qubits, 4), dtype=bool)
    n_live_ent = max(0, n_qubits - 1)
    n_protect = int(round(protect_frac * n_live_ent)) if n_live_ent > 0 else 0
    rng = np.random.RandomState(seed)
    for l in range(n_layers):
        if n_protect > 0:
            mask[l, rng.choice(n_live_ent, size=min(n_protect, n_live_ent), replace=False), 0] = True
    return mask.flatten()


def enforce_frozen(params, frozen_mask):
    if not frozen_mask.any():
        return
    with torch.no_grad():
        flat = params.view(-1)
        flat[torch.as_tensor(frozen_mask, device=params.device)] = 0.0


def drop_and_grow(frozen, protected, live, score, grad_ema, n_target, max_round_freeze, cycle_frac):
    frozen = frozen.copy()
    elig_frozen = np.where(frozen & live & ~protected)[0]
    n_grow = int(round(cycle_frac * elig_frozen.size))
    regrown = np.array([], dtype=int)
    if n_grow > 0:
        regrown = elig_frozen[np.argsort(-grad_ema[elig_frozen])][:n_grow]
        frozen[regrown] = False
    need = max(0, n_target - int((frozen & live).sum()))
    dropped = 0
    if need > 0:
        elig = np.setdiff1d(np.where(~frozen & live & ~protected)[0], regrown)
        if elig.size:
            take = elig[np.argsort(score[elig])][:min(need, max_round_freeze)]
            frozen[take] = True
            dropped = take.size
    return frozen, regrown.size, (regrown.size + dropped) > 0


def top_up(frozen, protected, live, score, n_target):
    need = n_target - int((frozen & live).sum())
    if need <= 0:
        return frozen
    rem = np.where(~frozen & live & ~protected)[0]
    frozen = frozen.copy()
    frozen[rem[np.argsort(score[rem])][:need]] = True
    return frozen


# ----------------------------------------------------------------------
# Metrics (unchanged from v11)
# ----------------------------------------------------------------------
def compute_auc(y_true, y_score):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        from scipy.stats import rankdata
        r = rankdata(y_score)
        n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
        return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_from_probs(p1, y):
    p1, y = np.asarray(p1, dtype=float), np.asarray(y)
    pc = np.clip(p1, 1e-7, 1 - 1e-7)
    return {"val_acc": float(np.mean((p1 > 0.5).astype(int) == y)),
            "val_auc": compute_auc(y, p1),
            "val_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))}


def check_convergence(hist, window_frac=0.15, rel_threshold=0.03):
    n = len(hist)
    w = max(5, int(round(n * window_frac)))
    if n < 2 * w:
        return True
    prev, last = float(np.mean(hist[-2 * w:-w])), float(np.mean(hist[-w:]))
    if prev <= 1e-9:
        return True
    return bool((prev - last) / prev < rel_threshold)


def bernoulli_kl_from_logits(z, p_t, eps=1e-7):
    p_t = p_t.clamp(eps, 1 - eps)
    return (p_t * (torch.log(p_t) + F.softplus(-z))
            + (1 - p_t) * (torch.log1p(-p_t) + F.softplus(z))).mean()


# ----------------------------------------------------------------------
# Experiment arms
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSpec:
    run_type: str
    sparsity: float
    quant: bool
    criterion: str = "movement"  # movement | magnitude | random
    regrow: bool = True
    distill: bool = False
    protect: bool = True


DENSE_TYPES = ("dense_fp", "dense_qat")


def build_arm_specs(sparsities, wanted=None):
    specs = [ArmSpec("dense_fp", 0.0, False, regrow=False),
             ArmSpec("dense_qat", 0.0, True, regrow=False)]
    per_s = [
        ("random_qat", dict(quant=True, criterion="random", regrow=False)),
        ("magnitude_qat", dict(quant=True, criterion="magnitude", regrow=False)),
        ("movement_qat", dict(quant=True, criterion="movement", regrow=False)),
        ("movement_rigl_qat", dict(quant=True, criterion="movement", regrow=True)),
        ("v12", dict(quant=True, criterion="movement", regrow=True, distill=True)),
        ("v12_fp", dict(quant=False, criterion="movement", regrow=True, distill=True)),
        ("v12_noprotect", dict(quant=True, criterion="movement", regrow=True, distill=True, protect=False)),
    ]
    for s in sparsities:
        for name, kw in per_s:
            specs.append(ArmSpec(name, s, **kw))
    if wanted:
        specs = [sp for sp in specs if sp.run_type in wanted or sp.run_type == "dense_fp"]
    return specs


# ----------------------------------------------------------------------
# Training (single code path for every arm)
# ----------------------------------------------------------------------
def forward_logits(sim, params_q, W, b, psi0, X):
    return sim.features_sv_forward(psi0, X, params_q) @ W + b


def train_arm(spec, sim, D, init, hp, teacher_probs, seed):
    dev = sim.device
    L, n = sim.L, sim.n
    n_params = L * n * 4  # NEW in v12: 4 slots/qubit/layer (1 ZZ + 3 rotation), was 2
    live = sim.live_mask.flatten()
    n_live = int(live.sum())
    protected = entangling_protected_mask(L, n, hp.protect_entangling_frac if spec.protect else 0.0, seed) & live

    params = torch.tensor(np.where(sim.live_mask, init["params"], 0.0), dtype=DTYPE, device=dev, requires_grad=True)
    W = torch.tensor(init["W"], dtype=DTYPE, device=dev, requires_grad=True)
    b = torch.tensor(init["b"], dtype=DTYPE, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [params], "lr": hp.lr}, {"params": [W, b], "lr": hp.lr_readout}])

    steps = hp.steps
    start_step = max(hp.win_sz, int(round(hp.prune_start_frac * steps)))
    end_step = max(start_step + hp.win_sz, int(round(hp.prune_end_frac * steps)))
    q_start = max(0, int(round(hp.quant_start_frac * steps)))
    q_end = max(q_start + 1, int(round(hp.quant_end_frac * steps)))
    max_round_freeze = max(1, int(math.ceil(hp.max_round_freeze_frac * n_live)))
    final_target = int(math.ceil(spec.sparsity * n_live))

    frozen = ~live
    y = D["y_train_t"]
    psi0_tr = D["psi_train"]
    X_tr = D["X_train_t"]
    # NEW in v12: `movement` is an EMA of (-theta*grad), not a running SUM over
    # all of training (v11). A sum is dominated by large, noisy pre-convergence
    # gradients from early steps; an EMA weights recent, more-informative steps
    # more heavily. See docstring point 2(a).
    movement = np.zeros(n_params)
    grad_ema = np.zeros(n_params)
    rng = np.random.RandomState(seed * 7919 + 13)
    bce_hist = []
    pruning = spec.sparsity > 0
    locked = not pruning
    last_change = -10 ** 9

    def wrapped_abs():
        return np.abs(np.remainder(params.detach().cpu().numpy().ravel() + math.pi, 2 * math.pi) - math.pi)

    def current_score():
        if spec.criterion == "movement":
            return movement
        if spec.criterion == "magnitude":
            return wrapped_abs()
        return rng.rand(n_params)

    def distill_loss(bce, logits, hard_weight):
        return (1 - hard_weight) * bce + hard_weight * bernoulli_kl_from_logits(logits, teacher_probs)

    t0 = time.time()
    for t in range(steps):
        cosf = hp.lr_min_frac + (1 - hp.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * t / steps))
        boost = 1.0 + (hp.lr_boost - 1.0) * max(0.0, 1.0 - (t - last_change) / hp.win_sz) if pruning else 1.0
        opt.param_groups[0]["lr"] = hp.lr * cosf * boost
        opt.param_groups[1]["lr"] = hp.lr_readout * cosf

        bits_now = (quant_bits_at_step(t, q_start, q_end, hp.quant_initial_bits, hp.quant_bits, hp.quant_stages)
                    if spec.quant else None)
        qp = fake_quantize_angles(params, bits_now)
        theta_before = qp.detach().cpu().numpy().ravel()

        opt.zero_grad(set_to_none=True)
        logits = forward_logits(sim, qp, W, b, psi0_tr, X_tr)
        bce = F.binary_cross_entropy_with_logits(logits, y)
        loss = bce
        if spec.distill and teacher_probs is not None:
            # NEW in v12: distillation weight scales with how much of the
            # sparsity target has actually been reached, not with wall-clock
            # time (v11). Little distillation before pruning has done any
            # damage; more as more parameters get zeroed. See docstring
            # point 2(b). For dense/never-pruned arms (final_target==0) this
            # just uses distill_hardness_min throughout, unchanged in spirit.
            prog = 0.0 if final_target <= 0 else min(1.0, float((frozen & live).sum()) / final_target)
            hard = hp.distill_hardness_min + (hp.distill_hardness_max - hp.distill_hardness_min) * prog
            loss = distill_loss(bce, logits, hard)
        loss.backward()
        bce_hist.append(float(bce.item()))

        g = params.grad.detach().cpu().numpy().ravel()
        beta_m = hp.movement_score_ema_beta
        movement = beta_m * movement + (1 - beta_m) * (-(theta_before * g))
        grad_ema = hp.movement_ema_beta * grad_ema + (1 - hp.movement_ema_beta) * np.abs(g)

        with torch.no_grad():
            params.grad.view(-1)[torch.as_tensor(frozen, device=dev)] = 0.0
        opt.step()
        enforce_frozen(params, frozen)

        if pruning and not locked:
            if t != 0 and t % hp.win_sz == 0:
                target_now = cubic_sparsity_at_step(t, start_step, end_step, spec.sparsity)
                n_target = int(math.ceil(target_now * n_live))
                cyc = rigl_cycle_fraction(t, start_step, end_step, hp.cycle_frac_init) if spec.regrow else 0.0
                frozen, _, changed = drop_and_grow(frozen, protected, live, current_score(), grad_ema,
                                                    n_target, max_round_freeze, cyc)
                if changed:
                    last_change = t
                enforce_frozen(params, frozen)
            if t >= end_step:
                frozen = top_up(frozen, protected, live, current_score(), final_target)
                enforce_frozen(params, frozen)
                locked = True
                last_change = t

    if not locked:
        frozen = top_up(frozen, protected, live, current_score(), final_target)
        enforce_frozen(params, frozen)

    final_bits = hp.quant_bits if spec.quant else None

    # NEW in v12: adaptive convergence extension. v11 shipped 160/368 groups
    # that had not converged in the fixed step budget and told the user to
    # globally raise --steps. Here, each individual seed gets extra
    # win_sz-sized fine-tuning chunks at a small constant LR (the pruning
    # mask is already locked at this point for every pruned arm, so this
    # only refines surviving parameters) until it converges on its own, or
    # --max-extra-steps is exhausted. See docstring point 3.
    extra_steps_used = 0
    converged = check_convergence(bce_hist, hp.convergence_window_frac, hp.convergence_rel_threshold)
    fine_lr = hp.lr * hp.lr_min_frac
    fine_lr_ro = hp.lr_readout * hp.lr_min_frac
    while (not converged) and extra_steps_used < hp.max_extra_steps:
        chunk = min(hp.win_sz, hp.max_extra_steps - extra_steps_used)
        for _ in range(chunk):
            qp = fake_quantize_angles(params, final_bits)
            opt.param_groups[0]["lr"] = fine_lr
            opt.param_groups[1]["lr"] = fine_lr_ro
            opt.zero_grad(set_to_none=True)
            logits = forward_logits(sim, qp, W, b, psi0_tr, X_tr)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            loss = bce
            if spec.distill and teacher_probs is not None:
                loss = distill_loss(bce, logits, hp.distill_hardness_min)
            loss.backward()
            bce_hist.append(float(bce.item()))
            with torch.no_grad():
                params.grad.view(-1)[torch.as_tensor(frozen, device=dev)] = 0.0
            opt.step()
            enforce_frozen(params, frozen)
        extra_steps_used += chunk
        converged = check_convergence(bce_hist, hp.convergence_window_frac, hp.convergence_rel_threshold)

    with torch.no_grad():
        fq = fake_quantize_angles(params, final_bits)
        logits_tr = forward_logits(sim, fq, W, b, psi0_tr, X_tr)
        logits_ev = forward_logits(sim, fq, W, b, D["psi_eval"], D["X_eval_t"])
    train_p = torch.sigmoid(logits_tr)
    fq_np = fq.detach().cpu().numpy()
    return {
        "params_q": fq_np, "W": W.detach().cpu().numpy(), "b": b.detach().cpu().numpy(),
        "train_probs": train_p.detach(), "train_acc": float(((train_p > 0.5).double() == y).double().mean().item()),
        "train_bce": float(F.binary_cross_entropy_with_logits(logits_tr, y).item()),
        "eval_probs_noiseless": torch.sigmoid(logits_ev).cpu().numpy(),
        "converged": converged, "extra_steps_used": extra_steps_used,
        "final_bits": final_bits, "time": time.time() - t0, "n_live": n_live,
    }


def evaluate_arm(sim, D, res, noise_levels, twoq_mults):
    """Every noise level (incl. 0) is scored on the SAME full eval set,
    swept over twoq_mults (the 2q/1q depolarizing noise ratio). noise=0 is
    mult-invariant, so it is recorded once under the first (base) mult only,
    to avoid duplicate identical rows. Unchanged from v11."""
    base_mult = twoq_mults[0]
    out = {(0.0, base_mult): metrics_from_probs(res["eval_probs_noiseless"], D["y_eval"])}
    W = torch.tensor(res["W"], dtype=DTYPE, device=sim.device)
    b = torch.tensor(res["b"], dtype=DTYPE, device=sim.device)
    with torch.no_grad():
        for p in sorted({x for x in noise_levels if x > 0}):
            for mult in twoq_mults:
                feats = sim.features_noisy(D["X_eval_t"], res["params_q"], p, mult)
                out[(p, mult)] = metrics_from_probs(torch.sigmoid(feats @ W + b).cpu().numpy(), D["y_eval"])
    return out


# ----------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------
RECORD_FIELDS = [
    "dataset", "n_qubits", "n_layers", "seed", "run_type", "criterion", "target_sparsity",
    "quant_bits", "achieved_sparsity_live", "n_params_live", "n_params_active",
    "n_gates_1q_var", "n_gates_1q_enc", "n_gates_2q", "n_gates_total", "depth", "model_bits",
    "compression_vs_fp32", "converged", "extra_steps_used", "train_acc", "train_bce",
    "noise_level", "twoq_noise_mult", "val_acc", "val_auc", "val_loss", "eval_split", "train_time_s",
]
SANITY_FIELDS = ["dataset", "n_qubits", "seed", "logreg_eval_acc"]
_INT = {"n_qubits", "n_layers", "seed", "n_params_live", "n_params_active", "n_gates_1q_var",
        "n_gates_1q_enc", "n_gates_2q", "n_gates_total", "depth", "model_bits", "extra_steps_used"}
_FLOAT = {"target_sparsity", "achieved_sparsity_live", "compression_vs_fp32", "train_acc", "train_bce",
          "noise_level", "twoq_noise_mult", "val_acc", "val_auc", "val_loss", "train_time_s"}


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def read_records(path):
    recs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = dict(row)
            for k in _INT:
                r[k] = int(r[k])
            for k in _FLOAT:
                r[k] = float(r[k])
            r["converged"] = r["converged"] == "True"
            r["quant_bits"] = None if r["quant_bits"] == "" else int(float(r["quant_bits"]))
            recs.append(r)
    return recs


def read_sanity(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return [{"dataset": r["dataset"], "n_qubits": int(r["n_qubits"]), "seed": int(r["seed"]),
                 "logreg_eval_acc": float(r["logreg_eval_acc"])} for r in csv.DictReader(f)]


def make_rows(dataset, sim, seed, spec, res, ev, hp):
    st = circuit_stats(sim, res["params_q"])
    live = sim.live_mask
    n_live = int(live.sum())
    n_active = int(np.count_nonzero(res["params_q"][live]))
    bits = hp.quant_bits if spec.quant else 32
    sparse_arm = spec.sparsity > 0
    model_bits = n_active * bits + (n_live if sparse_arm else 0)
    base = {
        "dataset": dataset, "n_qubits": sim.n, "n_layers": sim.L, "seed": seed, "run_type": spec.run_type,
        "criterion": spec.criterion if sparse_arm else "", "target_sparsity": spec.sparsity,
        "quant_bits": res["final_bits"], "achieved_sparsity_live": 1.0 - n_active / n_live,
        "n_params_live": n_live, "n_params_active": n_active, **st, "model_bits": model_bits,
        "compression_vs_fp32": (n_live * 32) / max(1, model_bits), "converged": res["converged"],
        "extra_steps_used": res["extra_steps_used"],
        "train_acc": res["train_acc"], "train_bce": res["train_bce"], "eval_split": hp.eval_split,
        "train_time_s": res["time"],
    }
    return [dict(base, noise_level=p, twoq_noise_mult=mult, **m) for (p, mult), m in sorted(ev.items())]


def run_combo(dataset, n_qubits, seed, hp, device, sim_cache):
    data = load_binary_subset(dataset, n_qubits, hp.n_train, hp.n_val, seed, hp.eval_split, hp.data_root)
    key = (n_qubits, hp.layers, hp.reupload)
    if key not in sim_cache:
        sim_cache[key] = QSim(n_qubits, hp.layers, hp.feature_mode, device, reupload=hp.reupload)
    sim = sim_cache[key]

    D = {"y_eval": data["y_eval"]}
    D["X_eval_t"] = torch.tensor(data["X_eval"], dtype=DTYPE, device=device)
    D["X_train_t"] = torch.tensor(data["X_train"], dtype=DTYPE, device=device)
    D["y_train_t"] = torch.tensor(data["y_train"], dtype=DTYPE, device=device)
    with torch.no_grad():
        D["psi_train"] = sim.encode_sv(D["X_train_t"])
        D["psi_eval"] = sim.encode_sv(D["X_eval_t"])

    rs = np.random.RandomState(seed)
    init = {"params": rs.uniform(-hp.init_scale, hp.init_scale, size=(hp.layers, n_qubits, 4))}
    rr = np.random.RandomState(seed + 10_000)
    init["W"] = rr.uniform(-1.0, 1.0, size=sim.n_features) / math.sqrt(sim.n_features)
    init["b"] = np.zeros(())

    sanity = None
    if not hp.skip_sanity_check:
        acc = classical_sanity_baseline(data["F_train"], data["y_train"], data["F_eval"], data["y_eval"])
        if acc is not None:
            sanity = {"dataset": dataset, "n_qubits": n_qubits, "seed": seed, "logreg_eval_acc": acc}

    specs = build_arm_specs(hp.sparsities, hp.arms)
    teacher = train_arm(specs[0], sim, D, init, hp, None, seed)
    rows = []
    for spec in specs:
        res = teacher if spec.run_type == "dense_fp" else train_arm(
            spec, sim, D, init, hp, teacher["train_probs"], seed)
        ev = evaluate_arm(sim, D, res, hp.noise_levels, hp.twoq_noise_mults)
        rows.extend(make_rows(dataset, sim, seed, spec, res, ev, hp))
    return rows, sanity


# ----------------------------------------------------------------------
# Statistics (unchanged from v11)
# ----------------------------------------------------------------------
def t_crit(df, q=0.975):
    try:
        from scipy.stats import t
        return float(t.ppf(q, df))
    except Exception:
        return {0.975: 1.96, 0.80: 0.84}.get(round(q, 3), 1.28)


def mean_std_ci(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), float("nan")
    n = len(v)
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return m, s, (t_crit(n - 1) * s / math.sqrt(n) if n > 1 else 0.0)


AGG_METRICS = ("val_acc", "val_auc", "val_loss")
AGG_EXTRA = ("achieved_sparsity_live", "n_params_active", "n_gates_2q", "n_gates_1q_enc", "n_gates_total",
             "depth", "compression_vs_fp32")
RUN_ORDER = {"dense_fp": 0, "dense_qat": 1, "random_qat": 2, "magnitude_qat": 3, "movement_qat": 4,
             "movement_rigl_qat": 5, "v12": 6, "v12_fp": 7, "v12_noprotect": 8}


def aggregate_records(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["dataset"], r["n_qubits"], r["run_type"], r["target_sparsity"], r["noise_level"],
                 r["twoq_noise_mult"])].append(r)
    agg = []
    for (ds, q, rt, sp, nl, mult), rs in groups.items():
        row = {"dataset": ds, "n_qubits": q, "run_type": rt, "target_sparsity": sp, "noise_level": nl,
               "twoq_noise_mult": mult, "n_seeds": len(rs), "converged_frac": float(np.mean([r["converged"] for r in rs]))}
        for m in AGG_METRICS:
            row[m + "_mean"], row[m + "_std"], row[m + "_ci95"] = mean_std_ci([r[m] for r in rs])
        for m in AGG_EXTRA:
            row[m + "_mean"] = float(np.mean([r[m] for r in rs]))
        agg.append(row)
    agg.sort(key=lambda r: (r["dataset"], r["n_qubits"], RUN_ORDER.get(r["run_type"], 99),
                            r["target_sparsity"], r["noise_level"], r["twoq_noise_mult"]))
    return agg


CONTRASTS = [
    ("quantization_cost", "dense_qat", "dense_fp"),
    ("v12_vs_dense_fp", "v12", "dense_fp"),
    ("pruning_cost_given_qat", "v12", "dense_qat"),
    ("v12_vs_random", "v12", "random_qat"),
    ("v12_vs_magnitude", "v12", "magnitude_qat"),
    ("v12_vs_movement_only", "v12", "movement_qat"),
    ("regrow_effect", "movement_rigl_qat", "movement_qat"),
    ("distill_effect", "v12", "movement_rigl_qat"),
    ("quant_effect_in_pruned", "v12", "v12_fp"),
    ("protect_effect", "v12", "v12_noprotect"),
]
# gate-count-matched contrasts (kept from v11 -- see module docstring)
GATE_MATCH_REFS = [
    ("v12_vs_random_gatematched", "v12", "random_qat"),
    ("v12_vs_magnitude_gatematched", "v12", "magnitude_qat"),
]
# NEW in v12: the contrasts whose mde80 drives the adaptive seed-budget loop
# in main(). These are the ones the v11 postmortem cared about most: does the
# fancy criterion actually beat trivial baselines at matched circuit cost.
KEY_CONTRASTS = {"v12_vs_random_gatematched", "v12_vs_magnitude_gatematched"}


def paired_pvalue(diffs, rng, n_perm=20000):
    d = np.asarray(diffs, dtype=float)
    if len(d) < MIN_SEEDS_FOR_TEST:
        return None, "insufficient_seeds"
    if np.allclose(d, 0.0):
        return 1.0, "all_zero"
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue), "wilcoxon"
    except Exception:
        obs = abs(d.mean())
        signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
        return float((np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-15) * n_perm + 1) / (n_perm + 1)), "sign_flip"


def holm_adjust(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    adj = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def summarize_diffs(d, rng, n_boot, ni_margin):
    n = len(d)
    boot = d[rng.randint(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p5 = float(np.percentile(boot, 5.0))
    p, meth = paired_pvalue(d, rng)
    s = float(np.std(d, ddof=1)) if n > 1 else 0.0
    df = max(1, n - 1)
    mde80 = (t_crit(df, 0.975) + t_crit(df, 0.80)) * s / math.sqrt(n) if n > 1 else float("nan")
    return {"n_seeds": n, "diff_mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": p, "test_method": meth, "noninferior": p5 > -ni_margin, "mde80": mde80}


def compute_contrast_table(records, metric, higher_is_better, ni_margin, n_boot, alpha=0.05):
    vals = defaultdict(dict)
    for r in records:
        v = r[metric]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals[(r["dataset"], r["n_qubits"], r["run_type"], r["target_sparsity"], r["noise_level"],
              r["twoq_noise_mult"])][r["seed"]] = v
    datasets = sorted({(r["dataset"], r["n_qubits"]) for r in records})
    sparsities = sorted({r["target_sparsity"] for r in records if r["target_sparsity"] > 0})
    noises = sorted({r["noise_level"] for r in records})
    mults = sorted({r["twoq_noise_mult"] for r in records})
    rng = np.random.RandomState(12345)
    rows = []
    for (ds, q) in datasets:
        for cname, a, b in CONTRASTS:
            fam = []
            sp_list = [0.0] if (a in DENSE_TYPES and b in DENSE_TYPES) else sparsities
            for s in sp_list:
                sa = 0.0 if a in DENSE_TYPES else s
                sb = 0.0 if b in DENSE_TYPES else s
                for nl in noises:
                    for mult in mults:
                        va, vb = vals.get((ds, q, a, sa, nl, mult)), vals.get((ds, q, b, sb, nl, mult))
                        if not va or not vb:
                            continue
                        seeds = sorted(set(va) & set(vb))
                        if not seeds:
                            continue
                        d = np.array([(va[k] - vb[k]) if higher_is_better else (vb[k] - va[k]) for k in seeds])
                        if len(d) < MIN_SEEDS_FOR_TEST:
                            continue
                        stat = summarize_diffs(d, rng, n_boot, ni_margin)
                        fam.append({"dataset": ds, "n_qubits": q, "contrast": cname, "arm": a, "reference": b,
                                    "target_sparsity": s, "noise_level": nl, "twoq_noise_mult": mult, **stat})
            adj = holm_adjust([r["p_value"] for r in fam])
            for r, pa in zip(fam, adj):
                r["p_holm"] = pa
                r["significant_holm"] = (pa is not None) and (pa < alpha)
                rows.append(r)
    return rows


def compute_gatematched_contrasts(records, metric, higher_is_better, ni_margin, n_boot, alpha=0.05):
    idx = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if r["target_sparsity"] <= 0:
            continue
        v = r[metric]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        key = (r["dataset"], r["n_qubits"], r["run_type"], r["noise_level"], r["twoq_noise_mult"])
        idx[key][r["seed"]][r["target_sparsity"]] = (r["n_gates_total"], v)
    datasets = sorted({(r["dataset"], r["n_qubits"]) for r in records})
    sparsities = sorted({r["target_sparsity"] for r in records if r["target_sparsity"] > 0})
    noises = sorted({r["noise_level"] for r in records})
    mults = sorted({r["twoq_noise_mult"] for r in records})
    rng = np.random.RandomState(24680)
    rows = []
    for (ds, q) in datasets:
        for cname, a, b in GATE_MATCH_REFS:
            fam = []
            for nl in noises:
                for mult in mults:
                    per_a = idx.get((ds, q, a, nl, mult), {})
                    per_b = idx.get((ds, q, b, nl, mult), {})
                    for sp in sparsities:
                        diffs = []
                        for seed in sorted(set(per_a) & set(per_b)):
                            pts_a, pts_b = per_a[seed], per_b[seed]
                            if sp not in pts_a or len(pts_b) < 2:
                                continue
                            gcount_a, val_a = pts_a[sp]
                            ordered = sorted(pts_b.items(), key=lambda kv: kv[1][0])
                            gcs = [v[0] for _, v in ordered]
                            vs = [v[1] for _, v in ordered]
                            interp_val = float(np.interp(gcount_a, gcs, vs))
                            d = (val_a - interp_val) if higher_is_better else (interp_val - val_a)
                            diffs.append(d)
                        if len(diffs) < MIN_SEEDS_FOR_TEST:
                            continue
                        stat = summarize_diffs(np.asarray(diffs), rng, n_boot, ni_margin)
                        fam.append({"dataset": ds, "n_qubits": q, "contrast": cname, "arm": a, "reference": b,
                                    "target_sparsity": sp, "noise_level": nl, "twoq_noise_mult": mult, **stat})
            adj = holm_adjust([r["p_value"] for r in fam])
            for r, pa in zip(fam, adj):
                r["p_holm"] = pa
                r["significant_holm"] = (pa is not None) and (pa < alpha)
                rows.append(r)
    return rows


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def fmt_dur(sec):
    return str(timedelta(seconds=round(sec, 2)))


def _write_contrast_table(f, title, tab, ni_margin):
    f.write("=" * 100 + f"\n{title}\n" + "=" * 100 + "\n")
    f.write(f"{'dataset':<13}{'q':<3}{'contrast':<31}{'spars':<7}{'noise':<7}{'mult':<6}{'n':<4}{'diff':<9}"
            f"{'ci_lo':<9}{'ci_hi':<9}{'mde80':<8}{'p_raw':<8}{'p_holm':<8}{'sig':<5}{'NI':<4}\n")
    for r in tab:
        pr = "n/a" if r["p_value"] is None else f"{r['p_value']:.4f}"
        ph = "n/a" if r["p_holm"] is None else f"{r['p_holm']:.4f}"
        ni = "" if r["noninferior"] is None else ("yes" if r["noninferior"] else "no")
        f.write(f"{r['dataset']:<13}{r['n_qubits']:<3}{r['contrast']:<31}{r['target_sparsity']:<7.2f}"
                f"{r['noise_level']:<7.3f}{r['twoq_noise_mult']:<6.1f}{r['n_seeds']:<4}{r['diff_mean']:<+9.4f}"
                f"{r['ci_lo']:<+9.4f}{r['ci_hi']:<+9.4f}{r['mde80']:<8.4f}{pr:<8}{ph:<8}"
                f"{'YES' if r['significant_holm'] else 'no':<5}{ni:<4}\n")
    f.write(f"NI = non-inferior at margin {ni_margin}. mde80 = smallest true effect this cell's n_seeds/variance\n"
            "could detect at 80% power, alpha=0.05 two-sided -- a non-significant row with small mde80 is\n"
            "evidence of a small true effect; with large mde80 it is just underpowered, not evidence of none.\n\n")


def write_report(path, hp, records, agg, sanity_rows, contrasts, gatematched, elapsed, final_n_seeds):
    base_mult = hp.twoq_noise_mults[0]
    agg_base = [r for r in agg if r["twoq_noise_mult"] == base_mult]
    with open(path, "w") as f:
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("Algorithm: QAdaPrune-RigL-Q (v12) -- v11's re-uploading + gate-count-matched contrasts + "
                "mde80 + swept noise ratio, PLUS: RZ-RY-RZ (universal single-qubit) variational rotation "
                "instead of RY-only (directly targets the capacity gap re-uploading alone didn't close), "
                "an EMA'd movement-pruning score instead of a raw sum (v11's criterion lost to plain "
                "magnitude pruning at matched gate count), sparsity-progress-scaled distillation with a "
                "lower max weight (v11's own distill_effect trended negative), adaptive per-seed "
                "convergence extension (v11 shipped 160/368 unconverged groups), and an adaptive seed-"
                "budget loop that keeps adding seeds until the key gate-matched contrasts hit a target "
                "mde80 instead of stopping at a fixed, frequently-underpowered n.\n")
        f.write("Run config:\n")
        for k, v in vars(hp).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal sweep run time: {fmt_dur(elapsed)}\nTotal records: {len(records)}\n")
        f.write(f"Final seed count (adaptive; started at {len(hp.seeds)}): {final_n_seeds}\n\n")

        gaps = {}
        if sanity_rows:
            by = defaultdict(list)
            for r in sanity_rows:
                by[(r["dataset"], r["n_qubits"])].append(r["logreg_eval_acc"])
            dense = {(a["dataset"], a["n_qubits"]): a["val_acc_mean"] for a in agg_base
                     if a["run_type"] == "dense_fp" and a["noise_level"] == 0.0}
            for (ds, q), v in sorted(by.items()):
                m, _, _ = mean_std_ci(v)
                dq = dense.get((ds, q), float("nan"))
                gaps[(ds, q)] = m - dq

        gap_flags = [(ds, q, g) for (ds, q), g in gaps.items() if g > hp.gap_warn_threshold]
        if gap_flags:
            f.write("!" * 100 + "\nCAPACITY WARNING: the deployed circuit remains more than "
                    f"{hp.gap_warn_threshold:.2f} accuracy below the classical (logistic regression, same "
                    "PCA features) ceiling for:\n")
            for ds, q, g in gap_flags:
                f.write(f"  {ds} q={q}: gap={g:+.4f}\n")
            f.write("Every pruning/quantization/noise comparison below is still a valid comparison BETWEEN "
                    "quantum arms, but none of it is evidence that this circuit is a good classifier in "
                    "absolute terms. RZ-RY-RZ (v12) should have narrowed this vs v11 (RY-only); if the gap "
                    "is still this large, look at --layers, --steps, and --init-scale before trusting "
                    "anything else in this report.\n" + "!" * 100 + "\n\n")

        if sanity_rows:
            f.write("=" * 100 + "\nCLASSICAL SANITY BASELINE (logistic regression, same PCA features) -- "
                    "CHECK THIS FIRST\n" + "=" * 100 + "\n")
            for (ds, q), g in sorted(gaps.items()):
                dq = next(a["val_acc_mean"] for a in agg_base if a["run_type"] == "dense_fp"
                          and a["noise_level"] == 0.0 and a["dataset"] == ds and a["n_qubits"] == q)
                f.write(f"{ds:<14} q={q:<3} logreg={dq + g:.4f}   dense_fp circuit={dq:.4f}   gap={g:+.4f}"
                        f"   (reupload={hp.reupload})\n")
            f.write("A large positive gap means the quantum pipeline (not the data) is still the bottleneck. "
                    "Pruning conclusions are only meaningful to the extent this gap is small.\n\n")

        f.write("=" * 100 + f"\nVALIDATION METRICS at twoq_noise_mult={base_mult} (mean +/- 95% t-CI across "
                "seeds; every noise level on the same eval set; other noise ratios appear only in the "
                "contrast tables below)\n" + "=" * 100 + "\n")
        f.write(f"{'dataset':<13}{'q':<3}{'run_type':<19}{'spars':<7}{'noise':<7}{'n':<4}"
                f"{'acc':<8}{'ci':<8}{'auc':<8}{'ci':<8}{'loss':<8}{'conv%':<6}\n")
        for r in agg_base:
            f.write(f"{r['dataset']:<13}{r['n_qubits']:<3}{r['run_type']:<19}{r['target_sparsity']:<7.2f}"
                    f"{r['noise_level']:<7.3f}{r['n_seeds']:<4}{r['val_acc_mean']:<8.4f}{r['val_acc_ci95']:<8.4f}"
                    f"{r['val_auc_mean']:<8.4f}{r['val_auc_ci95']:<8.4f}{r['val_loss_mean']:<8.4f}"
                    f"{100 * r['converged_frac']:<6.0f}\n")
        n_nc = sum(1 for r in agg_base if r["converged_frac"] < 1.0)
        f.write(f"\nNote: 'converged' here reflects the FINAL state after adaptive extension (up to "
                f"{hp.max_extra_steps} extra steps per seed), not just the fixed {hp.steps}-step budget.\n")
        f.write(f"{n_nc}/{len(agg_base)} groups still had a seed that did not converge even after "
                f"extension -- raise --max-extra-steps for those.\n\n")

        f.write("=" * 100 + f"\nEFFICIENCY OF THE DEPLOYED CIRCUIT (noise=0, twoq_noise_mult={base_mult}; means "
                "over seeds)\n" + "=" * 100 + "\n")
        f.write(f"{'dataset':<13}{'q':<3}{'run_type':<19}{'spars':<7}{'ach.sp':<8}{'active':<8}{'2q':<7}"
                f"{'enc1q':<7}{'gates':<8}{'depth':<7}{'compress':<9}\n")
        for r in agg_base:
            if r["noise_level"] != 0.0:
                continue
            f.write(f"{r['dataset']:<13}{r['n_qubits']:<3}{r['run_type']:<19}{r['target_sparsity']:<7.2f}"
                    f"{r['achieved_sparsity_live_mean']:<8.3f}{r['n_params_active_mean']:<8.1f}"
                    f"{r['n_gates_2q_mean']:<7.1f}{r['n_gates_1q_enc_mean']:<7.1f}"
                    f"{r['n_gates_total_mean']:<8.1f}{r['depth_mean']:<7.1f}{r['compression_vs_fp32_mean']:<9.2f}\n")
        f.write("ach.sp = fraction of LIVE params exactly 0 in the deployed circuit. enc1q = 1-qubit encoding "
                "gates (never pruned; counted once per re-upload). 1q variational gates now include the "
                "RZ-RY-RZ triple per qubit per layer (3x v11's count at the same target sparsity). compress = "
                "(live params * 32 bit) / (active * bits + 1-bit mask).\n\n")

        for metric, tab in contrasts.items():
            _write_contrast_table(
                f, f"PAIRED CONTRASTS (sparsity-matched) on {metric}: diff = (arm - reference), positive = "
                f"arm better. CI = bootstrap 95%; p = Wilcoxon, Holm-adjusted within each (dataset, qubits, "
                f"contrast) family; NI = non-inferior (lower CI bound > -{hp.ni_margin}).",
                tab, hp.ni_margin)
        for metric, tab in gatematched.items():
            _write_contrast_table(
                f, f"GATE-COUNT-MATCHED CONTRASTS on {metric}: v12's metric at its achieved gate count vs. "
                f"the reference criterion's metric LINEARLY INTERPOLATED to that same gate count. These two "
                f"({', '.join(sorted(KEY_CONTRASTS))}) drove the adaptive seed budget above -- their mde80 "
                f"at noise=0 should now be <= {hp.target_mde80} unless --max-seeds was hit first.",
                tab, hp.ni_margin)

        f.write("How to read this report: only Holm-significant rows are evidence of a difference; only NI=yes "
                "rows are evidence of (near-)equivalence at the stated margin; mde80 tells you whether a "
                "non-significant row was actually underpowered. v12_vs_random / v12_vs_magnitude (sparsity-"
                "matched) test whether the movement+RigL machinery beats trivial pruning at the same nominal "
                "sparsity; the *_gatematched versions test the same question at matched circuit cost, which is "
                "the more meaningful comparison under noise, and are the ones the adaptive seed budget targets. "
                "pruning_cost_given_qat isolates pruning from quantization; quantization_cost isolates "
                "quantization. Two-qubit-noise-ratio robustness: compare the mult column across rows of the "
                "same contrast to see whether an effect strengthens under the more realistic (higher-mult) "
                "noise model.\n")
        f.write("See *_results.csv, *_aggregate.csv, *_contrasts_<metric>.csv, *_gatematched_<metric>.csv, "
                "*_sanity_baseline.csv.\n")


# ----------------------------------------------------------------------
# Self-test against PennyLane -- updated for the RZ-RY-RZ variational gate
# ----------------------------------------------------------------------
def selftest():
    import pennylane as qml
    dev_t = torch.device("cpu")
    n, L = 4, 2
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, (6, n, 4))
    P = rng.uniform(-np.pi, np.pi, (L, n, 4))
    P[0, 0, 1] = 0.0
    P[1, 1, 0] = 0.0  # exercise the "zeroed gate is skipped" path
    Xt = torch.tensor(X, dtype=DTYPE)
    Pt = torch.tensor(P, dtype=DTYPE)
    ok = True

    def encode_pl(x, noise):
        for q in range(n):
            for k, g in enumerate((qml.RY, qml.RX, qml.RZ, qml.RY)):
                g(float(x[q, k]), wires=q)
                if noise > 0:
                    qml.DepolarizingChannel(noise, wires=q)

    def layer_pl(l, noise):
        for i in range(n - 1):
            th = float(P[l, i, 0])
            if abs(th) > FREEZE_TOL:
                qml.IsingZZ(th, wires=[i, i + 1])
                if noise > 0:
                    qml.DepolarizingChannel(noise, wires=i)
                    qml.DepolarizingChannel(noise, wires=i + 1)
        for i in range(n):
            for k, kind in enumerate(ROT_ORDER):
                th = float(P[l, i, 1 + k])
                if abs(th) > FREEZE_TOL:
                    (qml.RZ if kind == "RZ" else qml.RY)(th, wires=i)
                    if noise > 0:
                        qml.DepolarizingChannel(noise, wires=i)

    def build(x, noise, reupload):
        encode_pl(x, noise)
        for l in range(L):
            layer_pl(l, noise)
            if reupload and l < L - 1:
                encode_pl(x, noise)
        return ([qml.expval(qml.PauliZ(i)) for i in range(n)]
                + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(n - 1)])

    for reupload in (False, True):
        sim = QSim(n, L, "z+zz", dev_t, reupload=reupload)
        psi0 = sim.encode_sv(Xt)
        ours = sim.features_sv_forward(psi0, Xt, Pt).numpy()
        ref_sv = qml.QNode(lambda x, ru=reupload: build(x, 0.0, ru), qml.device("default.qubit", wires=n))
        ref = np.array([[float(v) for v in ref_sv(x)] for x in X])
        e = np.abs(ours - ref).max()
        print(f"[reupload={reupload}] statevector vs PennyLane default.qubit: max abs err = {e:.2e}")
        ok &= e < 1e-9
        for p in (0.03, 0.1):
            ref_dm = qml.QNode(lambda x, p=p, ru=reupload: build(x, p, ru), qml.device("default.mixed", wires=n))
            ours = sim.features_noisy(Xt, P, p).numpy()
            ref = np.array([[float(v) for v in ref_dm(x)] for x in X])
            e = np.abs(ours - ref).max()
            print(f"[reupload={reupload}] density matrix p={p} vs PennyLane default.mixed: max abs err = {e:.2e}")
            ok &= e < 1e-9

    # Autograd vs finite differences through the UNQUANTIZED circuit.
    sim = QSim(n, L, "z+zz", dev_t, reupload=True)
    Wt = torch.tensor(rng.randn(sim.n_features), dtype=DTYPE)
    y = torch.tensor(rng.randint(0, 2, 6), dtype=DTYPE)
    psi0 = sim.encode_sv(Xt)

    def loss_fn(pp):
        feats = sim.features_sv_forward(psi0, Xt, pp)
        return F.binary_cross_entropy_with_logits(feats @ Wt, y)
    pp = Pt.clone().requires_grad_(True)
    loss_fn(pp).backward()
    g = pp.grad.numpy()
    fd = np.zeros_like(P)
    eps = 1e-6
    for idx in np.ndindex(*P.shape):
        if not sim.live_mask[idx]:
            continue
        a, b = Pt.clone(), Pt.clone()
        a[idx] += eps
        b[idx] -= eps
        fd[idx] = (loss_fn(a).item() - loss_fn(b).item()) / (2 * eps)
    e = np.abs(g - fd).max()
    print(f"autograd vs finite-difference gradient, unquantized (reupload=True): max abs err = {e:.2e} "
          f"(dead-slot grads: {np.abs(g[~sim.live_mask]).max():.1e})")
    ok &= e < 1e-6

    theta = torch.tensor(0.377, dtype=DTYPE, requires_grad=True)  # away from any 6-bit grid boundary
    fake_quantize_angles(theta, 6).backward()
    ste_err = abs(theta.grad.item() - 1.0)
    print(f"straight-through estimator gradient in a flat grid cell: |grad - 1| = {ste_err:.2e}")
    ok &= ste_err < 1e-9
    ok &= float(fake_quantize_angles(torch.zeros(1, dtype=DTYPE), 6).item()) == 0.0
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="QAdaPrune-RigL-Q v12")
    p.add_argument("-s", "--save", default="qadaprune_v12_run")
    p.add_argument("--report-name", default="exp12.txt")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--datasets", nargs="+", default=["mnist", "fashionmnist"], choices=list(DATASETS))
    p.add_argument("--qubits", type=int, nargs="+", default=[4, 6])
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)),
                    help="starting seed pool; more may be added automatically, see --adaptive-seeds")
    p.add_argument("--sparsities", type=float, nargs="+", default=[0.10, 0.25, 0.40])
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    p.add_argument("--twoq-noise-mults", type=float, nargs="+", default=[1.0, 6.0],
                    help="2q/1q depolarizing-noise ratio(s) to sweep; index 0 is the 'base' ratio used for the "
                         "main VALIDATION METRICS/EFFICIENCY tables")
    p.add_argument("--no-reupload", dest="reupload", action="store_false", default=True,
                    help="disable data re-uploading")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=800)
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--feature-mode", default="z+zz", choices=["z", "z+zz"])
    p.add_argument("--init-scale", type=float, default=math.pi)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--win-sz", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr-readout", type=float, default=0.05)
    p.add_argument("--lr-min-frac", type=float, default=0.02)
    p.add_argument("--lr-boost", type=float, default=1.5)
    p.add_argument("--prune-start-frac", type=float, default=0.10)
    p.add_argument("--prune-end-frac", type=float, default=0.60)
    p.add_argument("--max-round-freeze-frac", type=float, default=0.10)
    p.add_argument("--movement-ema-beta", type=float, default=0.6,
                    help="EMA beta for the RigL regrow priority signal (grad_ema); unchanged from v11")
    p.add_argument("--movement-score-ema-beta", type=float, default=0.9,
                    help="NEW in v12: EMA beta for the movement PRUNING-CRITERION score itself (was an "
                         "unbounded running sum in v11). Higher = smoother/less reactive to recent steps")
    p.add_argument("--cycle-frac-init", type=float, default=0.30)
    p.add_argument("--protect-entangling-frac", type=float, default=0.34)
    p.add_argument("--distill-hardness-max", type=float, default=0.5,
                    help="NEW in v12: lowered from 0.7 (v11); reached only once pruning hits its full "
                         "sparsity target, not at the start of training")
    p.add_argument("--distill-hardness-min", type=float, default=0.2)
    p.add_argument("--quant-bits", type=int, default=6)
    p.add_argument("--quant-initial-bits", type=int, default=16)
    p.add_argument("--quant-start-frac", type=float, default=0.20)
    p.add_argument("--quant-end-frac", type=float, default=0.70)
    p.add_argument("--quant-stages", type=int, default=4)
    p.add_argument("--convergence-window-frac", type=float, default=0.15)
    p.add_argument("--convergence-rel-threshold", type=float, default=0.03)
    p.add_argument("--max-extra-steps", type=int, default=900,
                    help="NEW in v12: per-seed adaptive fine-tuning budget granted after the main --steps "
                         "schedule if that seed has not converged yet (was a hard stop + a warning in v11)")
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--ni-margin", type=float, default=0.02)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument("--gap-warn-threshold", type=float, default=0.05,
                    help="NEW in v12: report opens with a capacity-warning banner if the classical-vs-"
                         "circuit accuracy gap exceeds this for any (dataset, qubits)")
    p.add_argument("--adaptive-seeds", dest="adaptive_seeds", action="store_true", default=True)
    p.add_argument("--no-adaptive-seeds", dest="adaptive_seeds", action="store_false",
                    help="disable the adaptive seed-budget loop and just run --seeds once, like v11 did")
    p.add_argument("--target-mde80", type=float, default=0.03,
                    help="NEW in v12: adaptive seed loop keeps adding seeds until every KEY_CONTRASTS row "
                         "at noise=0 has mde80 <= this, or --max-seeds is hit")
    p.add_argument("--max-seeds", type=int, default=40)
    p.add_argument("--seed-batch-size", type=int, default=10)
    return p.parse_args()


def main():
    hp = parse_args()
    if hp.selftest:
        sys.exit(0 if selftest() else 1)
    if hp.quant_bits < MIN_QUANT_BITS:
        sys.exit(f"--quant-bits must be >= {MIN_QUANT_BITS}")
    device = pick_device(hp.device)
    print(f"device: {device}")

    res_path, san_path = f"{hp.save}_results.csv", f"{hp.save}_sanity_baseline.csv"
    records, sanity_rows, done = [], [], set()
    if hp.resume and Path(res_path).exists():
        records = read_records(res_path)
        sanity_rows = read_sanity(san_path)
        done = {(r["dataset"], r["n_qubits"], r["seed"]) for r in records}
        print(f"resuming: {len(done)} combos already done")

    sim_cache = {}
    start = time.time()
    seed_pool = list(dict.fromkeys(hp.seeds))
    round_idx = 0
    n_arms = len(build_arm_specs(hp.sparsities, hp.arms))
    while True:
        round_idx += 1
        combos = [(d, q, s) for d in hp.datasets for q in hp.qubits for s in seed_pool
                  if (d, q, s) not in done]
        if combos:
            print(f"--- seed round {round_idx}: {len(combos)} new combos x {n_arms} arms "
                  f"(total seeds so far: {len(seed_pool)}) ---")
        for i, (ds, q, seed) in enumerate(combos, 1):
            t0 = time.time()
            rows, san = run_combo(ds, q, seed, hp, device, sim_cache)
            records.extend(rows)
            if san:
                sanity_rows.append(san)
            done.add((ds, q, seed))
            write_csv(res_path, RECORD_FIELDS, records)
            if sanity_rows:
                write_csv(san_path, SANITY_FIELDS, sanity_rows)
            dense = next(r["val_acc"] for r in rows if r["run_type"] == "dense_fp" and r["noise_level"] == 0.0)
            print(f"[{i}/{len(combos)}] {ds} q={q} seed={seed}: dense_fp acc={dense:.4f}"
                  + (f" logreg={san['logreg_eval_acc']:.4f}" if san else "")
                  + f" | {time.time() - t0:.0f}s", flush=True)

        if not hp.adaptive_seeds or len(seed_pool) >= hp.max_seeds:
            break
        # NEW in v12: sequential seed-budget loop, see module docstring point 4.
        gm = compute_gatematched_contrasts(records, "val_acc", True, hp.ni_margin, hp.n_boot)
        underpowered = [r for r in gm if r["contrast"] in KEY_CONTRASTS and r["noise_level"] == 0.0
                        and r["twoq_noise_mult"] == hp.twoq_noise_mults[0]
                        and not math.isnan(r["mde80"]) and r["mde80"] > hp.target_mde80]
        if not underpowered:
            print(f"power target met for {sorted(KEY_CONTRASTS)} at noise=0 "
                  f"(mde80 <= {hp.target_mde80}) after {len(seed_pool)} seeds.")
            break
        next_start = max(seed_pool) + 1
        n_add = min(hp.seed_batch_size, hp.max_seeds - len(seed_pool))
        if n_add <= 0:
            print(f"--max-seeds ({hp.max_seeds}) reached with contrasts still underpowered "
                  f"(worst mde80 = {max(r['mde80'] for r in underpowered):.4f}).")
            break
        added = list(range(next_start, next_start + n_add))
        print(f"{len(underpowered)} key gate-matched cells still underpowered (worst mde80="
              f"{max(r['mde80'] for r in underpowered):.4f} > target {hp.target_mde80}); "
              f"adding {n_add} more seeds ({added[0]}..{added[-1]}).")
        seed_pool += added

    total = time.time() - start

    agg = aggregate_records(records)
    contrasts = {m: compute_contrast_table(records, m, hib, hp.ni_margin, hp.n_boot)
                 for m, hib in (("val_acc", True), ("val_auc", True), ("val_loss", False))}
    gatematched = {m: compute_gatematched_contrasts(records, m, hib, hp.ni_margin, hp.n_boot)
                   for m, hib in (("val_acc", True), ("val_auc", True))}
    write_csv(f"{hp.save}_aggregate.csv", list(agg[0].keys()), agg)
    for m, tab in contrasts.items():
        if tab:
            write_csv(f"{hp.save}_contrasts_{m}.csv", list(tab[0].keys()), tab)
    for m, tab in gatematched.items():
        if tab:
            write_csv(f"{hp.save}_gatematched_{m}.csv", list(tab[0].keys()), tab)
    report_contrasts = {m: contrasts[m] for m in ("val_acc", "val_auc")}
    report_gatematched = {m: gatematched[m] for m in ("val_acc", "val_auc")}
    write_report(hp.report_name, hp, records, agg, sanity_rows, report_contrasts, report_gatematched,
                 total, final_n_seeds=len(seed_pool))
    print(f"\nDone in {fmt_dur(total)}. Report: {hp.report_name}")


if __name__ == "__main__":
    main()