import argparse
import csv
import gc
import math
import pickle
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torchvision
import torchvision.transforms as T

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
IMG_COLS = 4  # fixed: encode() needs exactly 4 pixel values (RY,RX,RZ,RY) per qubit

RMSPROP_ALPHA = 0.9
RMSPROP_EPS = 1e-8
MIN_SEEDS_FOR_TEST = 5  # below this, a paired test is too underpowered to trust

if not torch.cuda.is_available():
    sys.exit(
        "CUDA GPU not available. This script is configured to run on GPU "
        "only -- install a CUDA-enabled torch build and run on a machine "
        "with an NVIDIA GPU."
    )
TORCH_DEVICE = torch.device("cuda")
DTYPE = torch.float64

print(f"Torch device: {TORCH_DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

DATASET_CLASSES = {
    "mnist": (torchvision.datasets.MNIST, 3, 6, ("3", "6")),
    "fashionmnist": (torchvision.datasets.FashionMNIST, 3, 6, ("dress", "shirt")),
}


# ----------------------------------------------------------------------
# Data -- UNCHANGED from v7/v8
# ----------------------------------------------------------------------
def _split_indices(full, class_a, class_b, n_train, n_val, seed):
    idx_a = [i for i in range(len(full)) if full.targets[i] == class_a]
    idx_b = [i for i in range(len(full)) if full.targets[i] == class_b]

    rng = np.random.RandomState(seed)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)

    n_train_each = n_train // 2
    n_val_each = n_val // 2

    train_idx = idx_a[:n_train_each] + idx_b[:n_train_each]
    val_idx = (idx_a[n_train_each:n_train_each + n_val_each]
               + idx_b[n_train_each:n_train_each + n_val_each])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    a_set = set(idx_a)
    return train_idx, val_idx, a_set


def load_binary_subset(dataset_name, n_qubits, n_train=800, n_val=800, seed=42,
                        root="./data", encoding="pca"):
    """Returns (X_train, y_train), (X_val, y_val) with X shaped
    (N, n_qubits, IMG_COLS), scaled to [0, pi].

    n_train/n_val defaults raised from v8 (500/300) to 800/800: a bigger
    validation set alone shrinks the binomial-sampling component of
    run-to-run variance (see module docstring, diagnosis point 2), before
    any modeling change.

    encoding="pca" (default): fit a min(n_features, N-1)-component PCA on
    the training crops, min-max scale each component using train-set
    stats (val is clipped to the same range, not re-fit -- no val-set
    leakage).
    encoding="resize": CenterCrop then a naive box-filter resize to
    (n_qubits, IMG_COLS). Kept for comparison.
    """
    ds_cls, class_a, class_b, _ = DATASET_CLASSES[dataset_name]
    n_features = n_qubits * IMG_COLS

    base_tfm = T.Compose([T.CenterCrop(24), T.ToTensor()])
    full = ds_cls(root=root, train=True, download=True, transform=base_tfm)
    train_idx, val_idx, idx_a_set = _split_indices(full, class_a, class_b, n_train, n_val, seed)

    def raw_flat(idx_list):
        X, y = [], []
        for i in idx_list:
            img, _ = full[i]
            X.append(img.squeeze(0).numpy().flatten())
            y.append(0 if i in idx_a_set else 1)
        return np.stack(X).astype(np.float64), np.array(y, dtype=np.int64)

    X_train_raw, y_train = raw_flat(train_idx)
    X_val_raw, y_val = raw_flat(val_idx)

    if encoding == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            sys.exit("encoding='pca' requires scikit-learn: pip install scikit-learn")
        pca = PCA(n_components=n_features, random_state=seed)
        X_train_feat = pca.fit_transform(X_train_raw)
        X_val_feat = pca.transform(X_val_raw)
        lo = X_train_feat.min(axis=0, keepdims=True)
        hi = X_train_feat.max(axis=0, keepdims=True)
        span = np.clip(hi - lo, 1e-8, None)
        X_train = (X_train_feat - lo) / span
        X_val = np.clip((X_val_feat - lo) / span, 0.0, 1.0)
    elif encoding == "resize":
        def resize_feat(idx_list):
            X = []
            for i in idx_list:
                img, _ = full[i]  # 1 x 24 x 24, already in [0,1]
                small = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(n_qubits, IMG_COLS),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0).numpy()
                X.append(small.flatten())
            return np.stack(X).astype(np.float64)
        X_train = resize_feat(train_idx)
        X_val = resize_feat(val_idx)
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")

    X_train = (X_train.reshape(-1, n_qubits, IMG_COLS)) * np.pi
    X_val = (X_val.reshape(-1, n_qubits, IMG_COLS)) * np.pi
    return (X_train, y_train), (X_val, y_val)


def classical_sanity_baseline(X_train_feat, y_train, X_val_feat, y_val):
    """Quick logistic-regression probe on the EXACT SAME PCA features fed
    to the quantum model, as a data-separability sanity check.

    This is NOT a fair "classical vs quantum" comparison (no
    hyperparameter tuning, no cross-validation, and it gets the same
    limited feature budget the circuit gets, nothing more). Its only job
    is: if a plain linear model on these features scores far above the
    dense quantum baseline, the bottleneck is capacity/training/readout
    in the quantum pipeline, not the data or the task -- exactly the
    pattern v8's results (dense accuracy ~76-77% on an easy binary split)
    pointed at. Returns None if scikit-learn isn't available.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    X_tr_flat = X_train_feat.reshape(X_train_feat.shape[0], -1)
    X_val_flat = X_val_feat.reshape(X_val_feat.shape[0], -1)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr_flat, y_train)
    return float(clf.score(X_val_flat, y_val))


# ----------------------------------------------------------------------
# Circuit (training path, torch-differentiable) -- UNCHANGED from v7/v8
# ----------------------------------------------------------------------
def encode(img, wires):
    for q, w in enumerate(wires):
        row = img[q]
        qml.RY(row[0], wires=w)
        qml.RX(row[1], wires=w)
        qml.RZ(row[2], wires=w)
        qml.RY(row[3], wires=w)


def variational_layer(param, wires):
    n = len(wires)
    for i in range(n - 1):
        qml.IsingZZ(param[i, 0], wires=[wires[i], wires[i + 1]])
    for i in range(n):
        qml.RY(param[i, 1], wires=wires[i])


def ansatz(params, img):
    n_qubits = params.shape[1]
    wires = list(range(n_qubits))
    encode(img, wires)
    for l in range(params.shape[0]):
        variational_layer(params[l], wires)
    return [qml.expval(qml.PauliZ(w)) for w in wires]


def get_qdevice(n_qubits, shots):
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits, shots=shots)
        print(f"using lightning.gpu (cuQuantum), wires={n_qubits}, shots={shots}")
        return dev, True
    except Exception as e:
        print(f"lightning.gpu unavailable ({e}); falling back to default.qubit (CPU simulator).")
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        return dev, False


def diff_method_for(dev, shots):
    if shots is None and dev.name in ("lightning.gpu", "lightning.qubit"):
        return "adjoint"
    return "parameter-shift" if shots is not None else "best"


def make_qnode(n_qubits, shots):
    dev, is_gpu_sim = get_qdevice(n_qubits, shots)
    qnode = qml.QNode(ansatz, dev, interface="torch", diff_method=diff_method_for(dev, shots))
    return qnode, is_gpu_sim


# ----------------------------------------------------------------------
# NOTE (v9 change #1): v7/v8's `reduce_to_two` -- a fixed, untrained sum
# of Z-expectation values over a hand-picked half of the qubits per class
# -- has been REMOVED. It is replaced by a small trainable linear readout
# (readout_W, readout_b) applied in `batch_forward` below and trained
# jointly with the circuit. See module docstring, v9 fix #1.
# ----------------------------------------------------------------------
def batch_forward(qckt, params, readout_W, readout_b, X_t):
    """readout_W: (n_qubits, 2) torch tensor, readout_b: (2,) torch
    tensor, both trainable and NEVER touched by the pruning/freezing
    logic (that only ever operates on `params`, the quantum ansatz
    tensor)."""
    exp_stack = torch.stack([torch.stack(qckt(params, x)) for x in X_t])  # (batch, n_qubits)
    logits = exp_stack @ readout_W + readout_b
    return torch.softmax(logits, dim=1)


def bce_loss(probs, y_t, eps=1e-7):
    yhat = probs[:, 1].clamp(eps, 1 - eps)
    per_sample = -(y_t * torch.log(yhat) + (1 - y_t) * torch.log(1 - yhat))
    return per_sample.mean()


def kd_kl_loss(student_probs, teacher_probs, eps=1e-7):
    teacher_probs = teacher_probs.to(device=student_probs.device)
    s = student_probs.clamp(eps, 1 - eps)
    t = teacher_probs.clamp(eps, 1 - eps)
    per_sample = (t * (torch.log(t) - torch.log(s))).sum(dim=1)
    return per_sample.mean()


def accuracy_from_probs(probs, y_t):
    preds = (probs[:, 1] > 0.5).long()
    return (preds == y_t.long()).float().mean().item()


def numpy_bce(y_true, y_prob, eps=1e-7):
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_auc(y_true, y_score):
    """ROC-AUC via scikit-learn if available, else a rank-based
    (Mann-Whitney U) fallback that needs no dependency beyond numpy.
    Returns nan if only one class is present (AUC is undefined there)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1)
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        sum_ranks_pos = ranks[y_true == 1].sum()
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def check_convergence(cost_hist, window_frac=0.15, rel_threshold=0.03):
    """True if training looks plateaued by the end (mean loss over the
    final window changed by less than `rel_threshold` relative to the
    window immediately before it); False if it still looks like it was
    improving fast enough that more --steps would likely help; None if
    there aren't enough recorded steps yet to judge (treated as "assume
    converged" by callers, since a short run that already hit its `tol`
    early-stop is a separate, stronger convergence signal handled
    upstream)."""
    n = len(cost_hist)
    window = max(5, int(round(n * window_frac)))
    if n < 2 * window:
        return None
    prev = float(np.mean(cost_hist[-2 * window:-window]))
    last = float(np.mean(cost_hist[-window:]))
    if prev <= 1e-9:
        return True
    rel_drop = (prev - last) / prev
    return bool(rel_drop < rel_threshold)


def get_sparsity(arr):
    return 1.0 - np.count_nonzero(arr) / arr.size


def enforce_frozen(params, frozen_mask, param_shape):
    if not frozen_mask.any():
        return
    with torch.no_grad():
        flat = params.flatten()
        flat[frozen_mask] = 0
        params.data = flat.reshape(param_shape)


def cubic_sparsity_at_step(t, start_step, end_step, final_sparsity, initial_sparsity=0.0):
    """Automated gradual pruning schedule (Zhu & Gupta, 2017,
    arXiv:1710.01878), also the schedule used by RigL and movement pruning
    for ramping the *target* sparsity level over the pruning window."""
    if t <= start_step:
        return initial_sparsity
    if t >= end_step:
        return final_sparsity
    progress = (t - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3


def rigl_cycle_fraction(t, start_step, end_step, zeta_init):
    """RigL's cosine-annealed drop/grow fraction (Evci et al. 2020, Sec 3 /
    Fig. 9): zeta_t = zeta_0 * 0.5 * (1 + cos(pi * progress)). Starts at
    zeta_0 and decays smoothly to 0 by end_step, so mask churn is largest
    early (exploration) and the mask locks in before the end of training
    (exploitation)."""
    if t <= start_step or t >= end_step or zeta_init <= 0:
        return 0.0
    progress = (t - start_step) / (end_step - start_step)
    return zeta_init * 0.5 * (1 + math.cos(math.pi * progress))


def entangling_protected_mask(n_layers, n_qubits, protect_frac, seed=0):
    """Structural prior: protect a fraction of entangling (IsingZZ)
    parameters from ever being pruned, since removing entangling gates
    tends to be disproportionately damaging to a PQC's trainability
    compared to removing single-qubit rotation parameters of the same
    count. Unchanged from v7/v8."""
    mask = np.zeros((n_layers, n_qubits, 2), dtype=bool)
    n_live_ent = max(0, n_qubits - 1)
    n_protect = int(np.ceil(protect_frac * n_live_ent)) if n_live_ent > 0 else 0
    rng = np.random.RandomState(seed)
    for l in range(n_layers):
        if n_protect == 0 or n_live_ent == 0:
            continue
        live_idx = np.arange(n_live_ent)
        protect_idx = rng.choice(live_idx, size=min(n_protect, n_live_ent), replace=False)
        mask[l, protect_idx, 0] = True
    return mask.flatten()


def top_up_to_target_sparsity(params, frozen_mask, protected_mask, movement_score,
                               target_sparsity, param_shape):
    """Safety net: if rounding/regrowth left us short of the final target
    sparsity, freeze the remaining lowest-movement-score eligible params
    directly (movement score, not raw magnitude, so the fallback ranking
    stays consistent with the main drop criterion)."""
    n_params = frozen_mask.size
    n_target = int(np.ceil(target_sparsity * n_params))
    n_frozen = int(frozen_mask.sum())
    if n_frozen >= n_target:
        return frozen_mask
    remaining = np.where(~frozen_mask & ~protected_mask)[0]
    if remaining.size == 0:
        return frozen_mask
    order = remaining[np.argsort(movement_score.flatten()[remaining])]
    need = n_target - n_frozen
    to_freeze = order[:need]
    frozen_mask = frozen_mask.copy()
    frozen_mask[to_freeze] = True
    enforce_frozen(params, frozen_mask, param_shape)
    return frozen_mask


def rigl_drop_and_grow_round(params, frozen_mask, protected_mask, movement_score,
                              grad_ema, n_target, param_shape, max_round_freeze,
                              cycle_frac):
    """Core decision rule (UNCHANGED from v7/v8 -- only its *inputs*,
    movement_score and grad_ema, changed in v8, and only the readout
    around it changed in v9; see module docstrings).

    GROW (RigL, Evci et al. 2020): among currently-frozen, unprotected
    params, reactivate the `cycle_frac`-fraction with the largest-magnitude
    (EMA-smoothed) gradient, initialized to zero on regrowth per RigL.

    DROP (Movement Pruning, Sanh et al. 2020): among the eligible active
    (non-frozen, non-protected, not just-regrown) params, freeze enough of
    the lowest-movement-score ones to hit n_target for this round, capped
    at max_round_freeze.
    """
    frozen_mask = frozen_mask.copy()

    # ---- GROW step ----
    frozen_eligible = np.where(frozen_mask & ~protected_mask)[0]
    n_grow = int(round(cycle_frac * frozen_eligible.size))
    regrown = np.array([], dtype=int)
    if n_grow > 0:
        grad_mag = np.abs(grad_ema.flatten()[frozen_eligible])
        order = frozen_eligible[np.argsort(-grad_mag)]
        regrown = order[:n_grow]
        frozen_mask[regrown] = False  # reactivated at value 0, per RigL

    # ---- DROP step ----
    n_frozen_now = int(frozen_mask.sum())
    need = max(0, n_target - n_frozen_now)
    if need > 0:
        eligible = np.where(~frozen_mask & ~protected_mask)[0]
        eligible = np.setdiff1d(eligible, regrown)  # don't immediately re-drop what we just grew
        if eligible.size > 0:
            score = movement_score.flatten()
            order = eligible[np.argsort(score[eligible])]  # ascending: least important first
            to_freeze = order[:min(need, max_round_freeze)]
            if to_freeze.size > 0:
                frozen_mask[to_freeze] = True

    enforce_frozen(params, frozen_mask, param_shape)
    return frozen_mask, regrown.size


# ----------------------------------------------------------------------
# Noisy (density-matrix) evaluation path
# ----------------------------------------------------------------------
def noisy_encode(img, wires, noise_prob):
    gates = (qml.RY, qml.RX, qml.RZ, qml.RY)
    for q, w in enumerate(wires):
        row = img[q]
        for val, gate in zip(row, gates):
            gate(float(val), wires=w)
            if noise_prob > 0:
                qml.DepolarizingChannel(noise_prob, wires=w)


def noisy_variational_layer(param_layer, wires, noise_prob, freeze_tol=1e-9):
    n = len(wires)
    for i in range(n - 1):
        theta = float(param_layer[i, 0])
        if abs(theta) > freeze_tol:
            qml.IsingZZ(theta, wires=[wires[i], wires[i + 1]])
            if noise_prob > 0:
                qml.DepolarizingChannel(noise_prob, wires=wires[i])
                qml.DepolarizingChannel(noise_prob, wires=wires[i + 1])
    for i in range(n):
        theta = float(param_layer[i, 1])
        if abs(theta) > freeze_tol:
            qml.RY(theta, wires=wires[i])
            if noise_prob > 0:
                qml.DepolarizingChannel(noise_prob, wires=wires[i])


def numpy_softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def evaluate_noise_robustness(final_params_np, final_readout_W_np, final_readout_b_np,
                               X_val, y_val, n_qubits, noise_levels,
                               max_eval_samples=300, freeze_tol=1e-9, eval_seed=0):
    """v9 change: uses the trained readout (final_readout_W/b) instead of
    v7/v8's fixed sum-of-halves, and now also returns val_loss / val_auc
    alongside val_acc for each noise level (see module docstring, fix #6).
    Returns {} if no positive noise levels were requested."""
    noise_levels_pos = sorted({nl for nl in noise_levels if nl > 0})
    if not noise_levels_pos:
        return {}
    if n_qubits > 10:
        print(f"WARNING: noisy density-matrix simulation at {n_qubits} qubits "
              f"scales as 4^{n_qubits} -- this may be very slow/memory heavy.")

    n_eval = min(max_eval_samples, len(X_val))
    rng = np.random.RandomState(eval_seed)
    idx = rng.choice(len(X_val), size=n_eval, replace=False)
    Xs, ys = X_val[idx], y_val[idx]

    dev = qml.device("default.mixed", wires=n_qubits)
    wires = list(range(n_qubits))

    def circuit(noise_prob, img):
        noisy_encode(img, wires, noise_prob)
        for l in range(final_params_np.shape[0]):
            noisy_variational_layer(final_params_np[l], wires, noise_prob, freeze_tol)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    qnode = qml.QNode(circuit, dev)

    results = {}
    for noise_prob in noise_levels_pos:
        p1_list, y_list = [], []
        for x, y in zip(Xs, ys):
            out = np.array(qnode(noise_prob, x), dtype=float)  # (n_qubits,)
            logits = out @ final_readout_W_np + final_readout_b_np
            probs = numpy_softmax(logits)
            p1_list.append(float(probs[1]))
            y_list.append(int(y))
        p1_arr = np.array(p1_list)
        y_arr = np.array(y_list)
        acc = float(np.mean((p1_arr > 0.5).astype(int) == y_arr))
        loss = numpy_bce(y_arr, p1_arr)
        auc = compute_auc(y_arr, p1_arr)
        results[noise_prob] = {"val_acc": acc, "val_loss": loss, "val_auc": auc}
    return results


# ----------------------------------------------------------------------
# Reporting helpers -- UNCHANGED from v7/v8
# ----------------------------------------------------------------------
def format_duration(seconds):
    return str(timedelta(seconds=round(seconds, 2)))


def print_no_prune_step(step, acc, loss, t):
    print(f"step {step}: accuracy={acc:.4f}, loss={loss:.6f}, time={t:.4f}s")


def print_prune_step(step, acc, sparsity, loss, t, n_regrown=0):
    print(f"step {step}: accuracy={acc:.4f}, sparsity={sparsity:.4f}, "
          f"loss={loss:.6f}, regrown={n_regrown}, time={t:.4f}s")


def print_no_prune_cumulative(acc, loss, total_t):
    print(f"cumulative result: accuracy={acc:.4f}, loss={loss:.6f}, "
          f"time={format_duration(total_t)} ({total_t:.4f}s)")


def print_prune_cumulative(acc, sparsity, loss, total_t):
    print(f"cumulative result: accuracy={acc:.4f}, sparsity={sparsity:.4f}, "
          f"loss={loss:.6f}, time={format_duration(total_t)} ({total_t:.4f}s)")


# ----------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------
def _init_readout(n_qubits, seed, compute_device):
    rng = np.random.RandomState(seed + 10_000)  # offset so it never aliases circuit/mask RNGs
    W0 = rng.uniform(-1.0, 1.0, size=(n_qubits, 2)) / np.sqrt(n_qubits)
    readout_W = torch.tensor(W0, dtype=DTYPE, device=compute_device, requires_grad=True)
    readout_b = torch.zeros(2, dtype=DTYPE, device=compute_device, requires_grad=True)
    return readout_W, readout_b


def optimize_and_prune(qckt, iparams, X_train, y_train, X_val, y_val,
                        win_sz=4, steps=200, tol=0.01, lr=0.1,
                        target_sparsity=0.25, compute_device=None,
                        prune_start_frac=0.15, prune_end_frac=0.65,
                        max_round_freeze_frac=0.08, verbose=True,
                        movement_ema_beta=0.6, cycle_frac_init=0.30,
                        protect_entangling_frac=0.34,
                        seed=0, teacher_params=None,
                        teacher_readout_W=None, teacher_readout_b=None,
                        distill_hardness_max=0.7, distill_hardness_min=0.2,
                        lr_warm_mult=2.0,
                        convergence_window_frac=0.15, convergence_rel_threshold=0.03):
    """QAdaPrune-RigL (v9): dynamic sparse training with a Movement-Pruning
    drop criterion and a RigL grow criterion. The drop/grow decision rule
    is identical to v7/v8. v9 changes: (a) a trainable classical readout
    (readout_W/readout_b) replaces the fixed sum-of-halves decoder and is
    trained jointly but never pruned; (b) a convergence check is recorded
    alongside the result; (c) val_auc/val_loss are computed alongside
    val_acc. See module docstring.
    """
    compute_device = compute_device if compute_device is not None else TORCH_DEVICE
    X_t = torch.tensor(X_train, dtype=DTYPE, device=compute_device)
    y_t = torch.tensor(y_train, dtype=DTYPE, device=compute_device)
    Xv_t = torch.tensor(X_val, dtype=DTYPE, device=compute_device)
    yv_t = torch.tensor(y_val, dtype=DTYPE, device=compute_device)

    params = torch.tensor(iparams, dtype=DTYPE, device=compute_device, requires_grad=True)
    param_shape = params.shape
    n_layers, n_qubits, _ = param_shape
    n_params = int(np.prod(param_shape))

    readout_W, readout_b = _init_readout(n_qubits, seed, compute_device)

    opt = torch.optim.RMSprop([params, readout_W, readout_b], lr=lr,
                               alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)

    start_step = max(win_sz, int(round(prune_start_frac * steps)))
    end_step = max(start_step + win_sz, int(round(prune_end_frac * steps)))
    max_round_freeze = max(1, int(np.ceil(max_round_freeze_frac * n_params)))

    protected_mask = entangling_protected_mask(n_layers, n_qubits, protect_entangling_frac, seed=seed)

    teacher_t = None
    teacher_readout_W_t = None
    teacher_readout_b_t = None
    if teacher_params is not None:
        teacher_t = torch.tensor(teacher_params, dtype=DTYPE, device=compute_device)
        teacher_readout_W_t = torch.tensor(teacher_readout_W, dtype=DTYPE, device=compute_device)
        teacher_readout_b_t = torch.tensor(teacher_readout_b, dtype=DTYPE, device=compute_device)

    # Movement Pruning score: accumulated running SUM of -(theta * grad),
    # per Sanh et al. -- updated EVERY step from the real training
    # gradient (v8 fix, unchanged in v9).
    movement_score = np.zeros(n_params)
    # RigL grow-criterion EMA of |grad|, also from the real training
    # gradient.
    grad_ema = np.zeros(n_params)
    frozen_mask = np.zeros(n_params, dtype=bool)

    cost_hists, train_acc_hists = [], []
    step_records = []

    if verbose:
        print("\ntraining with QAdaPrune-RigL (v9) pruning:")
    loop_start = time.time()
    new_cost = float("nan")
    for t in range(steps):
        step_start = time.time()

        steps_since_round = t - (t // win_sz) * win_sz
        warm_progress = min(1.0, steps_since_round / max(1, win_sz))
        cur_lr = lr + (lr * lr_warm_mult - lr) * (1 - warm_progress)
        opt.param_groups[0]["lr"] = cur_lr

        if teacher_t is not None:
            explore_progress = min(1.0, max(0.0, t / max(1, end_step)))
            hardness = distill_hardness_max + (distill_hardness_min - distill_hardness_max) * explore_progress
        else:
            hardness = 0.0

        # theta_t, captured BEFORE this step's update, for the movement
        # score theta_t * grad_t.
        params_before = params.detach().cpu().numpy().flatten()

        opt.zero_grad()
        probs = batch_forward(qckt, params, readout_W, readout_b, X_t)
        loss = bce_loss(probs, y_t)
        if teacher_t is not None and hardness > 0:
            with torch.no_grad():
                teacher_probs = batch_forward(qckt, teacher_t, teacher_readout_W_t, teacher_readout_b_t, X_t)
            loss = (1 - hardness) * loss + hardness * kd_kl_loss(probs, teacher_probs)
        loss.backward()

        # Reuse the REAL training gradient (same one opt.step() is about
        # to apply) for both the movement-pruning drop score and the RigL
        # grow-criterion EMA. Only `params` (the quantum ansatz tensor)
        # feeds the pruning logic -- readout_W/readout_b are optimized but
        # never pruned or frozen.
        cur_grad = params.grad.detach().cpu().numpy().flatten()
        movement_score += -(params_before * cur_grad)
        grad_ema = movement_ema_beta * grad_ema + (1 - movement_ema_beta) * np.abs(cur_grad)

        opt.step()
        enforce_frozen(params, frozen_mask, param_shape)

        new_cost = loss.item()
        cost_hists.append(new_cost)
        with torch.no_grad():
            train_acc = accuracy_from_probs(batch_forward(qckt, params, readout_W, readout_b, X_t), y_t)
        train_acc_hists.append(train_acc)

        n_regrown_this_step = 0
        if t != 0 and t % win_sz == 0:
            sparsity_target_now = cubic_sparsity_at_step(t, start_step, end_step, target_sparsity)
            n_target = int(np.ceil(sparsity_target_now * n_params))

            cycle_frac = rigl_cycle_fraction(t, start_step, end_step, cycle_frac_init)

            frozen_mask, n_regrown_this_step = rigl_drop_and_grow_round(
                params, frozen_mask, protected_mask, movement_score, grad_ema,
                n_target, param_shape, max_round_freeze, cycle_frac,
            )

        step_sparsity = get_sparsity(params.detach().cpu().numpy())
        step_time = time.time() - step_start
        step_records.append((t + 1, train_acc, step_sparsity, new_cost, step_time))
        if verbose:
            print_prune_step(t + 1, train_acc, step_sparsity, new_cost, step_time, n_regrown_this_step)

        if new_cost < tol:
            break

    frozen_mask = top_up_to_target_sparsity(
        params, frozen_mask, protected_mask, movement_score, target_sparsity, param_shape
    )

    if new_cost < tol:
        converged = True
    else:
        plateaued = check_convergence(cost_hists, convergence_window_frac, convergence_rel_threshold)
        converged = True if plateaued is None else plateaued
        if not converged and verbose:
            print(f"WARNING: loss still decreasing meaningfully after {len(cost_hists)} steps "
                  f"(tol={tol} not reached, no plateau detected) -- consider more --steps.")

    sparsity = get_sparsity(params.detach().cpu().numpy())
    with torch.no_grad():
        val_probs = batch_forward(qckt, params, readout_W, readout_b, Xv_t)
        val_acc = accuracy_from_probs(val_probs, yv_t)
        val_loss = bce_loss(val_probs, yv_t).item()
    val_auc = compute_auc(y_val, val_probs[:, 1].detach().cpu().numpy())

    total_time = time.time() - loop_start
    final_train_acc = step_records[-1][1] if step_records else float("nan")
    final_train_loss = step_records[-1][3] if step_records else float("nan")
    print_prune_cumulative(final_train_acc, sparsity, final_train_loss, total_time)
    print(f"(validation: acc={val_acc:.4f}, auc={val_auc:.4f}, loss={val_loss:.4f}, converged={converged})")

    return {
        "step_records": step_records,
        "cost_hists": cost_hists,
        "train_acc_hists": train_acc_hists,
        "sparsity": sparsity,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_loss": val_loss,
        "converged": converged,
        "total_time": total_time,
        "final_params": params.detach().cpu().numpy(),
        "final_readout_W": readout_W.detach().cpu().numpy(),
        "final_readout_b": readout_b.detach().cpu().numpy(),
    }


def optimize(qckt, iparams, X_train, y_train, X_val, y_val, steps=200, tol=0.01, lr=0.1,
             compute_device=None, verbose=True, seed=0,
             convergence_window_frac=0.15, convergence_rel_threshold=0.03):
    """Dense (no pruning) baseline / teacher. v9 adds the trainable
    readout, val_auc/val_loss, and a convergence check -- otherwise
    unchanged from v7/v8."""
    compute_device = compute_device if compute_device is not None else TORCH_DEVICE
    X_t = torch.tensor(X_train, dtype=DTYPE, device=compute_device)
    y_t = torch.tensor(y_train, dtype=DTYPE, device=compute_device)
    Xv_t = torch.tensor(X_val, dtype=DTYPE, device=compute_device)
    yv_t = torch.tensor(y_val, dtype=DTYPE, device=compute_device)

    params = torch.tensor(iparams, dtype=DTYPE, device=compute_device, requires_grad=True)
    n_qubits = params.shape[1]
    readout_W, readout_b = _init_readout(n_qubits, seed, compute_device)

    opt = torch.optim.RMSprop([params, readout_W, readout_b], lr=lr,
                               alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)

    cost_hists, train_acc_hists = [], []
    step_records = []

    if verbose:
        print("\ntraining without pruning (dense teacher):")
    loop_start = time.time()
    new_cost = float("nan")
    for t in range(steps):
        step_start = time.time()

        opt.zero_grad()
        probs = batch_forward(qckt, params, readout_W, readout_b, X_t)
        loss = bce_loss(probs, y_t)
        loss.backward()
        opt.step()

        new_cost = loss.item()
        cost_hists.append(new_cost)
        with torch.no_grad():
            train_acc = accuracy_from_probs(batch_forward(qckt, params, readout_W, readout_b, X_t), y_t)
        train_acc_hists.append(train_acc)

        step_time = time.time() - step_start
        step_records.append((t + 1, train_acc, new_cost, step_time))
        if verbose:
            print_no_prune_step(t + 1, train_acc, new_cost, step_time)

        if new_cost < tol:
            break

    if new_cost < tol:
        converged = True
    else:
        plateaued = check_convergence(cost_hists, convergence_window_frac, convergence_rel_threshold)
        converged = True if plateaued is None else plateaued
        if not converged and verbose:
            print(f"WARNING: loss still decreasing meaningfully after {len(cost_hists)} steps "
                  f"(tol={tol} not reached, no plateau detected) -- consider more --steps.")

    with torch.no_grad():
        val_probs = batch_forward(qckt, params, readout_W, readout_b, Xv_t)
        val_acc = accuracy_from_probs(val_probs, yv_t)
        val_loss = bce_loss(val_probs, yv_t).item()
    val_auc = compute_auc(y_val, val_probs[:, 1].detach().cpu().numpy())

    total_time = time.time() - loop_start
    final_train_acc = step_records[-1][1] if step_records else float("nan")
    final_train_loss = step_records[-1][2] if step_records else float("nan")
    print_no_prune_cumulative(final_train_acc, final_train_loss, total_time)
    print(f"(validation: acc={val_acc:.4f}, auc={val_auc:.4f}, loss={val_loss:.4f}, converged={converged})")

    return {
        "step_records": step_records,
        "cost_hists": cost_hists,
        "train_acc_hists": train_acc_hists,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_loss": val_loss,
        "converged": converged,
        "total_time": total_time,
        "final_params": params.detach().cpu().numpy(),
        "final_readout_W": readout_W.detach().cpu().numpy(),
        "final_readout_b": readout_b.detach().cpu().numpy(),
    }


# ----------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------
RECORD_FIELDS = [
    "dataset", "n_qubits", "seed", "run_type", "algorithm", "target_sparsity",
    "achieved_sparsity", "n_params", "n_params_active",
    "train_final_acc", "train_final_loss", "converged",
    "noise_level", "val_acc", "val_auc", "val_loss", "total_time_s",
]

SANITY_FIELDS = ["dataset", "n_qubits", "seed", "logreg_val_acc"]


def _build_rows(dataset, n_qubits, seed, run_type, algorithm, target_sparsity, achieved_sparsity,
                 n_params, train_final_acc, train_final_loss, converged,
                 val_acc0, val_auc0, val_loss0, noisy_results, total_time):
    n_active = int(round((1 - achieved_sparsity) * n_params))
    base = {
        "dataset": dataset, "n_qubits": n_qubits, "seed": seed, "run_type": run_type,
        "algorithm": algorithm,
        "target_sparsity": target_sparsity, "achieved_sparsity": achieved_sparsity,
        "n_params": n_params, "n_params_active": n_active,
        "train_final_acc": train_final_acc, "train_final_loss": train_final_loss,
        "converged": converged, "total_time_s": total_time,
    }
    rows = [dict(base, noise_level=0.0, val_acc=val_acc0, val_auc=val_auc0, val_loss=val_loss0)]
    for noise_level, m in sorted(noisy_results.items()):
        rows.append(dict(base, noise_level=noise_level, val_acc=m["val_acc"],
                          val_auc=m["val_auc"], val_loss=m["val_loss"]))
    return rows


def run_single(dataset_name, n_qubits, seed, n_layers, steps, win_sz, sparsities,
               noise_levels, prune_start_frac, prune_end_frac, max_round_freeze_frac,
               shots, noise_eval_samples, quiet, encoding, n_train, n_val,
               skip_sanity_check, v9_hparams):
    (X_train, y_train), (X_val, y_val) = load_binary_subset(
        dataset_name, n_qubits, n_train=n_train, n_val=n_val, seed=seed, encoding=encoding
    )

    sanity_row = None
    if not skip_sanity_check:
        logreg_acc = classical_sanity_baseline(X_train, y_train, X_val, y_val)
        if logreg_acc is not None:
            print(f"[{dataset_name}/q{n_qubits}/seed{seed}] classical sanity-check "
                  f"(LogisticRegression on same PCA features): val_acc={logreg_acc:.4f}")
            sanity_row = {"dataset": dataset_name, "n_qubits": n_qubits, "seed": seed,
                          "logreg_val_acc": logreg_acc}

    np.random.seed(seed)
    init_params = np.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 2))
    n_params = n_layers * n_qubits * 2

    qckt, is_gpu_sim = make_qnode(n_qubits, shots)
    compute_device = TORCH_DEVICE if is_gpu_sim else torch.device("cpu")
    print(f"[{dataset_name}/q{n_qubits}/seed{seed}] circuit device: {compute_device} "
          f"({'lightning.gpu' if is_gpu_sim else 'default.qubit fallback'})")

    no_prune_result = optimize(
        qckt, init_params.copy(), X_train, y_train, X_val, y_val, steps=steps,
        compute_device=compute_device, verbose=not quiet, seed=seed,
        convergence_window_frac=v9_hparams["convergence_window_frac"],
        convergence_rel_threshold=v9_hparams["convergence_rel_threshold"],
    )
    noise_np = evaluate_noise_robustness(
        no_prune_result["final_params"], no_prune_result["final_readout_W"],
        no_prune_result["final_readout_b"], X_val, y_val, n_qubits, noise_levels,
        max_eval_samples=noise_eval_samples, eval_seed=seed,
    )
    records = _build_rows(
        dataset_name, n_qubits, seed, "no_pruning", "dense", None, 0.0, n_params,
        no_prune_result["step_records"][-1][1], no_prune_result["step_records"][-1][2],
        no_prune_result["converged"],
        no_prune_result["val_acc"], no_prune_result["val_auc"], no_prune_result["val_loss"],
        noise_np, no_prune_result["total_time"],
    )

    pruning_hparams = {k: v for k, v in v9_hparams.items()}
    pruned_results = {}
    for ts in sparsities:
        prune_result = optimize_and_prune(
            qckt, init_params.copy(), X_train, y_train, X_val, y_val,
            win_sz=win_sz, steps=steps, target_sparsity=ts, compute_device=compute_device,
            prune_start_frac=prune_start_frac, prune_end_frac=prune_end_frac,
            max_round_freeze_frac=max_round_freeze_frac, verbose=not quiet,
            teacher_params=no_prune_result["final_params"],
            teacher_readout_W=no_prune_result["final_readout_W"],
            teacher_readout_b=no_prune_result["final_readout_b"],
            seed=seed, **pruning_hparams,
        )
        noise_p = evaluate_noise_robustness(
            prune_result["final_params"], prune_result["final_readout_W"],
            prune_result["final_readout_b"], X_val, y_val, n_qubits, noise_levels,
            max_eval_samples=noise_eval_samples, eval_seed=seed,
        )
        records.extend(_build_rows(
            dataset_name, n_qubits, seed, "pruned", "qadaprune_rigl_v9", ts, prune_result["sparsity"],
            n_params, prune_result["step_records"][-1][1], prune_result["step_records"][-1][3],
            prune_result["converged"],
            prune_result["val_acc"], prune_result["val_auc"], prune_result["val_loss"],
            noise_p, prune_result["total_time"],
        ))
        pruned_results[ts] = prune_result

    raw_bundle = {
        "dataset": dataset_name, "n_qubits": n_qubits, "seed": seed,
        "no_pruning": no_prune_result, "pruned": pruned_results,
    }

    # Cleanup: drop the only remaining Python reference to the QNode (and,
    # through it, the lightning.gpu device / cuStateVec handle) before the
    # caller moves on to the next combo. See v8 module docstring for why;
    # unchanged in v9.
    del qckt

    return records, raw_bundle, sanity_row


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


# Used only by --dispatch mode, to read a completed combo's _results.csv /
# _sanity_baseline.csv back in for merging.
_INT_FIELDS = {"n_qubits", "seed", "n_params", "n_params_active"}
_FLOAT_FIELDS = {
    "achieved_sparsity", "train_final_acc", "train_final_loss",
    "noise_level", "val_acc", "val_auc", "val_loss", "total_time_s",
}
_BOOL_FIELDS = {"converged"}


def read_records(path):
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = dict(row)
            for k in _INT_FIELDS:
                rec[k] = int(rec[k])
            for k in _FLOAT_FIELDS:
                rec[k] = float(rec[k])
            for k in _BOOL_FIELDS:
                rec[k] = (rec[k] == "True")
            rec["target_sparsity"] = (
                None if rec["target_sparsity"] == "" else float(rec["target_sparsity"])
            )
            records.append(rec)
    return records


def read_sanity_rows(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "dataset": row["dataset"], "n_qubits": int(row["n_qubits"]),
                "seed": int(row["seed"]), "logreg_val_acc": float(row["logreg_val_acc"]),
            })
    return rows


def _mean_std_ci(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return mean, std, ci95


def aggregate_records(records):
    groups = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["n_qubits"], r["run_type"], r["target_sparsity"], r["noise_level"])
        groups[key].append(r)
    agg = []
    for (dataset, n_qubits, run_type, target_sparsity, noise_level), rs in groups.items():
        row = {
            "dataset": dataset, "n_qubits": n_qubits, "run_type": run_type,
            "target_sparsity": target_sparsity, "noise_level": noise_level,
            "n_seeds": len(rs),
        }
        for metric in ("val_acc", "val_auc", "val_loss"):
            mean, std, ci95 = _mean_std_ci([r[metric] for r in rs])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        row["converged_frac"] = float(np.mean([1.0 if r["converged"] else 0.0 for r in rs]))
        agg.append(row)
    agg.sort(key=lambda r: (r["dataset"], r["n_qubits"], r["run_type"] != "no_pruning",
                             r["target_sparsity"] or 0, r["noise_level"]))
    return agg


def _paired_test(a_vals, b_vals):
    """Paired significance test on (a - b) per seed. Prefers Wilcoxon
    signed-rank (nonparametric, robust to the small/unequal sample sizes
    typical here); falls back to a paired t-test if scipy is unavailable
    or the sample is degenerate. Returns (p_value or None, method_str).
    UNCHANGED from v7/v8."""
    diffs = np.array(a_vals) - np.array(b_vals)
    if len(diffs) < MIN_SEEDS_FOR_TEST:
        return None, "insufficient_seeds"
    if np.allclose(diffs, diffs[0]):
        return None, "degenerate"
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(diffs)
        return float(p), "wilcoxon"
    except Exception:
        try:
            from scipy.stats import ttest_rel
            stat, p = ttest_rel(a_vals, b_vals)
            return float(p), "paired_t"
        except Exception:
            return None, "scipy_unavailable"


def compute_gap_table(records, metric, higher_is_better, alpha=0.05):
    """Paired (no_pruning - pruned) gap on `metric`, per seed, oriented so
    a POSITIVE gap always means "pruning hurt" regardless of whether the
    metric is higher-is-better (accuracy, AUC) or lower-is-better (loss).
    v9 generalizes v8's accuracy-only gap table to run over
    {val_acc, val_auc, val_loss} (see module docstring, fix #6)."""
    lookup = defaultdict(dict)
    for r in records:
        val = r[metric]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        key = (r["dataset"], r["n_qubits"], r["seed"], r["noise_level"])
        if r["run_type"] == "no_pruning":
            lookup[key]["baseline"] = val
        else:
            lookup[key].setdefault("pruned", {})[r["target_sparsity"]] = val

    paired = defaultdict(list)
    for (dataset, n_qubits, seed, noise_level), d in lookup.items():
        if "baseline" not in d or "pruned" not in d:
            continue
        for sparsity, val in d["pruned"].items():
            paired[(dataset, n_qubits, sparsity, noise_level)].append((d["baseline"], val))

    gap_rows = []
    for (dataset, n_qubits, sparsity, noise_level), pairs in paired.items():
        baseline_vals = [p[0] for p in pairs]
        pruned_vals = [p[1] for p in pairs]
        if higher_is_better:
            diffs = [b - p for b, p in pairs]  # positive = pruning hurt
        else:
            diffs = [p - b for b, p in pairs]  # positive = pruning hurt (loss went up)
        n = len(diffs)
        mean = float(np.mean(diffs))
        std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        p_value, test_method = _paired_test(baseline_vals, pruned_vals)
        significant = (p_value is not None) and (p_value < alpha)
        gap_rows.append({
            "dataset": dataset, "n_qubits": n_qubits, "target_sparsity": sparsity,
            "noise_level": noise_level, "n_seeds": n,
            "gap_mean": mean, "gap_ci95": ci95,
            "p_value": p_value, "test_method": test_method, "significant": significant,
        })
    gap_rows.sort(key=lambda r: (r["dataset"], r["n_qubits"], r["target_sparsity"], r["noise_level"]))
    return gap_rows


def write_report(path, config, records, agg, sanity_rows, total_elapsed):
    gap_metrics = [
        ("val_acc", True, "ACCURACY"),
        ("val_auc", True, "AUC"),
        ("val_loss", False, "LOSS (higher gap = pruning increased loss)"),
    ]
    gap_tables = {m: compute_gap_table(records, m, higher) for m, higher, _ in gap_metrics}

    with open(path, "w") as f:
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("Algorithm: QAdaPrune-RigL (v9) -- movement-pruning drop criterion "
                "(Sanh et al. 2020, arXiv:2005.07683) + RigL grow criterion "
                "(Evci et al. 2020, arXiv:1911.11134) + automated gradual pruning "
                "sparsity ramp (Zhu & Gupta 2017, arXiv:1710.01878), unchanged from "
                "v7/v8. v9 adds a trainable classical readout (replacing v7/v8's "
                "fixed sum-of-halves decoder), larger circuit/step/seed/validation "
                "budgets, a convergence check per run, AUC/loss metrics alongside "
                "accuracy, and a classical logistic-regression sanity-check baseline "
                "on the same PCA features. See qadaprune_v9.py module docstring for "
                "the full diagnosis of why v8's results were inconclusive.\n")
        f.write(f"Torch device: {TORCH_DEVICE}\n")
        f.write("Run config:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal sweep run time: {format_duration(total_elapsed)} ({total_elapsed:.2f}s)\n")
        f.write(f"Total records: {len(records)}\n\n")

        if sanity_rows:
            by_combo = defaultdict(list)
            for r in sanity_rows:
                by_combo[(r["dataset"], r["n_qubits"])].append(r["logreg_val_acc"])
            f.write("=" * 90 + "\n")
            f.write("CLASSICAL SANITY-CHECK BASELINE (LogisticRegression on same PCA features)\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'dataset':<16}{'q':<4}{'n_seeds':<9}{'logreg_acc_mean':<18}\n")
            for (dataset, n_qubits), vals in sorted(by_combo.items()):
                mean, _, _ = _mean_std_ci(vals)
                f.write(f"{dataset:<16}{n_qubits:<4}{len(vals):<9}{mean:<18.4f}\n")
            f.write("If this is far above the dense (no_pruning) val_acc below for the "
                    "same dataset/qubit combo, the bottleneck is the quantum pipeline "
                    "(capacity, training length, or readout) rather than the data.\n\n")

        f.write("=" * 90 + "\n")
        f.write("VALIDATION METRICS (mean +/- 95% CI across seeds)\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'dataset':<12}{'q':<4}{'run_type':<11}{'sparsity':<9}{'noise':<7}{'n':<4}"
                f"{'acc':<9}{'ci':<7}{'auc':<9}{'ci':<7}{'loss':<9}{'ci':<7}{'conv%':<7}\n")
        for r in agg:
            sparsity_str = "-" if r["target_sparsity"] is None else f"{r['target_sparsity']:.2f}"
            f.write(f"{r['dataset']:<12}{r['n_qubits']:<4}{r['run_type']:<11}{sparsity_str:<9}"
                    f"{r['noise_level']:<7.3f}{r['n_seeds']:<4}"
                    f"{r['val_acc_mean']:<9.4f}{r['val_acc_ci95']:<7.4f}"
                    f"{r['val_auc_mean']:<9.4f}{r['val_auc_ci95']:<7.4f}"
                    f"{r['val_loss_mean']:<9.4f}{r['val_loss_ci95']:<7.4f}"
                    f"{100 * r['converged_frac']:<7.0f}\n")

        n_not_converged = sum(1 for r in agg if r["converged_frac"] < 1.0)
        f.write(f"\nCAUTION: {n_not_converged}/{len(agg)} (dataset,qubits,run_type,sparsity,noise) "
                f"groups had at least one seed whose loss had not plateaued by the end of "
                f"training (conv% < 100) -- results for those groups may still be improvable "
                f"with more --steps rather than reflecting a real pruning effect.\n")

        for metric, higher_is_better, label in gap_metrics:
            gaps = gap_tables[metric]
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"PAIRED GAP: {label} (no_pruning - pruned, per seed) + significance\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'dataset':<14}{'q':<4}{'sparsity':<10}{'noise':<8}{'n_seeds':<9}"
                    f"{'gap_mean':<10}{'ci95':<8}{'p_value':<10}{'sig?':<6}method\n")
            for r in gaps:
                p_str = "n/a" if r["p_value"] is None else f"{r['p_value']:.4f}"
                sig_str = "YES" if r["significant"] else "no"
                f.write(f"{r['dataset']:<14}{r['n_qubits']:<4}{r['target_sparsity']:<10.2f}"
                        f"{r['noise_level']:<8.3f}{r['n_seeds']:<9}{r['gap_mean']:<10.4f}"
                        f"{r['gap_ci95']:<8.4f}{p_str:<10}{sig_str:<6}{r['test_method']}\n")
            n_underpowered = sum(1 for r in gaps if r["test_method"] == "insufficient_seeds")
            if gaps:
                f.write(f"{n_underpowered}/{len(gaps)} cells had fewer than {MIN_SEEDS_FOR_TEST} "
                        f"seeds -- no significance test was run for those.\n")

        f.write("\nOnly rows marked sig?=YES in any of the three gap tables above should be "
                "treated as a real difference from the dense baseline; everything else is "
                "consistent with noise at the tested seed count.\n")
        f.write("Note: noise_level=0.0 rows use the noiseless (fast) simulator; "
                "noise_level>0.0 rows use a density-matrix simulator with a "
                "DepolarizingChannel inserted after every gate that is actually "
                "applied (frozen/zeroed gates are skipped).\n")
        f.write("See {results,aggregate}.csv, the three gap CSVs, and "
                "_sanity_baseline.csv for the full machine-readable tables.\n")


def run_experiment_grid(datasets, qubit_counts, sparsities, seeds, noise_levels,
                         n_layers, steps, win_sz, shots,
                         prune_start_frac, prune_end_frac,
                         max_round_freeze_frac, noise_eval_samples,
                         quiet, encoding, n_train, n_val, skip_sanity_check,
                         save_prefix, save_raw, v9_hparams):
    combos = [(d, q, s) for d in datasets for q in qubit_counts for s in seeds]
    total = len(combos)
    all_records = []
    all_sanity_rows = []
    raw_bundles = []

    for i, (dataset_name, n_qubits, seed) in enumerate(combos, 1):
        print(f"\n{'#' * 78}\n[{i}/{total}] dataset={dataset_name} qubits={n_qubits} seed={seed}\n{'#' * 78}")
        records, raw_bundle, sanity_row = run_single(
            dataset_name, n_qubits, seed, n_layers, steps, win_sz, sparsities,
            noise_levels, prune_start_frac, prune_end_frac, max_round_freeze_frac,
            shots, noise_eval_samples, quiet, encoding, n_train, n_val,
            skip_sanity_check, v9_hparams,
        )
        all_records.extend(records)
        if sanity_row is not None:
            all_sanity_rows.append(sanity_row)
        raw_bundles.append(raw_bundle)

        write_csv(f"{save_prefix}_results.csv", RECORD_FIELDS, all_records)
        if all_sanity_rows:
            write_csv(f"{save_prefix}_sanity_baseline.csv", SANITY_FIELDS, all_sanity_rows)
        if save_raw:
            with open(f"{save_prefix}_raw.pkl", "wb") as f:
                pickle.dump(raw_bundles, f)

        # First-line mitigation for the custatevec memory-pool accumulation
        # issue diagnosed in v8 (does not guarantee cuQuantum's internal
        # pool is released -- if long single-process sweeps still crash,
        # use --dispatch, which reclaims GPU memory unconditionally between
        # combos via subprocess isolation).
        gc.collect()
        torch.cuda.empty_cache()

    return all_records, all_sanity_rows, raw_bundles


# ----------------------------------------------------------------------
# --dispatch mode: run each (dataset, n_qubits, seed) combo as its own OS
# subprocess (re-invoking THIS SAME FILE without --dispatch). Unchanged
# from v8 apart from threading the new v9 CLI flags through.
# ----------------------------------------------------------------------
def build_worker_cmd(args, dataset_name, n_qubits, seed, combo_prefix):
    cmd = [
        sys.executable, __file__,
        "--datasets", dataset_name,
        "--qubits", str(n_qubits),
        "--seeds", str(seed),
        "--sparsities", *[str(x) for x in args.sparsities],
        "--save", str(combo_prefix),
        "--report-name", f"{combo_prefix}_report.txt",
        "--layers", str(args.layers),
        "--steps", str(args.steps),
        "--win-sz", str(args.win_sz),
        "--encoding", args.encoding,
        "--n-train", str(args.n_train),
        "--n-val", str(args.n_val),
        "--noise-levels", *[str(x) for x in args.noise_levels],
        "--noise-eval-samples", str(args.noise_eval_samples),
        "--prune-start-frac", str(args.prune_start_frac),
        "--prune-end-frac", str(args.prune_end_frac),
        "--max-round-freeze-frac", str(args.max_round_freeze_frac),
        "--movement-ema-beta", str(args.movement_ema_beta),
        "--cycle-frac-init", str(args.cycle_frac_init),
        "--protect-entangling-frac", str(args.protect_entangling_frac),
        "--distill-hardness-max", str(args.distill_hardness_max),
        "--distill-hardness-min", str(args.distill_hardness_min),
        "--lr-warm-mult", str(args.lr_warm_mult),
        "--convergence-window-frac", str(args.convergence_window_frac),
        "--convergence-rel-threshold", str(args.convergence_rel_threshold),
    ]
    if args.shots is not None:
        cmd += ["--shots", str(args.shots)]
    if args.quiet:
        cmd.append("--quiet")
    if args.save_raw:
        cmd.append("--save-raw")
    if args.skip_sanity_check:
        cmd.append("--skip-sanity-check")
    return cmd


def run_dispatch(args):
    combos = [(d, q, s) for d in args.datasets for q in args.qubits for s in args.seeds]
    total = len(combos)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_sanity_rows = []
    failed_combos = []
    driver_start = time.time()

    config = {
        "mode": "dispatch (subprocess-per-combo)",
        "datasets": args.datasets, "qubits": args.qubits, "sparsities": args.sparsities,
        "seeds": args.seeds, "encoding": args.encoding, "noise_levels": args.noise_levels,
        "layers": args.layers, "steps": args.steps, "win_sz": args.win_sz, "shots": args.shots,
        "n_train": args.n_train, "n_val": args.n_val, "work_dir": str(work_dir),
    }

    for i, (dataset_name, n_qubits, seed) in enumerate(combos, 1):
        combo_prefix = work_dir / f"{dataset_name}_q{n_qubits}_seed{seed}"
        results_csv = Path(f"{combo_prefix}_results.csv")
        sanity_csv = Path(f"{combo_prefix}_sanity_baseline.csv")

        if args.skip_existing and results_csv.exists():
            print(f"[{i}/{total}] SKIP {dataset_name}/q{n_qubits}/seed{seed} "
                  f"(found existing {results_csv})")
            all_records.extend(read_records(results_csv))
            all_sanity_rows.extend(read_sanity_rows(sanity_csv))
            continue

        cmd = build_worker_cmd(args, dataset_name, n_qubits, seed, combo_prefix)
        print(f"\n{'#' * 78}")
        print(f"[{i}/{total}] dataset={dataset_name} qubits={n_qubits} seed={seed}")
        print(" ".join(shlex.quote(c) for c in cmd))
        print("#" * 78)

        ok = False
        attempts = args.max_retries + 1
        for attempt in range(1, attempts + 1):
            proc = subprocess.run(cmd)
            if proc.returncode == 0 and results_csv.exists():
                ok = True
                break
            print(f"[{i}/{total}] combo {dataset_name}/q{n_qubits}/seed{seed} "
                  f"FAILED (attempt {attempt}/{attempts}, exit code {proc.returncode})")

        if not ok:
            failed_combos.append((dataset_name, n_qubits, seed))
            print(f"[{i}/{total}] giving up on {dataset_name}/q{n_qubits}/seed{seed} "
                  f"after {attempts} attempt(s); continuing with remaining combos.")
            continue

        all_records.extend(read_records(results_csv))
        all_sanity_rows.extend(read_sanity_rows(sanity_csv))

        agg = aggregate_records(all_records)
        elapsed_so_far = time.time() - driver_start
        write_csv(f"{args.save}_results.csv", RECORD_FIELDS, all_records)
        write_csv(f"{args.save}_aggregate.csv",
                  ["dataset", "n_qubits", "run_type", "target_sparsity", "noise_level",
                   "n_seeds", "val_acc_mean", "val_acc_std", "val_acc_ci95",
                   "val_auc_mean", "val_auc_std", "val_auc_ci95",
                   "val_loss_mean", "val_loss_std", "val_loss_ci95", "converged_frac"], agg)
        if all_sanity_rows:
            write_csv(f"{args.save}_sanity_baseline.csv", SANITY_FIELDS, all_sanity_rows)
        for metric in ("val_acc", "val_auc", "val_loss"):
            higher = metric != "val_loss"
            gaps = compute_gap_table(all_records, metric, higher)
            write_csv(f"{args.save}_gap_{metric}.csv",
                      ["dataset", "n_qubits", "target_sparsity", "noise_level", "n_seeds",
                       "gap_mean", "gap_ci95", "p_value", "test_method", "significant"], gaps)
        write_report(args.report_name, config, all_records, agg, all_sanity_rows, elapsed_so_far)
        print(f"[{i}/{total}] merged {len(all_records)} records so far into "
              f"{args.save}_results.csv / {args.report_name}")

    total_elapsed = time.time() - driver_start

    print("\n" + "=" * 78)
    print(f"Dispatch complete | total run time: {format_duration(total_elapsed)}")
    print(f"Combos completed: {total - len(failed_combos)}/{total}")
    if failed_combos:
        print(f"Combos FAILED after retries ({len(failed_combos)}):")
        for dataset_name, n_qubits, seed in failed_combos:
            print(f"  - {dataset_name} / q{n_qubits} / seed{seed}")
        print("Re-run the same command with --skip-existing to retry only "
              "the remaining/failed combos.")
    if all_records:
        print(f"Wrote: {args.save}_results.csv, {args.save}_aggregate.csv, "
              f"{args.save}_gap_{{val_acc,val_auc,val_loss}}.csv, {args.report_name}")
    print("=" * 78)

    if failed_combos:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QAdaPrune-RigL (v9): adds a trainable classical readout, "
                     "larger circuit/step/seed/validation budgets, a convergence "
                     "check, AUC/loss metrics, and a classical sanity-check "
                     "baseline on top of v8's real-gradient pruning-decision fix. "
                     "See module docstring for the full diagnosis of why v8's "
                     "results (exp8.txt) were inconclusive."
    )
    parser.add_argument("-s", "--save", type=str, default="qadaprune_v9_run")
    parser.add_argument("--report-name", type=str, default="exp8.txt")
    parser.add_argument("--layers", type=int, default=4,
                         help="Raised from v8's default of 3 for more parameter "
                              "budget to prune (see module docstring, fix #2).")
    parser.add_argument("--steps", type=int, default=200,
                         help="Raised from v8's default of 40 -- 40 steps was very "
                              "likely leaving both dense and pruned models "
                              "undertrained (see module docstring, fix #3). Check "
                              "the 'converged' field / CAUTION line in the report; "
                              "raise further if runs are still flagged unconverged.")
    parser.add_argument("--win-sz", type=int, default=4)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="+",
                         default=["mnist", "fashionmnist"], choices=list(DATASET_CLASSES.keys()))
    parser.add_argument("--qubits", type=int, nargs="+", default=[6],
                         help="Raised from v8's default of 4 for more parameter "
                              "budget to prune (see module docstring, fix #2).")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.10, 0.25, 0.40])
    parser.add_argument("--seeds", type=int, nargs="+",
                         default=list(range(16)),
                         help="Raised from v8's default of 8 seeds. At the effect "
                              "sizes and per-seed std observed in v8's exp8.txt, 8 "
                              "seeds could never reach significance; see module "
                              "docstring, fix #5.")
    parser.add_argument("--encoding", type=str, default="pca", choices=["pca", "resize"])
    parser.add_argument("--n-train", type=int, default=800,
                         help="Raised from v8's default of 500 (see module "
                              "docstring, fix #4).")
    parser.add_argument("--n-val", type=int, default=800,
                         help="Raised from v8's default of 300 to shrink the "
                              "binomial-sampling component of run-to-run variance "
                              "(see module docstring, fix #4).")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    parser.add_argument("--noise-eval-samples", type=int, default=300,
                         help="Raised from v8's default of 100 (see module "
                              "docstring, fix #4).")
    parser.add_argument("--prune-start-frac", type=float, default=0.15)
    parser.add_argument("--prune-end-frac", type=float, default=0.65)
    parser.add_argument("--max-round-freeze-frac", type=float, default=0.08)
    parser.add_argument("--movement-ema-beta", type=float, default=0.6,
                         help="Decay used ONLY for the RigL grow-criterion |grad| "
                              "EMA (the movement-pruning score itself is a raw "
                              "accumulated sum per Sanh et al. -- see v8 docstring).")
    parser.add_argument("--cycle-frac-init", type=float, default=0.30)
    parser.add_argument("--protect-entangling-frac", type=float, default=0.34)
    parser.add_argument("--distill-hardness-max", type=float, default=0.7)
    parser.add_argument("--distill-hardness-min", type=float, default=0.2)
    parser.add_argument("--lr-warm-mult", type=float, default=2.0)
    parser.add_argument("--convergence-window-frac", type=float, default=0.15,
                         help="Fraction of steps used as the 'final window' for the "
                              "loss-plateau convergence check (see module docstring, "
                              "fix #3).")
    parser.add_argument("--convergence-rel-threshold", type=float, default=0.03,
                         help="A run is flagged 'converged' if the mean loss over "
                              "the final window changed by less than this fraction "
                              "relative to the window before it.")
    parser.add_argument("--skip-sanity-check", action="store_true",
                         help="Skip the classical LogisticRegression sanity-check "
                              "baseline (module docstring, fix #7). On by default "
                              "because it is cheap relative to circuit training.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--dispatch", action="store_true",
                         help="Instead of training every (dataset, n_qubits, seed) "
                              "combo in this one process, re-invoke this same "
                              "script once per combo as its own subprocess (without "
                              "--dispatch) and merge results after each one. Use "
                              "this if a single-process sweep crashes with "
                              "'custatevec memory allocation failed' -- see "
                              "run_dispatch() docstring above for why.")
    parser.add_argument("--work-dir", type=str, default="sweep_runs",
                         help="[--dispatch only] Directory for per-combo "
                              "intermediate files.")
    parser.add_argument("--skip-existing", action="store_true",
                         help="[--dispatch only] Reuse an existing combo results "
                              "CSV in --work-dir instead of re-training it.")
    parser.add_argument("--max-retries", type=int, default=1,
                         help="[--dispatch only] Extra attempts per combo if its "
                              "subprocess fails or doesn't produce a results CSV.")
    args = parser.parse_args()

    if args.dispatch:
        run_dispatch(args)
        sys.exit(0)

    v9_hparams = dict(
        movement_ema_beta=args.movement_ema_beta,
        cycle_frac_init=args.cycle_frac_init,
        protect_entangling_frac=args.protect_entangling_frac,
        distill_hardness_max=args.distill_hardness_max,
        distill_hardness_min=args.distill_hardness_min,
        lr_warm_mult=args.lr_warm_mult,
        convergence_window_frac=args.convergence_window_frac,
        convergence_rel_threshold=args.convergence_rel_threshold,
    )

    config = {
        "datasets": args.datasets, "qubits": args.qubits, "sparsities": args.sparsities,
        "seeds": args.seeds, "encoding": args.encoding, "noise_levels": args.noise_levels,
        "layers": args.layers, "steps": args.steps, "win_sz": args.win_sz, "shots": args.shots,
        "n_train": args.n_train, "n_val": args.n_val,
        "prune_start_frac": args.prune_start_frac, "prune_end_frac": args.prune_end_frac,
        "max_round_freeze_frac": args.max_round_freeze_frac,
        "noise_eval_samples": args.noise_eval_samples,
        **{f"v9_{k}": v for k, v in v9_hparams.items()},
    }
    print("Run config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    if len(args.seeds) < MIN_SEEDS_FOR_TEST:
        print(f"WARNING: {len(args.seeds)} seeds < {MIN_SEEDS_FOR_TEST} -- gap-table "
              f"significance tests will be skipped for every cell.")
    n_combos = len(args.datasets) * len(args.qubits) * len(args.seeds)
    n_runs = n_combos * (1 + len(args.sparsities))
    print(f"\nThis sweep will train {n_runs} models "
          f"({n_combos} combos x (1 no-pruning + {len(args.sparsities)} sparsities)).")

    run_start = time.time()
    all_records, all_sanity_rows, raw_bundles = run_experiment_grid(
        args.datasets, args.qubits, args.sparsities, args.seeds, args.noise_levels,
        n_layers=args.layers, steps=args.steps, win_sz=args.win_sz, shots=args.shots,
        prune_start_frac=args.prune_start_frac, prune_end_frac=args.prune_end_frac,
        max_round_freeze_frac=args.max_round_freeze_frac,
        noise_eval_samples=args.noise_eval_samples, quiet=args.quiet,
        encoding=args.encoding, n_train=args.n_train, n_val=args.n_val,
        skip_sanity_check=args.skip_sanity_check,
        save_prefix=args.save, save_raw=args.save_raw, v9_hparams=v9_hparams,
    )
    total_elapsed = time.time() - run_start

    agg = aggregate_records(all_records)

    write_csv(f"{args.save}_results.csv", RECORD_FIELDS, all_records)
    write_csv(f"{args.save}_aggregate.csv",
              ["dataset", "n_qubits", "run_type", "target_sparsity", "noise_level",
               "n_seeds", "val_acc_mean", "val_acc_std", "val_acc_ci95",
               "val_auc_mean", "val_auc_std", "val_auc_ci95",
               "val_loss_mean", "val_loss_std", "val_loss_ci95", "converged_frac"], agg)
    if all_sanity_rows:
        write_csv(f"{args.save}_sanity_baseline.csv", SANITY_FIELDS, all_sanity_rows)
    for metric in ("val_acc", "val_auc", "val_loss"):
        higher = metric != "val_loss"
        gaps = compute_gap_table(all_records, metric, higher)
        write_csv(f"{args.save}_gap_{metric}.csv",
                  ["dataset", "n_qubits", "target_sparsity", "noise_level", "n_seeds",
                   "gap_mean", "gap_ci95", "p_value", "test_method", "significant"], gaps)
    write_report(args.report_name, config, all_records, agg, all_sanity_rows, total_elapsed)

    print("\n" + "=" * 78)
    print(f"Sweep complete | total run time: {format_duration(total_elapsed)}")
    print(f"Wrote: {args.save}_results.csv, {args.save}_aggregate.csv, "
          f"{args.save}_gap_{{val_acc,val_auc,val_loss}}.csv, "
          f"{args.save}_sanity_baseline.csv, {args.report_name}"
          + (f", {args.save}_raw.pkl" if args.save_raw else ""))
    print("=" * 78)