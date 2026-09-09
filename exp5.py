import argparse
import csv
import math
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

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
# Data
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


def load_binary_subset(dataset_name, n_qubits, n_train=500, n_val=300, seed=42,
                        root="./data", encoding="pca"):
    """Returns (X_train, y_train), (X_val, y_val) with X shaped
    (N, n_qubits, IMG_COLS), scaled to [0, pi].

    encoding="pca" (default, v6): fit a 16-component PCA on the training
    crops, min-max scale each component using train-set stats (val is
    clipped to the same range, not re-fit -- no val-set leakage).
    encoding="resize" (v4/v5 behavior, kept for comparison): CenterCrop
    then a naive box-filter resize to (n_qubits, IMG_COLS).
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


# ----------------------------------------------------------------------
# Circuit (training path, torch-differentiable) -- UNCHANGED from v5
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


def reduce_to_two(out_vals):
    n = len(out_vals)
    half = n // 2
    c0 = sum(out_vals[:half]) if half > 0 else out_vals[0] * 0
    c1 = sum(out_vals[half:])
    return torch.stack([c0, c1])


def batch_forward(qckt, params, X_t):
    logits = torch.stack([reduce_to_two(qckt(params, x)) for x in X_t])
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


# ----------------------------------------------------------------------
# Gradient / saliency helpers  -- BUG FIXES ARE HERE
# ----------------------------------------------------------------------
def compute_sample_grad(qckt, params, X_batch, y_batch):
    """Mean per-example gradient of the REAL task loss (true label),
    averaged over X_batch/y_batch (len == grad_pts).

    v5 BUG: this function used to ignore y entirely and always compute
    the gradient against a hard-coded target of 0, so on a balanced
    binary task roughly half the importance/regrow signal pointed the
    wrong way. Fixed here by actually using the sample's label.
    """
    grads = []
    for x, y in zip(X_batch, y_batch):
        p = params.detach().clone().requires_grad_(True)
        probs = batch_forward(qckt, p, [x])
        target = torch.tensor([float(y)], dtype=DTYPE, device=p.device)
        loss = bce_loss(probs, target)
        (g,) = torch.autograd.grad(loss, p)
        grads.append(g.detach().cpu().numpy())
    return np.mean(grads, axis=0)


def sample_grad_batch(rng, n_available, grad_pts):
    """v5 BUG: importance gradients were always measured on the fixed
    first `grad_pts` training examples, every round, every step. Fixed
    here: draw a fresh random subset (no replacement) each call."""
    n = min(grad_pts, n_available)
    return rng.choice(n_available, size=n, replace=False)


def get_sparsity(arr):
    return 1.0 - np.count_nonzero(arr) / arr.size


def enforce_frozen(params, frozen_mask, param_shape):
    if not frozen_mask.any():
        return
    with torch.no_grad():
        flat = params.flatten()
        flat[frozen_mask] = 0
        params.data = flat.reshape(param_shape)


def top_up_to_target_sparsity(params, frozen_mask, protected_mask, grad_buffer,
                               target_sparsity, param_shape):
    n_params = frozen_mask.size
    n_target = int(np.ceil(target_sparsity * n_params))
    n_frozen = int(frozen_mask.sum())
    if n_frozen >= n_target:
        return frozen_mask
    remaining = np.where(~frozen_mask & ~protected_mask)[0]
    if remaining.size == 0:
        return frozen_mask
    order = remaining[np.argsort(np.abs(grad_buffer.flatten()[remaining]))]
    need = n_target - n_frozen
    to_freeze = order[:need]
    frozen_mask = frozen_mask.copy()
    frozen_mask[to_freeze] = True
    enforce_frozen(params, frozen_mask, param_shape)
    return frozen_mask


def taylor_importance(params_np, grad_buffer):
    flat_params = np.abs(params_np.flatten())
    flat_grad = np.abs(grad_buffer.flatten())
    return flat_grad * flat_params


def cubic_sparsity_at_step(t, start_step, end_step, final_sparsity, initial_sparsity=0.0):
    if t <= start_step:
        return initial_sparsity
    if t >= end_step:
        return final_sparsity
    progress = (t - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3


def entangling_protected_mask(n_layers, n_qubits, protect_frac, seed=0):
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


def regrow_fraction_schedule(t, start_step, end_step, init_frac):
    if t <= start_step or t >= end_step or init_frac <= 0:
        return 0.0
    progress = (t - start_step) / (end_step - start_step)
    return init_frac * 0.5 * (1 + math.cos(math.pi * progress))


def select_prune_and_regrow_round(params, frozen_mask, protected_mask, candidate_counts,
                                   importance_ema, grad_ema, n_target, param_shape,
                                   max_round_freeze, regrow_frac, hysteresis_rounds):
    frozen_mask = frozen_mask.copy()
    candidate_counts = candidate_counts.copy()

    frozen_eligible = np.where(frozen_mask & ~protected_mask)[0]
    n_regrow = int(round(regrow_frac * frozen_eligible.size))
    regrown = np.array([], dtype=int)
    if n_regrow > 0:
        grad_mag = np.abs(grad_ema.flatten()[frozen_eligible])
        order = frozen_eligible[np.argsort(-grad_mag)]
        regrown = order[:n_regrow]
        frozen_mask[regrown] = False
        candidate_counts[regrown] = 0

    n_frozen_now = int(frozen_mask.sum())
    need = max(0, n_target - n_frozen_now)
    if need > 0:
        eligible = np.where(~frozen_mask & ~protected_mask)[0]
        if eligible.size > 0:
            importance = importance_ema.flatten()
            order = eligible[np.argsort(importance[eligible])]
            pool_size = min(eligible.size, max(need * 2, max_round_freeze))
            pool = order[:pool_size]
            candidate_counts[pool] += 1
            not_in_pool = np.setdiff1d(np.where(~frozen_mask & ~protected_mask)[0], pool)
            candidate_counts[not_in_pool] = 0
            confirmed = pool[candidate_counts[pool] >= hysteresis_rounds]
            confirmed = confirmed[np.argsort(importance[confirmed])]
            to_freeze = confirmed[:min(need, max_round_freeze)]
            if to_freeze.size > 0:
                frozen_mask[to_freeze] = True
                candidate_counts[to_freeze] = 0

    enforce_frozen(params, frozen_mask, param_shape)
    return frozen_mask, candidate_counts, regrown.size


# ----------------------------------------------------------------------
# Noisy (density-matrix) evaluation path -- UNCHANGED from v5
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


def evaluate_noise_robustness(final_params_np, X_val, y_val, n_qubits, noise_levels,
                               max_eval_samples=100, freeze_tol=1e-9, eval_seed=0):
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
        correct = 0
        for x, y in zip(Xs, ys):
            out = np.array(qnode(noise_prob, x), dtype=float)
            half = len(out) // 2
            c0 = out[:half].sum() if half > 0 else out[0]
            c1 = out[half:].sum()
            probs = numpy_softmax(np.array([c0, c1]))
            pred = int(probs[1] > 0.5)
            correct += int(pred == int(y))
        results[noise_prob] = correct / n_eval
    return results


# ----------------------------------------------------------------------
# Reporting helpers
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
def optimize_and_prune(qckt, iparams, X_train, y_train, X_val, y_val,
                        win_sz=4, steps=40, tol=0.01, lr=0.1,
                        target_sparsity=0.25, grad_pts=16, compute_device=None,
                        prune_start_frac=0.15, prune_end_frac=0.65,
                        max_round_freeze_frac=0.08, verbose=True,
                        imp_ema_beta=0.6, hysteresis_rounds=2,
                        regrow_frac_init=0.30, protect_entangling_frac=0.34,
                        seed=0, teacher_params=None,
                        distill_hardness_max=0.7, distill_hardness_min=0.2,
                        lr_warm_mult=2.0):
    compute_device = compute_device if compute_device is not None else TORCH_DEVICE
    X_t = torch.tensor(X_train, dtype=DTYPE, device=compute_device)
    y_t = torch.tensor(y_train, dtype=DTYPE, device=compute_device)
    Xv_t = torch.tensor(X_val, dtype=DTYPE, device=compute_device)
    yv_t = torch.tensor(y_val, dtype=DTYPE, device=compute_device)
    n_train = X_t.shape[0]

    params = torch.tensor(iparams, dtype=DTYPE, device=compute_device, requires_grad=True)
    param_shape = params.shape
    n_layers, n_qubits, _ = param_shape
    n_params = int(np.prod(param_shape))

    opt = torch.optim.RMSprop([params], lr=lr, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)

    start_step = max(win_sz, int(round(prune_start_frac * steps)))
    end_step = max(start_step + win_sz, int(round(prune_end_frac * steps)))
    max_round_freeze = max(1, int(np.ceil(max_round_freeze_frac * n_params)))

    protected_mask = entangling_protected_mask(n_layers, n_qubits, protect_entangling_frac, seed=seed)

    teacher_t = None
    if teacher_params is not None:
        teacher_t = torch.tensor(teacher_params, dtype=DTYPE, device=compute_device)

    # v6: dedicated RNG for drawing a FRESH grad_pts subset each measurement
    # (v5 always reused the fixed first grad_pts examples -- see docstring).
    grad_rng = np.random.RandomState(seed * 100003 + 7)

    def fresh_grad(p):
        idx = sample_grad_batch(grad_rng, n_train, grad_pts)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=compute_device)
        return compute_sample_grad(qckt, p, X_t[idx_t], y_t[idx_t])

    tau = np.ones(n_params) / n_params
    grad_buffer = fresh_grad(params).flatten()
    importance_ema = np.abs(grad_buffer) * np.abs(params.detach().cpu().numpy().flatten())
    grad_ema = np.abs(grad_buffer).copy()
    frozen_mask = np.zeros(n_params, dtype=bool)
    candidate_counts = np.zeros(n_params, dtype=int)

    cost_hists, train_acc_hists = [], []
    step_records = []
    steps_since_round = win_sz

    if verbose:
        print("\ntraining with QAdaPrune-R (v6) pruning:")
    loop_start = time.time()
    for t in range(steps):
        step_start = time.time()

        warm_progress = min(1.0, steps_since_round / max(1, win_sz))
        cur_lr = lr + (lr * lr_warm_mult - lr) * (1 - warm_progress)
        opt.param_groups[0]["lr"] = cur_lr

        if teacher_t is not None:
            explore_progress = min(1.0, max(0.0, t / max(1, end_step)))
            hardness = distill_hardness_max + (distill_hardness_min - distill_hardness_max) * explore_progress
        else:
            hardness = 0.0

        cur_grad = fresh_grad(params).flatten()
        tau_prime = tau * (1 - np.abs(cur_grad))

        opt.zero_grad()
        probs = batch_forward(qckt, params, X_t)
        loss = bce_loss(probs, y_t)
        if teacher_t is not None and hardness > 0:
            with torch.no_grad():
                teacher_probs = batch_forward(qckt, teacher_t, X_t)
            loss = (1 - hardness) * loss + hardness * kd_kl_loss(probs, teacher_probs)
        loss.backward()
        opt.step()

        enforce_frozen(params, frozen_mask, param_shape)

        next_grad = fresh_grad(params).flatten()
        grad_buffer = grad_buffer + np.abs(cur_grad - next_grad)

        new_cost = loss.item()
        cost_hists.append(new_cost)
        with torch.no_grad():
            train_acc = accuracy_from_probs(batch_forward(qckt, params, X_t), y_t)
        train_acc_hists.append(train_acc)

        n_regrown_this_step = 0
        steps_since_round += 1
        if t != 0 and t % win_sz == 0:
            sparsity_target_now = cubic_sparsity_at_step(t, start_step, end_step, target_sparsity)
            n_target = int(np.ceil(sparsity_target_now * n_params))

            round_importance = taylor_importance(params.detach().cpu().numpy(), grad_buffer)
            importance_ema = imp_ema_beta * importance_ema + (1 - imp_ema_beta) * round_importance
            grad_ema = imp_ema_beta * grad_ema + (1 - imp_ema_beta) * np.abs(grad_buffer)

            regrow_frac = regrow_fraction_schedule(t, start_step, end_step, regrow_frac_init)

            frozen_mask, candidate_counts, n_regrown_this_step = select_prune_and_regrow_round(
                params, frozen_mask, protected_mask, candidate_counts,
                importance_ema, grad_ema, n_target, param_shape,
                max_round_freeze, regrow_frac, hysteresis_rounds,
            )
            grad_buffer = next_grad.copy()
            steps_since_round = 0

        tau = tau_prime

        step_sparsity = get_sparsity(params.detach().cpu().numpy())
        step_time = time.time() - step_start
        step_records.append((t + 1, train_acc, step_sparsity, new_cost, step_time))
        if verbose:
            print_prune_step(t + 1, train_acc, step_sparsity, new_cost, step_time, n_regrown_this_step)

        if new_cost < tol:
            break

    frozen_mask = top_up_to_target_sparsity(
        params, frozen_mask, protected_mask, grad_buffer, target_sparsity, param_shape
    )

    sparsity = get_sparsity(params.detach().cpu().numpy())
    with torch.no_grad():
        val_probs = batch_forward(qckt, params, Xv_t)
        val_acc = accuracy_from_probs(val_probs, yv_t)

    total_time = time.time() - loop_start
    final_train_acc = step_records[-1][1] if step_records else float("nan")
    final_train_loss = step_records[-1][3] if step_records else float("nan")
    print_prune_cumulative(final_train_acc, sparsity, final_train_loss, total_time)
    print(f"(validation accuracy on held-out set: {val_acc:.4f})")

    return {
        "step_records": step_records,
        "cost_hists": cost_hists,
        "train_acc_hists": train_acc_hists,
        "sparsity": sparsity,
        "val_acc": val_acc,
        "total_time": total_time,
        "final_params": params.detach().cpu().numpy(),
    }


def optimize(qckt, iparams, X_train, y_train, X_val, y_val, steps=100, tol=0.01, lr=0.1,
             compute_device=None, verbose=True):
    compute_device = compute_device if compute_device is not None else TORCH_DEVICE
    X_t = torch.tensor(X_train, dtype=DTYPE, device=compute_device)
    y_t = torch.tensor(y_train, dtype=DTYPE, device=compute_device)
    Xv_t = torch.tensor(X_val, dtype=DTYPE, device=compute_device)
    yv_t = torch.tensor(y_val, dtype=DTYPE, device=compute_device)

    params = torch.tensor(iparams, dtype=DTYPE, device=compute_device, requires_grad=True)
    opt = torch.optim.RMSprop([params], lr=lr, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)

    cost_hists, train_acc_hists = [], []
    step_records = []

    if verbose:
        print("\ntraining without pruning (dense teacher):")
    loop_start = time.time()
    for t in range(steps):
        step_start = time.time()

        opt.zero_grad()
        probs = batch_forward(qckt, params, X_t)
        loss = bce_loss(probs, y_t)
        loss.backward()
        opt.step()

        new_cost = loss.item()
        cost_hists.append(new_cost)
        with torch.no_grad():
            train_acc = accuracy_from_probs(batch_forward(qckt, params, X_t), y_t)
        train_acc_hists.append(train_acc)

        step_time = time.time() - step_start
        step_records.append((t + 1, train_acc, new_cost, step_time))
        if verbose:
            print_no_prune_step(t + 1, train_acc, new_cost, step_time)

        if new_cost < tol:
            break

    with torch.no_grad():
        val_probs = batch_forward(qckt, params, Xv_t)
        val_acc = accuracy_from_probs(val_probs, yv_t)

    total_time = time.time() - loop_start
    final_train_acc = step_records[-1][1] if step_records else float("nan")
    final_train_loss = step_records[-1][2] if step_records else float("nan")
    print_no_prune_cumulative(final_train_acc, final_train_loss, total_time)
    print(f"(validation accuracy on held-out set: {val_acc:.4f})")

    return {
        "step_records": step_records,
        "cost_hists": cost_hists,
        "train_acc_hists": train_acc_hists,
        "val_acc": val_acc,
        "total_time": total_time,
        "final_params": params.detach().cpu().numpy(),
    }


# ----------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------
RECORD_FIELDS = [
    "dataset", "n_qubits", "seed", "run_type", "algorithm", "target_sparsity",
    "achieved_sparsity", "n_params", "n_params_active",
    "train_final_acc", "train_final_loss",
    "noise_level", "val_acc", "total_time_s",
]


def _build_rows(dataset, n_qubits, seed, run_type, algorithm, target_sparsity, achieved_sparsity,
                 n_params, train_final_acc, train_final_loss, val_acc_noiseless,
                 noisy_results, total_time):
    n_active = int(round((1 - achieved_sparsity) * n_params))
    rows = [{
        "dataset": dataset, "n_qubits": n_qubits, "seed": seed, "run_type": run_type,
        "algorithm": algorithm,
        "target_sparsity": target_sparsity, "achieved_sparsity": achieved_sparsity,
        "n_params": n_params, "n_params_active": n_active,
        "train_final_acc": train_final_acc, "train_final_loss": train_final_loss,
        "noise_level": 0.0, "val_acc": val_acc_noiseless, "total_time_s": total_time,
    }]
    for noise_level, acc in sorted(noisy_results.items()):
        rows.append({
            "dataset": dataset, "n_qubits": n_qubits, "seed": seed, "run_type": run_type,
            "algorithm": algorithm,
            "target_sparsity": target_sparsity, "achieved_sparsity": achieved_sparsity,
            "n_params": n_params, "n_params_active": n_active,
            "train_final_acc": train_final_acc, "train_final_loss": train_final_loss,
            "noise_level": noise_level, "val_acc": acc, "total_time_s": total_time,
        })
    return rows


def run_single(dataset_name, n_qubits, seed, n_layers, steps, win_sz, sparsities,
               noise_levels, prune_start_frac, prune_end_frac, max_round_freeze_frac,
               shots, noise_eval_samples, quiet, encoding, v5_hparams):
    (X_train, y_train), (X_val, y_val) = load_binary_subset(
        dataset_name, n_qubits, seed=seed, encoding=encoding
    )

    np.random.seed(seed)
    init_params = np.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 2))
    n_params = n_layers * n_qubits * 2

    qckt, is_gpu_sim = make_qnode(n_qubits, shots)
    compute_device = TORCH_DEVICE if is_gpu_sim else torch.device("cpu")
    print(f"[{dataset_name}/q{n_qubits}/seed{seed}] circuit device: {compute_device} "
          f"({'lightning.gpu' if is_gpu_sim else 'default.qubit fallback'})")

    no_prune_result = optimize(
        qckt, init_params.copy(), X_train, y_train, X_val, y_val, steps=steps,
        compute_device=compute_device, verbose=not quiet,
    )
    noise_np = evaluate_noise_robustness(
        no_prune_result["final_params"], X_val, y_val, n_qubits, noise_levels,
        max_eval_samples=noise_eval_samples, eval_seed=seed,
    )
    records = _build_rows(
        dataset_name, n_qubits, seed, "no_pruning", "dense", None, 0.0, n_params,
        no_prune_result["step_records"][-1][1], no_prune_result["step_records"][-1][2],
        no_prune_result["val_acc"], noise_np, no_prune_result["total_time"],
    )

    pruned_results = {}
    for ts in sparsities:
        prune_result = optimize_and_prune(
            qckt, init_params.copy(), X_train, y_train, X_val, y_val,
            win_sz=win_sz, steps=steps, target_sparsity=ts, compute_device=compute_device,
            prune_start_frac=prune_start_frac, prune_end_frac=prune_end_frac,
            max_round_freeze_frac=max_round_freeze_frac, verbose=not quiet,
            teacher_params=no_prune_result["final_params"], seed=seed,
            **v5_hparams,
        )
        noise_p = evaluate_noise_robustness(
            prune_result["final_params"], X_val, y_val, n_qubits, noise_levels,
            max_eval_samples=noise_eval_samples, eval_seed=seed,
        )
        records.extend(_build_rows(
            dataset_name, n_qubits, seed, "pruned", "qadaprune_r_v6", ts, prune_result["sparsity"],
            n_params, prune_result["step_records"][-1][1], prune_result["step_records"][-1][3],
            prune_result["val_acc"], noise_p, prune_result["total_time"],
        ))
        pruned_results[ts] = prune_result

    raw_bundle = {
        "dataset": dataset_name, "n_qubits": n_qubits, "seed": seed,
        "no_pruning": no_prune_result, "pruned": pruned_results,
    }
    return records, raw_bundle


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def aggregate_records(records):
    groups = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["n_qubits"], r["run_type"], r["target_sparsity"], r["noise_level"])
        groups[key].append(r["val_acc"])
    agg = []
    for (dataset, n_qubits, run_type, target_sparsity, noise_level), vals in groups.items():
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        agg.append({
            "dataset": dataset, "n_qubits": n_qubits, "run_type": run_type,
            "target_sparsity": target_sparsity, "noise_level": noise_level,
            "n_seeds": n, "val_acc_mean": mean, "val_acc_std": std, "val_acc_ci95": ci95,
        })
    agg.sort(key=lambda r: (r["dataset"], r["n_qubits"], r["run_type"] != "no_pruning",
                             r["target_sparsity"] or 0, r["noise_level"]))
    return agg


def _paired_test(baseline_vals, pruned_vals):
    """Paired significance test on (baseline - pruned) per seed. Prefers
    Wilcoxon signed-rank (nonparametric, robust to the small/unequal
    sample sizes typical here); falls back to a paired t-test if scipy
    is unavailable or the sample is degenerate. Returns (p_value or
    None, method_str)."""
    diffs = np.array(baseline_vals) - np.array(pruned_vals)
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
            stat, p = ttest_rel(baseline_vals, pruned_vals)
            return float(p), "paired_t"
        except Exception:
            return None, "scipy_unavailable"


def compute_gap_table(records, alpha=0.05):
    lookup = defaultdict(dict)
    for r in records:
        key = (r["dataset"], r["n_qubits"], r["seed"], r["noise_level"])
        if r["run_type"] == "no_pruning":
            lookup[key]["baseline"] = r["val_acc"]
        else:
            lookup[key].setdefault("pruned", {})[r["target_sparsity"]] = r["val_acc"]

    paired = defaultdict(list)  # key -> list of (baseline, pruned) per seed
    for (dataset, n_qubits, seed, noise_level), d in lookup.items():
        if "baseline" not in d or "pruned" not in d:
            continue
        for sparsity, acc in d["pruned"].items():
            paired[(dataset, n_qubits, sparsity, noise_level)].append((d["baseline"], acc))

    gap_rows = []
    for (dataset, n_qubits, sparsity, noise_level), pairs in paired.items():
        baseline_vals = [p[0] for p in pairs]
        pruned_vals = [p[1] for p in pairs]
        vals = [b - p for b, p in pairs]
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        p_value, test_method = _paired_test(baseline_vals, pruned_vals)
        significant = (p_value is not None) and (p_value < alpha)
        gap_rows.append({
            "dataset": dataset, "n_qubits": n_qubits, "target_sparsity": sparsity,
            "noise_level": noise_level, "n_seeds": n,
            "accuracy_gap_mean": mean, "accuracy_gap_std": std, "accuracy_gap_ci95": ci95,
            "p_value": p_value, "test_method": test_method, "significant": significant,
        })
    gap_rows.sort(key=lambda r: (r["dataset"], r["n_qubits"], r["target_sparsity"], r["noise_level"]))
    return gap_rows


def write_report(path, config, records, agg, gaps, total_elapsed):
    with open(path, "w") as f:
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("Algorithm: QAdaPrune-R (v6) -- see qadaprune_v6.py module docstring "
                "for the v5 bugs this fixes and the rationale.\n")
        f.write(f"Torch device: {TORCH_DEVICE}\n")
        f.write("Run config:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal sweep run time: {format_duration(total_elapsed)} ({total_elapsed:.2f}s)\n")
        f.write(f"Total records: {len(records)}\n\n")

        f.write("=" * 78 + "\n")
        f.write("VALIDATION ACCURACY (mean +/- 95% CI across seeds)\n")
        f.write("=" * 78 + "\n")
        f.write(f"{'dataset':<14}{'q':<4}{'run_type':<12}{'sparsity':<10}"
                f"{'noise':<8}{'n_seeds':<9}{'acc_mean':<10}{'acc_std':<10}{'ci95':<8}\n")
        for r in agg:
            sparsity_str = "-" if r["target_sparsity"] is None else f"{r['target_sparsity']:.2f}"
            f.write(f"{r['dataset']:<14}{r['n_qubits']:<4}{r['run_type']:<12}{sparsity_str:<10}"
                    f"{r['noise_level']:<8.3f}{r['n_seeds']:<9}{r['val_acc_mean']:<10.4f}"
                    f"{r['val_acc_std']:<10.4f}{r['val_acc_ci95']:<8.4f}\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("PAIRED ACCURACY GAP (no_pruning - pruned, per seed) + significance\n")
        f.write("=" * 78 + "\n")
        f.write(f"{'dataset':<14}{'q':<4}{'sparsity':<10}{'noise':<8}{'n_seeds':<9}"
                f"{'gap_mean':<10}{'ci95':<8}{'p_value':<10}{'sig?':<6}method\n")
        for r in gaps:
            p_str = "n/a" if r["p_value"] is None else f"{r['p_value']:.4f}"
            sig_str = "YES" if r["significant"] else "no"
            f.write(f"{r['dataset']:<14}{r['n_qubits']:<4}{r['target_sparsity']:<10.2f}"
                    f"{r['noise_level']:<8.3f}{r['n_seeds']:<9}{r['accuracy_gap_mean']:<10.4f}"
                    f"{r['accuracy_gap_ci95']:<8.4f}{p_str:<10}{sig_str:<6}{r['test_method']}\n")

        n_underpowered = sum(1 for r in gaps if r["test_method"] == "insufficient_seeds")
        f.write(f"\nCAUTION: {n_underpowered}/{len(gaps)} cells have fewer than "
                f"{MIN_SEEDS_FOR_TEST} seeds -- no significance test was run for those, "
                f"and point-estimate differences between sparsity levels there should NOT "
                f"be read as findings. Only rows marked sig?=YES should be treated as "
                f"a real difference from the dense baseline; everything else is "
                f"consistent with noise at the tested seed count.\n")
        f.write("\nNote: noise_level=0.0 rows use the noiseless (fast) simulator; "
                "noise_level>0.0 rows use a density-matrix simulator with a "
                "DepolarizingChannel inserted after every gate that is actually "
                "applied (frozen/zeroed gates are skipped).\n")
        f.write("See {results,aggregate,gap}.csv for the full machine-readable tables.\n")


def run_experiment_grid(datasets, qubit_counts, sparsities, seeds, noise_levels,
                         n_layers, steps, win_sz, shots,
                         prune_start_frac, prune_end_frac,
                         max_round_freeze_frac, noise_eval_samples,
                         quiet, encoding, save_prefix, save_raw, v5_hparams):
    combos = [(d, q, s) for d in datasets for q in qubit_counts for s in seeds]
    total = len(combos)
    all_records = []
    raw_bundles = []

    for i, (dataset_name, n_qubits, seed) in enumerate(combos, 1):
        print(f"\n{'#' * 78}\n[{i}/{total}] dataset={dataset_name} qubits={n_qubits} seed={seed}\n{'#' * 78}")
        records, raw_bundle = run_single(
            dataset_name, n_qubits, seed, n_layers, steps, win_sz, sparsities,
            noise_levels, prune_start_frac, prune_end_frac, max_round_freeze_frac,
            shots, noise_eval_samples, quiet, encoding, v5_hparams,
        )
        all_records.extend(records)
        raw_bundles.append(raw_bundle)

        write_csv(f"{save_prefix}_results.csv", RECORD_FIELDS, all_records)
        if save_raw:
            with open(f"{save_prefix}_raw.pkl", "wb") as f:
                pickle.dump(raw_bundles, f)

    return all_records, raw_bundles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QAdaPrune-R (v6): fixes v5's mislabeled-gradient bug and fixed-batch "
                     "importance sampling, adds a PCA encoding option and paired "
                     "significance testing. See module docstring for the full diagnosis."
    )
    parser.add_argument("-s", "--save", type=str, default="qadaprune_v6_run")
    parser.add_argument("--report-name", type=str, default="exp5.txt")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--win-sz", type=int, default=4)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="+",
                         default=["mnist", "fashionmnist"], choices=list(DATASET_CLASSES.keys()))
    parser.add_argument("--qubits", type=int, nargs="+", default=[4])
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.10, 0.25, 0.40])
    parser.add_argument("--seeds", type=int, nargs="+",
                         default=[0, 1, 2, 3, 4, 5, 6, 7],
                         help="Default raised from v5's 3 seeds to 8: n=3 gives CIs of "
                              "+/-5 to +/-12pp on these tasks, too wide to distinguish "
                              "sparsity levels. Use --seeds with fewer values to trade "
                              "statistical power for run time.")
    parser.add_argument("--encoding", type=str, default="pca", choices=["pca", "resize"],
                         help="pca (default, v6): 16-component PCA projection, preserves "
                              "far more class-discriminative signal per qubit than a naive "
                              "spatial resize. resize: v4/v5's original box-filter downsample, "
                              "kept for direct comparison.")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    parser.add_argument("--noise-eval-samples", type=int, default=100)
    parser.add_argument("--prune-start-frac", type=float, default=0.15)
    parser.add_argument("--prune-end-frac", type=float, default=0.65)
    parser.add_argument("--max-round-freeze-frac", type=float, default=0.08)
    parser.add_argument("--grad-pts", type=int, default=16)
    parser.add_argument("--imp-ema-beta", type=float, default=0.6)
    parser.add_argument("--hysteresis-rounds", type=int, default=2)
    parser.add_argument("--regrow-frac-init", type=float, default=0.30)
    parser.add_argument("--protect-entangling-frac", type=float, default=0.34)
    parser.add_argument("--distill-hardness-max", type=float, default=0.7)
    parser.add_argument("--distill-hardness-min", type=float, default=0.2)
    parser.add_argument("--lr-warm-mult", type=float, default=2.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    v5_hparams = dict(
        grad_pts=args.grad_pts,
        imp_ema_beta=args.imp_ema_beta,
        hysteresis_rounds=args.hysteresis_rounds,
        regrow_frac_init=args.regrow_frac_init,
        protect_entangling_frac=args.protect_entangling_frac,
        distill_hardness_max=args.distill_hardness_max,
        distill_hardness_min=args.distill_hardness_min,
        lr_warm_mult=args.lr_warm_mult,
    )

    config = {
        "datasets": args.datasets, "qubits": args.qubits, "sparsities": args.sparsities,
        "seeds": args.seeds, "encoding": args.encoding, "noise_levels": args.noise_levels,
        "layers": args.layers, "steps": args.steps, "win_sz": args.win_sz, "shots": args.shots,
        "prune_start_frac": args.prune_start_frac, "prune_end_frac": args.prune_end_frac,
        "max_round_freeze_frac": args.max_round_freeze_frac,
        "noise_eval_samples": args.noise_eval_samples,
        **{f"v5_{k}": v for k, v in v5_hparams.items()},
    }
    print("Run config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    if len(args.seeds) < MIN_SEEDS_FOR_TEST:
        print(f"WARNING: {len(args.seeds)} seeds < {MIN_SEEDS_FOR_TEST} -- gap-table "
              f"significance tests will be skipped for every cell (marked "
              f"'insufficient_seeds'). Point estimates from a run this small should not "
              f"be reported as findings.")
    n_combos = len(args.datasets) * len(args.qubits) * len(args.seeds)
    n_runs = n_combos * (1 + len(args.sparsities))
    print(f"\nThis sweep will train {n_runs} models "
          f"({n_combos} combos x (1 no-pruning + {len(args.sparsities)} sparsities)).")

    run_start = time.time()
    all_records, raw_bundles = run_experiment_grid(
        args.datasets, args.qubits, args.sparsities, args.seeds, args.noise_levels,
        n_layers=args.layers, steps=args.steps, win_sz=args.win_sz, shots=args.shots,
        prune_start_frac=args.prune_start_frac, prune_end_frac=args.prune_end_frac,
        max_round_freeze_frac=args.max_round_freeze_frac,
        noise_eval_samples=args.noise_eval_samples, quiet=args.quiet,
        encoding=args.encoding, save_prefix=args.save, save_raw=args.save_raw,
        v5_hparams=v5_hparams,
    )
    total_elapsed = time.time() - run_start

    agg = aggregate_records(all_records)
    gaps = compute_gap_table(all_records)

    write_csv(f"{args.save}_results.csv", RECORD_FIELDS, all_records)
    write_csv(f"{args.save}_aggregate.csv",
              ["dataset", "n_qubits", "run_type", "target_sparsity", "noise_level",
               "n_seeds", "val_acc_mean", "val_acc_std", "val_acc_ci95"], agg)
    write_csv(f"{args.save}_gap.csv",
              ["dataset", "n_qubits", "target_sparsity", "noise_level", "n_seeds",
               "accuracy_gap_mean", "accuracy_gap_std", "accuracy_gap_ci95",
               "p_value", "test_method", "significant"], gaps)
    write_report(args.report_name, config, all_records, agg, gaps, total_elapsed)

    print("\n" + "=" * 78)
    print(f"Sweep complete | total run time: {format_duration(total_elapsed)}")
    print(f"Wrote: {args.save}_results.csv, {args.save}_aggregate.csv, "
          f"{args.save}_gap.csv, {args.report_name}"
          + (f", {args.save}_raw.pkl" if args.save_raw else ""))
    print("=" * 78)