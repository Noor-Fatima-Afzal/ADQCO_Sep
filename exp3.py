"""
QAdaPrune experiment sweep: multiple datasets x qubit counts x sparsity
levels x seeds, plus depolarizing-noise robustness evaluation.

  * Datasets:  MNIST (digits 3 vs 6) / FashionMNIST (dress vs shirt), or
               any other binary pair added to DATASET_CLASSES below.
               Images center-cropped to 24x24, downsampled to
               (n_qubits, 4) -- one row of 4 pixel values per qubit,
               matching the encoding sub-circuit's requirement of exactly
               4 rotation angles (RY,RX,RZ,RY) per qubit (Fig. 2).
               500 training images, 300 held-out validation images
               (split evenly across the two classes), matching the
               paper's "500 front images ... 300 randomly chosen
               validation" setup.
  * Circuit:  data-encoding sub-circuit applies RY, RX, RZ, RY per qubit
              using one image row (4 pixel values) per qubit, followed by
              a variational sub-circuit of alternating RZZ gates (via
              qml.IsingZZ) and a layer of RY rotations. Qubit count is
              now a free parameter (--qubits) instead of a fixed
              constant -- the circuit and data pipeline both generalize
              to it.
              Measurement: Pauli-Z on all qubits -> softmax over a
              2-class reduction (first half of qubits' expvals summed
              -> class 0, second half -> class 1; this generalizes the
              paper's 4-qubit "[14]" reduction to any qubit count).
  * Loss:     Binary cross-entropy (Eq. 6), not MSE.
  * Metric:   VALIDATION accuracy on the held-out 300-image set at the
              end of training.
  * Pruning target: swept over --sparsities (e.g. 0.10 0.25 0.40)
    instead of a single fixed value, so the sparsity-vs-accuracy curve
    can be reported rather than one point on it.

CHANGES IN v2 (kept, see full history in earlier versions of this file):
  * GPU required for training; interactive dataset prompt (later removed
    in v4 in favor of --datasets); exact print-statement formats for
    per-step / cumulative results.

CHANGES IN v3 (kept):
  * Pruning strategy: gradual cubic sparsity schedule (Zhu & Gupta, 2017)
    + Taylor (|gradient| x |magnitude|) importance (Molchanov et al.,
    2016) instead of one-shot raw-gradient thresholding, with a
    per-round freeze cap, to close the pruned-vs-dense accuracy gap.

CHANGES IN v4 (this version) -- validates the v3 pruning strategy across
multiple axes instead of a single run, and adds noise robustness, as
recent comparable quantum-pruning work does (e.g. ATP: Adaptive
Threshold Pruning, CVPR 2025, evaluates on 4 datasets and under 3%/5%/
10% depolarizing noise):

  1. MULTIPLE DATASETS (--datasets): sweep over any subset of
     DATASET_CLASSES (mnist, fashionmnist by default -- add more binary
     pairs to the dict to extend further).

  2. MULTIPLE QUBIT COUNTS (--qubits): N_QUBITS is no longer a global
     constant. The circuit (ansatz/reduce_to_two), the qnode, and the
     data loader (image rows resized to n_qubits) all generalize to an
     arbitrary qubit count, so scaling behavior can be reported instead
     of a single 4-qubit toy result.

  3. MULTIPLE SPARSITY LEVELS (--sparsities): each (dataset, n_qubits,
     seed) combination now runs ONE no-pruning baseline plus ONE pruned
     run per sparsity target, so a full sparsity-vs-accuracy curve is
     produced instead of a single sparsity point.

  4. MULTIPLE SEEDS (--seeds): the SAME seed drives both data sampling
     and parameter initialization for a given (dataset, n_qubits, seed)
     combination, and is reused across the no-pruning baseline and every
     sparsity level in that combination -- so the pruned-vs-baseline
     accuracy gap is a genuine PAIRED difference per seed, not two
     independently-sampled runs. Aggregate statistics (mean, sample std,
     95% CI via normal approximation) are computed across seeds for both
     raw validation accuracy and the paired gap.

  5. NOISE ROBUSTNESS (--noise-levels): after training, EVERY resulting
     model (no-pruning baseline and each pruned sparsity level) is
     re-evaluated on a subset of the validation set under simulated
     hardware noise using a density-matrix simulator (`default.mixed`)
     with a `qml.DepolarizingChannel` inserted after every gate that is
     actually applied. Frozen (exactly-zero) variational parameters
     SKIP their gate entirely in this noisy evaluation circuit (a
     zero-angle rotation is a no-op mathematically but still injects a
     depolarizing channel on real/simulated noisy hardware) -- so pruned
     circuits genuinely have fewer noisy operations, which is the
     mechanism by which pruning is expected to *improve* noise
     robustness, not just reduce parameter count. This mirrors ATP's use
     of depolarizing noise at multiple intensities to assess robustness
     under realistic conditions, and its multi-dataset design.

  OUTPUTS (all written incrementally so a long sweep surviving a partial
  crash still leaves usable results):
    - Per-step / per-run console output: UNCHANGED format from v2/v3
      (print_no_prune_step / print_prune_step / *_cumulative), so
      existing log-parsing continues to work. Suppress the per-step
      lines (keep cumulative + progress lines) with --quiet for large
      sweeps.
    - {save}_results.csv   : one row per (dataset, n_qubits, seed,
                              run_type, target_sparsity, noise_level) --
                              the long-format raw table for downstream
                              analysis/plotting.
    - {save}_aggregate.csv : the same, grouped and averaged across seeds
                              (mean, std, 95% CI, n_seeds).
    - {save}_gap.csv       : paired accuracy-gap (baseline - pruned)
                              statistics per (dataset, n_qubits, sparsity,
                              noise_level), averaged across seeds.
    - exp3.txt              : human-readable summary combining all of the
                              above (run config + formatted tables).
    - {save}_raw.pkl        : full per-run histories (step_records,
                              cost_hists, final_params) -- only written
                              if --save-raw is passed (can get large).

  A FULL grid (multiple datasets x qubit counts x sparsities x seeds)
  is expensive: each combination re-trains from scratch. Start with
  small --steps / few --seeds to sanity check before scaling up.

Install:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install pennylane pennylane-lightning[gpu] torchvision
"""

import argparse
import csv
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

# GPU is required for training.
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
    # dataset_name -> (torchvision class, class_a, class_b, label names)
    "mnist": (torchvision.datasets.MNIST, 3, 6, ("3", "6")),
    "fashionmnist": (torchvision.datasets.FashionMNIST, 3, 6, ("dress", "shirt")),
    # FashionMNIST label ids: 0 T-shirt/top,1 Trouser,2 Pullover,3 Dress,
    # 4 Coat,5 Sandal,6 Shirt,7 Sneaker,8 Bag,9 Ankle boot
    # Add more binary pairs here to extend the sweep, e.g.:
    # "kmnist": (torchvision.datasets.KMNIST, 0, 1, ("class0", "class1")),
}


# ----------------------------------------------------------------------
# Data: binary subset, images resized to (n_qubits, IMG_COLS)
# ----------------------------------------------------------------------
def load_binary_subset(dataset_name, n_qubits, n_train=500, n_val=300, seed=42, root="./data"):
    ds_cls, class_a, class_b, _ = DATASET_CLASSES[dataset_name]

    tfm = T.Compose([
        T.CenterCrop(24),
        T.Resize((n_qubits, IMG_COLS)),
        T.ToTensor(),  # -> [0, 1], shape (1, n_qubits, IMG_COLS)
    ])
    full = ds_cls(root=root, train=True, download=True, transform=tfm)

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

    def build(idx_list):
        X, y = [], []
        for i in idx_list:
            img, _ = full[i]
            X.append(img.squeeze(0).numpy())  # (n_qubits, IMG_COLS) in [0, 1]
            y.append(0 if i in idx_a else 1)
        return np.stack(X).astype(np.float64), np.array(y, dtype=np.int64)

    X_train, y_train = build(train_idx)
    X_val, y_val = build(val_idx)

    # scale pixel intensities [0,1] -> [0, pi] so they're usable rotation angles
    X_train = X_train * np.pi
    X_val = X_val * np.pi
    return (X_train, y_train), (X_val, y_val)


# ----------------------------------------------------------------------
# Circuit (training path, torch-differentiable): encoding sub-circuit +
# alternating RZZ/RY variational sub-circuit. Generalized to any qubit
# count (derived from params.shape rather than a global constant).
# ----------------------------------------------------------------------
def encode(img, wires):
    """img: (n_qubits, 4) array, one row of 4 pixel values per qubit."""
    for q, w in enumerate(wires):
        row = img[q]
        qml.RY(row[0], wires=w)
        qml.RX(row[1], wires=w)
        qml.RZ(row[2], wires=w)
        qml.RY(row[3], wires=w)


def variational_layer(param, wires):
    """param: (n_qubits, 2). Alternating RZZ (nearest-neighbor chain) + RY."""
    n = len(wires)
    for i in range(n - 1):
        qml.IsingZZ(param[i, 0], wires=[wires[i], wires[i + 1]])
    for i in range(n):
        qml.RY(param[i, 1], wires=wires[i])


def ansatz(params, img):
    """params: (L, n_qubits, 2). Returns n_qubits Pauli-Z expvals.
    Qubit count is derived from params.shape[1] -- no global constant."""
    n_qubits = params.shape[1]
    wires = list(range(n_qubits))
    encode(img, wires)
    for l in range(params.shape[0]):
        variational_layer(params[l], wires)
    return [qml.expval(qml.PauliZ(w)) for w in wires]


def get_qdevice(n_qubits, shots):
    """Returns (device, is_gpu_sim). is_gpu_sim is True only if lightning.gpu
    (the cuQuantum-backed device) actually loaded. default.qubit is a
    CPU-only simulator -- it never runs on CUDA, regardless of what device
    the classical torch tensors fed into it live on."""
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
    """Returns (qnode, is_gpu_sim)."""
    dev, is_gpu_sim = get_qdevice(n_qubits, shots)
    qnode = qml.QNode(ansatz, dev, interface="torch", diff_method=diff_method_for(dev, shots))
    return qnode, is_gpu_sim


# ----------------------------------------------------------------------
# n-qubit expval output -> 2-class probs -> BCE (Eq. 6)
# ----------------------------------------------------------------------
def reduce_to_two(out_vals):
    """out_vals: length-n_qubits sequence of PauliZ expvals in [-1, 1].
    Generalizes the paper's 4-qubit reduction (sum first half -> class 0,
    sum second half -> class 1) to any qubit count."""
    n = len(out_vals)
    half = n // 2
    c0 = sum(out_vals[:half]) if half > 0 else out_vals[0] * 0
    c1 = sum(out_vals[half:])
    return torch.stack([c0, c1])


def batch_forward(qckt, params, X_t):
    """Returns (N, 2) softmax class probabilities."""
    logits = torch.stack([reduce_to_two(qckt(params, x)) for x in X_t])
    return torch.softmax(logits, dim=1)


def bce_loss(probs, y_t, eps=1e-7):
    """Eq. 6: sum_i [ y_i log(yhat_i) + (1-y_i) log(1-yhat_i) ], using the
    class-1 probability as yhat, negated + averaged to be a minimizable loss."""
    yhat = probs[:, 1].clamp(eps, 1 - eps)
    per_sample = -(y_t * torch.log(yhat) + (1 - y_t) * torch.log(1 - yhat))
    return per_sample.mean()


def accuracy_from_probs(probs, y_t):
    preds = (probs[:, 1] > 0.5).long()
    return (preds == y_t.long()).float().mean().item()


# ----------------------------------------------------------------------
# Gradient / saliency helpers (Algorithm 1)
# ----------------------------------------------------------------------
def compute_sample_grad(qckt, params, X_batch):
    grads = []
    for x in X_batch:
        p = params.detach().clone().requires_grad_(True)
        probs = batch_forward(qckt, p, [x])
        loss = bce_loss(probs, torch.zeros(1, dtype=DTYPE, device=p.device))
        (g,) = torch.autograd.grad(loss, p)
        grads.append(g.detach().cpu().numpy())
    return np.mean(grads, axis=0)


def get_sparsity(arr):
    return 1.0 - np.count_nonzero(arr) / arr.size


def enforce_frozen(params, frozen_mask, param_shape):
    """Zero out every currently-frozen parameter. Called every step so that
    earlier pruning rounds can never be silently undone by later optimizer
    updates."""
    if not frozen_mask.any():
        return
    with torch.no_grad():
        flat = params.flatten()
        flat[frozen_mask] = 0
        params.data = flat.reshape(param_shape)


def top_up_to_target_sparsity(params, frozen_mask, grad_buffer, target_sparsity, param_shape):
    """If pruning rounds didn't reach target_sparsity by the end of training,
    freeze the remaining lowest-magnitude-gradient params until they do.
    Safety net -- with the gradual schedule below this should rarely fire."""
    n_params = frozen_mask.size
    n_target = int(np.ceil(target_sparsity * n_params))
    n_frozen = int(frozen_mask.sum())
    if n_frozen >= n_target:
        return frozen_mask
    remaining = np.where(~frozen_mask)[0]
    order = remaining[np.argsort(np.abs(grad_buffer.flatten()[remaining]))]
    need = n_target - n_frozen
    to_freeze = order[:need]
    frozen_mask = frozen_mask.copy()
    frozen_mask[to_freeze] = True
    enforce_frozen(params, frozen_mask, param_shape)
    return frozen_mask


# ----------------------------------------------------------------------
# v3: gradual cubic schedule + Taylor (magnitude x gradient) importance
# ----------------------------------------------------------------------
def taylor_importance(params_np, grad_buffer):
    """First-order (Taylor) importance estimate per parameter:
    importance_i = |grad_i| * |theta_i| (Molchanov et al., 2016)."""
    flat_params = np.abs(params_np.flatten())
    flat_grad = np.abs(grad_buffer.flatten())
    return flat_grad * flat_params


def cubic_sparsity_at_step(t, start_step, end_step, final_sparsity, initial_sparsity=0.0):
    """Gradual/cubic sparsity schedule (Zhu & Gupta, 2017)."""
    if t <= start_step:
        return initial_sparsity
    if t >= end_step:
        return final_sparsity
    progress = (t - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3


def select_prune_round(params, frozen_mask, grad_buffer, n_target, param_shape,
                        max_round_freeze):
    """Freeze up to `max_round_freeze` additional parameters this round,
    chosen as the lowest-Taylor-importance parameters among the
    currently-unfrozen ones, stopping once frozen_mask reaches n_target."""
    n_frozen = int(frozen_mask.sum())
    need = max(0, min(n_target - n_frozen, max_round_freeze))
    if need == 0:
        return frozen_mask
    importance = taylor_importance(params.detach().cpu().numpy(), grad_buffer)
    remaining = np.where(~frozen_mask)[0]
    order = remaining[np.argsort(importance[remaining])]  # ascending: least important first
    to_freeze = order[:need]
    frozen_mask = frozen_mask.copy()
    frozen_mask[to_freeze] = True
    enforce_frozen(params, frozen_mask, param_shape)
    return frozen_mask


# ----------------------------------------------------------------------
# v4: noisy (density-matrix) evaluation path -- NOT used for training,
# only for post-hoc noise-robustness evaluation of trained models. Frozen
# (exactly-zero) gates are SKIPPED entirely here, so pruned models
# genuinely execute fewer noisy operations.
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
    """Evaluate accuracy of a trained (numpy) parameter set under each
    positive noise level in `noise_levels`, on a random subset of the
    validation set, using a density-matrix simulator with a
    DepolarizingChannel after every gate that is actually applied.
    Returns {noise_prob: accuracy}. noise_level == 0.0 is skipped here
    (the caller already has the fast noiseless val_acc from training)."""
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
# Reporting helpers -- per-step / cumulative print formats UNCHANGED
# ----------------------------------------------------------------------
def format_duration(seconds):
    return str(timedelta(seconds=round(seconds, 2)))


def print_no_prune_step(step, acc, loss, t):
    print(f"step {step}: accuracy={acc:.4f}, loss={loss:.6f}, time={t:.4f}s")


def print_prune_step(step, acc, sparsity, loss, t):
    print(f"step {step}: accuracy={acc:.4f}, sparsity={sparsity:.4f}, "
          f"loss={loss:.6f}, time={t:.4f}s")


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
                        win_sz=5, steps=20, tol=0.01, lr=0.1,
                        target_sparsity=0.25, grad_pts=1, compute_device=None,
                        prune_start_frac=0.15, prune_end_frac=0.65,
                        max_round_freeze_frac=0.08, verbose=True):
    compute_device = compute_device if compute_device is not None else TORCH_DEVICE
    X_t = torch.tensor(X_train, dtype=DTYPE, device=compute_device)
    y_t = torch.tensor(y_train, dtype=DTYPE, device=compute_device)
    Xv_t = torch.tensor(X_val, dtype=DTYPE, device=compute_device)
    yv_t = torch.tensor(y_val, dtype=DTYPE, device=compute_device)

    params = torch.tensor(iparams, dtype=DTYPE, device=compute_device, requires_grad=True)
    param_shape = params.shape
    n_params = int(np.prod(param_shape))

    opt = torch.optim.RMSprop([params], lr=lr, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)

    start_step = max(win_sz, int(round(prune_start_frac * steps)))
    end_step = max(start_step + win_sz, int(round(prune_end_frac * steps)))
    max_round_freeze = max(1, int(np.ceil(max_round_freeze_frac * n_params)))

    tau = np.ones(n_params) / n_params
    grad_buffer = compute_sample_grad(qckt, params, X_t[:grad_pts]).flatten()
    frozen_mask = np.zeros(n_params, dtype=bool)

    cost_hists, train_acc_hists = [], []
    step_records = []  # (step, acc, sparsity, loss, step_time)

    if verbose:
        print("\ntraining with pruning:")
    loop_start = time.time()
    for t in range(steps):
        step_start = time.time()

        cur_grad = compute_sample_grad(qckt, params, X_t[:grad_pts]).flatten()
        tau_prime = tau * (1 - np.abs(cur_grad))

        opt.zero_grad()
        probs = batch_forward(qckt, params, X_t)
        loss = bce_loss(probs, y_t)
        loss.backward()
        opt.step()

        enforce_frozen(params, frozen_mask, param_shape)

        next_grad = compute_sample_grad(qckt, params, X_t[:grad_pts]).flatten()
        grad_buffer = grad_buffer + np.abs(cur_grad - next_grad)

        new_cost = loss.item()
        cost_hists.append(new_cost)
        with torch.no_grad():
            train_acc = accuracy_from_probs(batch_forward(qckt, params, X_t), y_t)
        train_acc_hists.append(train_acc)

        if t != 0 and t % win_sz == 0:
            sparsity_target_now = cubic_sparsity_at_step(
                t, start_step, end_step, target_sparsity
            )
            n_target = int(np.ceil(sparsity_target_now * n_params))
            frozen_mask = select_prune_round(
                params, frozen_mask, grad_buffer, n_target, param_shape, max_round_freeze
            )
            grad_buffer = next_grad.copy()

        tau = tau_prime

        step_sparsity = get_sparsity(params.detach().cpu().numpy())
        step_time = time.time() - step_start
        step_records.append((t + 1, train_acc, step_sparsity, new_cost, step_time))
        if verbose:
            print_prune_step(t + 1, train_acc, step_sparsity, new_cost, step_time)

        if new_cost < tol:
            break

    frozen_mask = top_up_to_target_sparsity(
        params, frozen_mask, grad_buffer, target_sparsity, param_shape
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
    step_records = []  # (step, acc, loss, step_time)

    if verbose:
        print("\ntraining without pruning:")
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
# v4: sweep driver -- datasets x qubit counts x seeds x sparsities,
# plus noise-robustness evaluation of every resulting model.
# ----------------------------------------------------------------------
RECORD_FIELDS = [
    "dataset", "n_qubits", "seed", "run_type", "target_sparsity",
    "achieved_sparsity", "n_params", "n_params_active",
    "train_final_acc", "train_final_loss",
    "noise_level", "val_acc", "total_time_s",
]


def _build_rows(dataset, n_qubits, seed, run_type, target_sparsity, achieved_sparsity,
                 n_params, train_final_acc, train_final_loss, val_acc_noiseless,
                 noisy_results, total_time):
    n_active = int(round((1 - achieved_sparsity) * n_params))
    rows = [{
        "dataset": dataset, "n_qubits": n_qubits, "seed": seed, "run_type": run_type,
        "target_sparsity": target_sparsity, "achieved_sparsity": achieved_sparsity,
        "n_params": n_params, "n_params_active": n_active,
        "train_final_acc": train_final_acc, "train_final_loss": train_final_loss,
        "noise_level": 0.0, "val_acc": val_acc_noiseless, "total_time_s": total_time,
    }]
    for noise_level, acc in sorted(noisy_results.items()):
        rows.append({
            "dataset": dataset, "n_qubits": n_qubits, "seed": seed, "run_type": run_type,
            "target_sparsity": target_sparsity, "achieved_sparsity": achieved_sparsity,
            "n_params": n_params, "n_params_active": n_active,
            "train_final_acc": train_final_acc, "train_final_loss": train_final_loss,
            "noise_level": noise_level, "val_acc": acc, "total_time_s": total_time,
        })
    return rows


def run_single(dataset_name, n_qubits, seed, n_layers, steps, win_sz, sparsities,
               noise_levels, prune_start_frac, prune_end_frac, max_round_freeze_frac,
               shots, noise_eval_samples, quiet):
    """Runs one (dataset, n_qubits, seed) combination: one no-pruning
    baseline + one pruned run per sparsity in `sparsities`, each followed
    by noise-robustness evaluation. Returns (records, raw_bundle)."""
    (X_train, y_train), (X_val, y_val) = load_binary_subset(dataset_name, n_qubits, seed=seed)

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
        dataset_name, n_qubits, seed, "no_pruning", None, 0.0, n_params,
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
        )
        noise_p = evaluate_noise_robustness(
            prune_result["final_params"], X_val, y_val, n_qubits, noise_levels,
            max_eval_samples=noise_eval_samples, eval_seed=seed,
        )
        records.extend(_build_rows(
            dataset_name, n_qubits, seed, "pruned", ts, prune_result["sparsity"], n_params,
            prune_result["step_records"][-1][1], prune_result["step_records"][-1][3],
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
    """Mean/std/95% CI of val_acc across seeds, grouped by
    (dataset, n_qubits, run_type, target_sparsity, noise_level)."""
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


def compute_gap_table(records):
    """Paired (same seed) accuracy gap: no_pruning val_acc - pruned val_acc,
    at matching noise_level, averaged across seeds."""
    lookup = defaultdict(dict)
    for r in records:
        key = (r["dataset"], r["n_qubits"], r["seed"], r["noise_level"])
        if r["run_type"] == "no_pruning":
            lookup[key]["baseline"] = r["val_acc"]
        else:
            lookup[key].setdefault("pruned", {})[r["target_sparsity"]] = r["val_acc"]

    gaps = defaultdict(list)
    for (dataset, n_qubits, seed, noise_level), d in lookup.items():
        if "baseline" not in d or "pruned" not in d:
            continue
        for sparsity, acc in d["pruned"].items():
            gaps[(dataset, n_qubits, sparsity, noise_level)].append(d["baseline"] - acc)

    gap_rows = []
    for (dataset, n_qubits, sparsity, noise_level), vals in gaps.items():
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        gap_rows.append({
            "dataset": dataset, "n_qubits": n_qubits, "target_sparsity": sparsity,
            "noise_level": noise_level, "n_seeds": n,
            "accuracy_gap_mean": mean, "accuracy_gap_std": std, "accuracy_gap_ci95": ci95,
        })
    gap_rows.sort(key=lambda r: (r["dataset"], r["n_qubits"], r["target_sparsity"], r["noise_level"]))
    return gap_rows


def write_report(path, config, records, agg, gaps, total_elapsed):
    with open(path, "w") as f:
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Torch device: {TORCH_DEVICE}\n")
        f.write("Run config:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal sweep run time: {format_duration(total_elapsed)} ({total_elapsed:.2f}s)\n")
        f.write(f"Total records: {len(records)}\n\n")

        f.write("=" * 78 + "\n")
        f.write("VALIDATION ACCURACY (mean +/- 95%% CI across seeds)\n".replace("%%", "%"))
        f.write("=" * 78 + "\n")
        f.write(f"{'dataset':<14}{'q':<4}{'run_type':<12}{'sparsity':<10}"
                f"{'noise':<8}{'n_seeds':<9}{'acc_mean':<10}{'acc_std':<10}{'ci95':<8}\n")
        for r in agg:
            sparsity_str = "-" if r["target_sparsity"] is None else f"{r['target_sparsity']:.2f}"
            f.write(f"{r['dataset']:<14}{r['n_qubits']:<4}{r['run_type']:<12}{sparsity_str:<10}"
                    f"{r['noise_level']:<8.3f}{r['n_seeds']:<9}{r['val_acc_mean']:<10.4f}"
                    f"{r['val_acc_std']:<10.4f}{r['val_acc_ci95']:<8.4f}\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("PAIRED ACCURACY GAP (no_pruning - pruned, mean +/- 95%% CI, per seed)\n".replace("%%", "%"))
        f.write("=" * 78 + "\n")
        f.write(f"{'dataset':<14}{'q':<4}{'sparsity':<10}{'noise':<8}{'n_seeds':<9}"
                f"{'gap_mean':<10}{'gap_std':<10}{'ci95':<8}\n")
        for r in gaps:
            f.write(f"{r['dataset']:<14}{r['n_qubits']:<4}{r['target_sparsity']:<10.2f}"
                    f"{r['noise_level']:<8.3f}{r['n_seeds']:<9}{r['accuracy_gap_mean']:<10.4f}"
                    f"{r['accuracy_gap_std']:<10.4f}{r['accuracy_gap_ci95']:<8.4f}\n")

        f.write("\nNote: noise_level=0.0 rows use the noiseless (fast) simulator; "
                "noise_level>0.0 rows use a density-matrix simulator with a "
                "DepolarizingChannel inserted after every gate that is actually "
                "applied (frozen/zeroed gates are skipped).\n")
        f.write("See {results,aggregate,gap}.csv for the full machine-readable tables.\n")


def run_experiment_grid(datasets, qubit_counts, sparsities, seeds, noise_levels,
                         n_layers=3, steps=20, win_sz=5, shots=None,
                         prune_start_frac=0.15, prune_end_frac=0.65,
                         max_round_freeze_frac=0.08, noise_eval_samples=100,
                         quiet=False, save_prefix="qadaprune", save_raw=False):
    combos = [(d, q, s) for d in datasets for q in qubit_counts for s in seeds]
    total = len(combos)
    all_records = []
    raw_bundles = []

    for i, (dataset_name, n_qubits, seed) in enumerate(combos, 1):
        print(f"\n{'#' * 78}\n[{i}/{total}] dataset={dataset_name} qubits={n_qubits} seed={seed}\n{'#' * 78}")
        records, raw_bundle = run_single(
            dataset_name, n_qubits, seed, n_layers, steps, win_sz, sparsities,
            noise_levels, prune_start_frac, prune_end_frac, max_round_freeze_frac,
            shots, noise_eval_samples, quiet,
        )
        all_records.extend(records)
        raw_bundles.append(raw_bundle)

        # incremental write so a crash mid-sweep still leaves usable results
        write_csv(f"{save_prefix}_results.csv", RECORD_FIELDS, all_records)
        if save_raw:
            with open(f"{save_prefix}_raw.pkl", "wb") as f:
                pickle.dump(raw_bundles, f)

    return all_records, raw_bundles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QAdaPrune sweep: datasets x qubit counts x sparsities x seeds, "
                     "with noise-robustness evaluation."
    )
    parser.add_argument("-s", "--save", type=str, default="qadaprune_run")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--win-sz", type=int, default=5)
    parser.add_argument("--shots", type=int, default=None,
                         help="None = noiseless (analytic) TRAINING. Noise robustness is "
                              "controlled separately via --noise-levels.")
    parser.add_argument("--datasets", type=str, nargs="+",
                         default=["mnist", "fashionmnist"], choices=list(DATASET_CLASSES.keys()))
    parser.add_argument("--qubits", type=int, nargs="+", default=[4],
                         help="One or more qubit counts to sweep, e.g. --qubits 4 6 8")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.10, 0.25, 0.40],
                         help="Target sparsity levels to sweep, e.g. --sparsities 0.1 0.25 0.4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123],
                         help="Seeds to sweep (drives both data sampling and init params).")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.03, 0.05, 0.10],
                         help="Depolarizing-noise probabilities for the robustness eval "
                              "(a noiseless, noise_level=0.0 row is always included "
                              "automatically from the fast training-time evaluation).")
    parser.add_argument("--noise-eval-samples", type=int, default=100,
                         help="Number of validation samples used per noise-level evaluation "
                              "(density-matrix simulation is expensive; keep this modest).")
    parser.add_argument("--prune-start-frac", type=float, default=0.15)
    parser.add_argument("--prune-end-frac", type=float, default=0.65)
    parser.add_argument("--max-round-freeze-frac", type=float, default=0.08)
    parser.add_argument("--quiet", action="store_true",
                         help="Suppress per-step console lines (cumulative lines still print).")
    parser.add_argument("--save-raw", action="store_true",
                         help="Also pickle full per-run histories to {save}_raw.pkl.")
    args = parser.parse_args()

    config = {
        "datasets": args.datasets, "qubits": args.qubits, "sparsities": args.sparsities,
        "seeds": args.seeds, "noise_levels": args.noise_levels, "layers": args.layers,
        "steps": args.steps, "win_sz": args.win_sz, "shots": args.shots,
        "prune_start_frac": args.prune_start_frac, "prune_end_frac": args.prune_end_frac,
        "max_round_freeze_frac": args.max_round_freeze_frac,
        "noise_eval_samples": args.noise_eval_samples,
    }
    print("Run config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
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
        save_prefix=args.save, save_raw=args.save_raw,
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
               "accuracy_gap_mean", "accuracy_gap_std", "accuracy_gap_ci95"], gaps)
    write_report("exp3.txt", config, all_records, agg, gaps, total_elapsed)

    print("\n" + "=" * 78)
    print(f"Sweep complete | total run time: {format_duration(total_elapsed)}")
    print(f"Wrote: {args.save}_results.csv, {args.save}_aggregate.csv, "
          f"{args.save}_gap.csv, exp3.txt"
          + (f", {args.save}_raw.pkl" if args.save_raw else ""))
    print("=" * 78)