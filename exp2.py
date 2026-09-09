import argparse
import pickle
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pennylane as qml
import torch
import torchvision
import torchvision.transforms as T

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
N_QUBITS = 4
IMG_SIZE = 4  # after center-crop(24) + resize down to 4x4

RMSPROP_ALPHA = 0.9
RMSPROP_EPS = 1e-8

# GPU is required for this run.
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
}


# ----------------------------------------------------------------------
# Data: MNIST-2 / FashionMNIST-2, 4x4 downsampled, 500 train / 300 val
# ----------------------------------------------------------------------
def load_binary_subset(dataset_name, n_train=500, n_val=300, seed=42, root="./data"):
    ds_cls, class_a, class_b, _ = DATASET_CLASSES[dataset_name]

    tfm = T.Compose([
        T.CenterCrop(24),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),  # -> [0, 1], shape (1, 4, 4)
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
            X.append(img.squeeze(0).numpy())  # (4, 4) in [0, 1]
            y.append(0 if i in idx_a else 1)
        return np.stack(X).astype(np.float64), np.array(y, dtype=np.int64)

    X_train, y_train = build(train_idx)
    X_val, y_val = build(val_idx)

    # scale pixel intensities [0,1] -> [0, pi] so they're usable rotation angles
    X_train = X_train * np.pi
    X_val = X_val * np.pi
    return (X_train, y_train), (X_val, y_val)


# ----------------------------------------------------------------------
# Circuit: encoding sub-circuit + alternating RZZ/RY variational sub-circuit
# ----------------------------------------------------------------------
def encode(img, wires):
    """img: (4, 4) array, one row of 4 pixel values per qubit (Fig. 2)."""
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
    """params: (L, n_qubits, 2). Returns 4 Pauli-Z expvals."""
    wires = list(range(N_QUBITS))
    encode(img, wires)
    for l in range(params.shape[0]):
        variational_layer(params[l], wires)
    return [qml.expval(qml.PauliZ(w)) for w in wires]


def get_qdevice(shots):
    """Returns (device, is_gpu_sim). is_gpu_sim is True only if lightning.gpu
    (the cuQuantum-backed device) actually loaded. default.qubit is a
    CPU-only simulator -- it never runs on CUDA, regardless of what device
    the classical torch tensors fed into it live on."""
    try:
        dev = qml.device("lightning.gpu", wires=N_QUBITS, shots=shots)
        print(f"using lightning.gpu (cuQuantum), shots={shots}")
        return dev, True
    except Exception as e:
        print(f"lightning.gpu unavailable ({e}); falling back to default.qubit (CPU simulator).")
        print("For true GPU-accelerated quantum simulation, install the cuQuantum-backed "
              "plugin: `pip install pennylane-lightning-gpu` (requires a visible CUDA "
              "toolkit + cuquantum-sdk). Falling back to CPU for the quantum circuit only "
              "-- classical torch ops still respect --shots/device as configured.")
        dev = qml.device("default.qubit", wires=N_QUBITS, shots=shots)
        return dev, False


def diff_method_for(dev, shots):
    if shots is None and dev.name in ("lightning.gpu", "lightning.qubit"):
        return "adjoint"
    return "parameter-shift" if shots is not None else "best"


def make_qnode(shots):
    """Returns (qnode, is_gpu_sim)."""
    dev, is_gpu_sim = get_qdevice(shots)
    qnode = qml.QNode(ansatz, dev, interface="torch", diff_method=diff_method_for(dev, shots))
    return qnode, is_gpu_sim


# ----------------------------------------------------------------------
# 4-valued expval output -> 2-class probs -> BCE (Eq. 6)
# ----------------------------------------------------------------------
def reduce_to_two(out4):
    """out4: length-4 sequence of PauliZ expvals in [-1, 1]."""
    c0 = out4[0] + out4[1]
    c1 = out4[2] + out4[3]
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


def build_saliency(grad_buffer, thresh_vec):
    gbuf = grad_buffer.flatten()
    return [i for i, g in enumerate(gbuf) if g < thresh_vec[i]]


def get_sparsity(arr):
    return 1.0 - np.count_nonzero(arr) / arr.size


def enforce_frozen(params, frozen_mask, param_shape):
    """Zero out every currently-frozen parameter. Called every step so that
    earlier pruning rounds can never be silently undone by later optimizer
    updates -- this is the fix for the cumulative-freezing bug."""
    if not frozen_mask.any():
        return
    with torch.no_grad():
        flat = params.flatten()
        flat[frozen_mask] = 0
        params.data = flat.reshape(param_shape)


def top_up_to_target_sparsity(params, frozen_mask, grad_buffer, target_sparsity, param_shape):
    """If pruning rounds didn't reach target_sparsity by the end of training,
    freeze the remaining lowest-magnitude-gradient params until they do.
    Because frozen_mask is now kept in sync with the real zeroed params on
    every step (see enforce_frozen), this check is now trustworthy: if
    frozen_mask already meets the target, the actual parameter sparsity
    does too."""
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
    importance_i = |grad_i| * |theta_i| (Molchanov et al., 2016). A
    parameter with small ACCUMULATED gradient sensitivity *and* small
    current magnitude contributes least to the loss and is safest to
    zero out. Pure gradient-magnitude thresholding (the original
    Algorithm 1 saliency) can freeze large-magnitude, currently
    influential weights just because their instantaneous gradient is
    small; multiplying by |theta| guards against that."""
    flat_params = np.abs(params_np.flatten())
    flat_grad = np.abs(grad_buffer.flatten())
    return flat_grad * flat_params


def cubic_sparsity_at_step(t, start_step, end_step, final_sparsity, initial_sparsity=0.0):
    """Gradual/cubic sparsity schedule (Zhu & Gupta, 2017). Ramps sparsity
    smoothly from initial_sparsity at start_step to final_sparsity at
    end_step (fast early, tapering off), instead of committing large
    chunks of parameters to zero in one or two big jumps. end_step is
    intentionally set below the total step count so the run finishes
    pruning with a fine-tuning tail to recover any accuracy lost while
    freezing (the single biggest lever in GMP*'s ablations for closing
    the pruned-vs-dense gap)."""
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
    currently-unfrozen ones, stopping once frozen_mask reaches n_target.
    Capping the per-round freeze count (rather than freezing everything
    the cubic schedule calls for in one shot) is what prevents the large
    single-step accuracy collapses seen with one-shot threshold-based
    freezing."""
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
# Reporting helpers -- exact output format requested (UNCHANGED from v2)
# ----------------------------------------------------------------------
def format_duration(seconds):
    """Human-readable H:MM:SS(.ss) style duration string."""
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
                        max_round_freeze_frac=0.08):
    compute_device = compute_device if compute_device is not None else TORCH_DEVICE
    X_t = torch.tensor(X_train, dtype=DTYPE, device=compute_device)
    y_t = torch.tensor(y_train, dtype=DTYPE, device=compute_device)
    Xv_t = torch.tensor(X_val, dtype=DTYPE, device=compute_device)
    yv_t = torch.tensor(y_val, dtype=DTYPE, device=compute_device)

    params = torch.tensor(iparams, dtype=DTYPE, device=compute_device, requires_grad=True)
    param_shape = params.shape
    n_params = int(np.prod(param_shape))

    opt = torch.optim.RMSprop([params], lr=lr, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)

    # v3: gradual schedule bounds + per-round freeze cap.
    start_step = max(win_sz, int(round(prune_start_frac * steps)))
    end_step = max(start_step + win_sz, int(round(prune_end_frac * steps)))
    max_round_freeze = max(1, int(np.ceil(max_round_freeze_frac * n_params)))

    tau = np.ones(n_params) / n_params
    grad_buffer = compute_sample_grad(qckt, params, X_t[:grad_pts]).flatten()
    frozen_mask = np.zeros(n_params, dtype=bool)

    cost_hists, train_acc_hists = [], []
    step_records = []  # (step, acc, sparsity, loss, step_time)

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

        # FIX (kept from v2): re-apply the FULL cumulative frozen mask every
        # step, not just the most recently pruned round's indices.
        enforce_frozen(params, frozen_mask, param_shape)

        next_grad = compute_sample_grad(qckt, params, X_t[:grad_pts]).flatten()
        grad_buffer = grad_buffer + np.abs(cur_grad - next_grad)

        new_cost = loss.item()
        cost_hists.append(new_cost)
        with torch.no_grad():
            train_acc = accuracy_from_probs(batch_forward(qckt, params, X_t), y_t)
        train_acc_hists.append(train_acc)

        if t != 0 and t % win_sz == 0:
            # v3: cubic-scheduled target for THIS step, capped incremental
            # freeze chosen by Taylor (magnitude x gradient) importance --
            # replaces the old one-shot raw-gradient-threshold freeze.
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
        print_prune_step(t + 1, train_acc, step_sparsity, new_cost, step_time)

        if new_cost < tol:
            break

    # enforce "at least target_sparsity" like the paper's reported >=25%
    # (safety net -- with the gradual schedule above this should rarely
    # need to freeze anything additional).
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
    }


def optimize(qckt, iparams, X_train, y_train, X_val, y_val, steps=100, tol=0.01, lr=0.1,
             compute_device=None):
    compute_device = compute_device if compute_device is not None else TORCH_DEVICE
    X_t = torch.tensor(X_train, dtype=DTYPE, device=compute_device)
    y_t = torch.tensor(y_train, dtype=DTYPE, device=compute_device)
    Xv_t = torch.tensor(X_val, dtype=DTYPE, device=compute_device)
    yv_t = torch.tensor(y_val, dtype=DTYPE, device=compute_device)

    params = torch.tensor(iparams, dtype=DTYPE, device=compute_device, requires_grad=True)
    opt = torch.optim.RMSprop([params], lr=lr, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)

    cost_hists, train_acc_hists = [], []
    step_records = []  # (step, acc, loss, step_time)

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
    }


# ----------------------------------------------------------------------
# Driver: run ONE dataset, both training modes (no-prune, prune), one
# shots setting (noiseless) -- kept simple to match the requested report.
# ----------------------------------------------------------------------
def run_dataset(dataset_name, n_layers=3, steps=20, win_sz=5, seed=42,
                 target_sparsity=0.25, shots=None,
                 prune_start_frac=0.15, prune_end_frac=0.65,
                 max_round_freeze_frac=0.08):
    (X_train, y_train), (X_val, y_val) = load_binary_subset(dataset_name, seed=seed)

    np.random.seed(seed)
    init_params = np.random.uniform(-np.pi, np.pi, size=(n_layers, N_QUBITS, 2))

    qckt, is_gpu_sim = make_qnode(shots)
    compute_device = TORCH_DEVICE if is_gpu_sim else torch.device("cpu")
    print(f"Quantum circuit tensors will run on: {compute_device} "
          f"({'lightning.gpu' if is_gpu_sim else 'default.qubit fallback'})")

    no_prune_result = optimize(
        qckt, init_params.copy(), X_train, y_train, X_val, y_val, steps=steps,
        compute_device=compute_device,
    )
    prune_result = optimize_and_prune(
        qckt, init_params.copy(), X_train, y_train, X_val, y_val,
        win_sz=win_sz, steps=steps, target_sparsity=target_sparsity,
        compute_device=compute_device,
        prune_start_frac=prune_start_frac, prune_end_frac=prune_end_frac,
        max_round_freeze_frac=max_round_freeze_frac,
    )
    return {"no_pruning": no_prune_result, "qadaprune": prune_result}


def prompt_dataset_choice():
    while True:
        choice = input(
            "\nWhich dataset do you want to run?\n"
            "  1) MNIST-2 (digits 3 vs 6)\n"
            "  2) FashionMNIST-2 (dress vs shirt)\n"
            "Enter 1 or 2: "
        ).strip()
        if choice == "1":
            return "mnist"
        if choice == "2":
            return "fashionmnist"
        print("Please enter 1 or 2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAdaPrune on MNIST-2 / FashionMNIST-2 (GPU)")
    parser.add_argument("-s", "--save", type=str, default="qadaprune_run")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--win-sz", type=int, default=5)
    parser.add_argument("--target-sparsity", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=None,
                         help="None = noiseless (analytic). Set e.g. 10000 for limited-shots.")
    parser.add_argument("--dataset", type=str, default=None, choices=["mnist", "fashionmnist"],
                         help="Skip the interactive prompt and run this dataset directly.")
    parser.add_argument("--prune-start-frac", type=float, default=0.15,
                         help="Fraction of steps to wait before pruning begins (warm-up).")
    parser.add_argument("--prune-end-frac", type=float, default=0.65,
                         help="Fraction of steps by which target sparsity is reached; "
                              "the rest of training is a fine-tuning recovery tail.")
    parser.add_argument("--max-round-freeze-frac", type=float, default=0.08,
                         help="Max fraction of all parameters that may be newly frozen "
                              "in a single pruning round.")
    args = parser.parse_args()

    dataset_name = args.dataset if args.dataset is not None else prompt_dataset_choice()
    print(f"\nRunning dataset: {dataset_name}")

    run_start = time.time()
    result = run_dataset(
        dataset_name, n_layers=args.layers, steps=args.steps, win_sz=args.win_sz,
        seed=args.seed, target_sparsity=args.target_sparsity, shots=args.shots,
        prune_start_frac=args.prune_start_frac, prune_end_frac=args.prune_end_frac,
        max_round_freeze_frac=args.max_round_freeze_frac,
    )
    total_elapsed = time.time() - run_start

    with open(f"{args.save}_{dataset_name}_results.pkl", "wb") as f:
        pickle.dump(result, f)

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name} | total run time: {format_duration(total_elapsed)}")
    print("=" * 70)

    # v3: summary text file is now always named exp2.txt (was
    # f"{args.save}_{dataset_name}_summary.txt" in v2).
    with open("exp2.txt", "w") as f:
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Torch device: {TORCH_DEVICE}\n\n")

        f.write("training without pruning:\n")
        for step, acc, loss, t in result["no_pruning"]["step_records"]:
            f.write(f"step {step}: accuracy={acc:.4f}, loss={loss:.6f}, time={t:.4f}s\n")
        np_final_acc = result["no_pruning"]["step_records"][-1][1]
        np_final_loss = result["no_pruning"]["step_records"][-1][2]
        f.write(f"cumulative result: accuracy={np_final_acc:.4f}, loss={np_final_loss:.6f}, "
                f"time={result['no_pruning']['total_time']:.4f}s\n\n")

        f.write("training with pruning:\n")
        for step, acc, sparsity, loss, t in result["qadaprune"]["step_records"]:
            f.write(f"step {step}: accuracy={acc:.4f}, sparsity={sparsity:.4f}, "
                    f"loss={loss:.6f}, time={t:.4f}s\n")
        p_final_acc = result["qadaprune"]["step_records"][-1][1]
        p_final_loss = result["qadaprune"]["step_records"][-1][3]
        f.write(f"cumulative result: accuracy={p_final_acc:.4f}, "
                f"sparsity={result['qadaprune']['sparsity']:.4f}, loss={p_final_loss:.6f}, "
                f"time={result['qadaprune']['total_time']:.4f}s\n\n")

        f.write(f"Total run time: {format_duration(total_elapsed)} ({total_elapsed:.2f}s)\n")
        f.write(f"No-pruning validation accuracy: {result['no_pruning']['val_acc']:.4f}\n")
        f.write(f"QAdaPrune validation accuracy: {result['qadaprune']['val_acc']:.4f}\n")