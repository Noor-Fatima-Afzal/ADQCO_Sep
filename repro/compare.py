"""Small MNIST-2 comparison: intern's QAdaPrune (exp1.py, unchanged) vs a version with
the fixes I identified. One seed per process; writes repro/out/seed{N}.json.

Both pruned arms use the paper's MNIST window w=2 and share data, init, optimizer
(RMSprop lr=0.1, as in exp1) and 20 steps, so only the pruning logic differs.
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import exp1
from exp1 import (DTYPE, batch_forward, bce_loss, accuracy_from_probs, compute_sample_grad,
                  RMSPROP_ALPHA, RMSPROP_EPS)

torch.set_num_threads(4)
WIN, STEPS, LR, GRAD_PTS = 2, 20, 0.1, 32


def true_label_grad(qckt, params, X, y):
    """Mean per-example gradient of BCE against the TRUE label (exp1 used label 0)."""
    gs = []
    for x, t in zip(X, y):
        p = params.detach().clone().requires_grad_(True)
        loss = bce_loss(batch_forward(qckt, p, [x]), t.reshape(1))
        (g,) = torch.autograd.grad(loss, p)
        gs.append(g.detach().cpu().numpy())
    return np.mean(gs, axis=0).flatten()


def fixed_qadaprune(qckt, init, Xtr, ytr, Xv, yv, seed):
    """QAdaPrune per the paper's text: thresholds tau0=1/n, tau'=tau(1-|grad|); accumulate
    |grad_{t+1}-grad_t| over window w; params below tau are TEMPORARILY frozen (keep their
    value, get no updates; mask recomputed each window). No zeroing, no forced top-up."""
    dev = torch.device("cpu")
    X = torch.tensor(Xtr, dtype=DTYPE, device=dev); y = torch.tensor(ytr, dtype=DTYPE, device=dev)
    Xval = torch.tensor(Xv, dtype=DTYPE, device=dev); yval = torch.tensor(yv, dtype=DTYPE, device=dev)
    params = torch.tensor(init, dtype=DTYPE, device=dev, requires_grad=True)
    n = params.numel()
    opt = torch.optim.RMSprop([params], lr=LR, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)
    rng = np.random.RandomState(seed + 1000)
    tau = np.ones(n) / n
    buf = np.zeros(n)
    frozen = np.zeros(n, dtype=bool)
    frozen_hist = []
    for t in range(STEPS):
        idx = rng.choice(len(X), GRAD_PTS, replace=False)
        g0 = true_label_grad(qckt, params, X[idx], y[idx])
        tau = tau * (1 - np.abs(g0))
        opt.zero_grad()
        loss = bce_loss(batch_forward(qckt, params, X), y)
        loss.backward()
        with torch.no_grad():
            params.grad.view(-1)[torch.from_numpy(frozen)] = 0.0
        opt.step()
        g1 = true_label_grad(qckt, params, X[idx], y[idx])
        buf += np.abs(g0 - g1)
        if t != 0 and (t % WIN == 0):
            frozen = buf < tau          # new mask each window (temporary freeze)
            buf = np.zeros(n)
        frozen_hist.append(float(frozen.mean()))
    with torch.no_grad():
        val = accuracy_from_probs(batch_forward(qckt, params, Xval), yval)
    return {"val_acc": val, "sparsity": float(frozen.mean()), "frozen_hist": frozen_hist}


def main(seed):
    t0 = time.time()
    (Xtr, ytr), (Xv, yv) = exp1.load_binary_subset("mnist", seed=seed)
    np.random.seed(seed)
    init = np.random.uniform(-np.pi, np.pi, size=(3, exp1.N_QUBITS, 2))
    qckt, is_gpu = exp1.make_qnode(None)
    dev = torch.device("cpu")
    dense = exp1.optimize(qckt, init.copy(), Xtr, ytr, Xv, yv, steps=STEPS, compute_device=dev)
    intern = exp1.optimize_and_prune(qckt, init.copy(), Xtr, ytr, Xv, yv, win_sz=WIN, steps=STEPS,
                                     target_sparsity=0.25, compute_device=dev)
    fixed = fixed_qadaprune(qckt, init.copy(), Xtr, ytr, Xv, yv, seed)
    out = {"seed": seed, "dense_val": dense["val_acc"],
           "intern_val": intern["val_acc"], "intern_sparsity": intern["sparsity"],
           "intern_natural_sparsity": float(intern["step_records"][-1][2]),
           "fixed_val": fixed["val_acc"], "fixed_sparsity": fixed["sparsity"],
           "fixed_frozen_hist": fixed["frozen_hist"], "secs": time.time() - t0}
    os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", f"seed{seed}.json"), "w"))
    print("RESULT", json.dumps(out))


if __name__ == "__main__":
    main(int(sys.argv[1]))
