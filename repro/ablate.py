"""One-factor ablation of QAdaPrune on MNIST-2 (w=2, 20 steps, RMSprop lr .1, 3 layers).
Usage: ablate.py ARM SEED.  Arms differ from the intern's exp1 logic in named flags only.
Flags: grad  = label0_1 (exp1: BCE vs label 0, first sample) | true_b32 (true-label BCE, 32 random)
                | out_1 (authors' code: grad of class-1 prob, first sample)
       tau   = paper (tau*(1-|g|)) | softmax (authors' code: tau - tau*softmax(g))
       buf   = sumdiff (paper: sum |g_t - g_t+1|) | absminus (authors' code: |buf - g|)
       keep  = cumul (zero all ever-pruned) | latest (authors' code: only last window's idx re-zeroed)
       topup = force >=25% sparsity after training (exp1)
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import exp1
from exp1 import DTYPE, batch_forward, bce_loss, accuracy_from_probs, RMSPROP_ALPHA, RMSPROP_EPS
torch.set_num_threads(2)
WIN, STEPS, LR = 2, 20, 0.1
ARMS = {
  "authors":        dict(grad="out_1",    tau="softmax", buf="absminus", keep="latest", topup=False),
  "tau_softmax":    dict(grad="label0_1", tau="softmax", buf="sumdiff",  keep="cumul",  topup=False),
  "truegrad_only":  dict(grad="true_b32", tau="paper",   buf="sumdiff",  keep="cumul",  topup=False),
  "fix_zero":       dict(grad="true_b32", tau="paper",   buf="sumdiff",  keep="cumul",  topup=False),  # same as truegrad_only; kept as 'fix + permanent pruning' alias
}
del ARMS["fix_zero"]
ARMS["truegrad_softmax"] = dict(grad="true_b32", tau="softmax", buf="sumdiff", keep="cumul", topup=False)

def grad_of(q, params, X, y, mode, rng):
    if mode == "label0_1":
        idx = [0]; lab = lambda t: torch.zeros(1, dtype=DTYPE)
    elif mode == "true_b32":
        idx = rng.choice(len(X), 32, replace=False); lab = lambda t: y[t].reshape(1)
    else:
        idx = [0]
    gs = []
    for i in idx:
        p = params.detach().clone().requires_grad_(True)
        pr = batch_forward(q, p, [X[i]])
        out = pr[:, 1].sum() if mode == "out_1" else bce_loss(pr, lab(i))
        (g,) = torch.autograd.grad(out, p); gs.append(g.numpy())
    return np.mean(gs, 0).flatten()

def run(arm, seed):
    f = ARMS[arm]
    (Xtr, ytr), (Xv, yv) = exp1.load_binary_subset("mnist", seed=seed)
    np.random.seed(seed); init = np.random.uniform(-np.pi, np.pi, (3, exp1.N_QUBITS, 2))
    q, _ = exp1.make_qnode(None)
    X = torch.tensor(Xtr, dtype=DTYPE); y = torch.tensor(ytr, dtype=DTYPE)
    Xv_ = torch.tensor(Xv, dtype=DTYPE); yv_ = torch.tensor(yv, dtype=DTYPE)
    P = torch.tensor(init, dtype=DTYPE, requires_grad=True); n = P.numel()
    opt = torch.optim.RMSprop([P], lr=LR, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS)
    rng = np.random.RandomState(seed + 1000)
    tau = np.ones(n) / n
    g_prev = grad_of(q, P, X, y, f["grad"], rng)
    buf = np.abs(g_prev) if f["buf"] == "absminus" else g_prev * 0 + np.abs(g_prev) * 0
    if f["buf"] == "sumdiff": buf = np.abs(g_prev) * 0 + g_prev * 0 + np.abs(g_prev)  # exp1 init: buffer starts at |g0|
    mask = np.zeros(n, bool); hist = []
    def zero():
        if mask.any():
            with torch.no_grad():
                flat = P.view(-1); flat[torch.from_numpy(mask)] = 0
    for t in range(STEPS):
        cur = grad_of(q, P, X, y, f["grad"], np.random.RandomState(seed * 100 + t))
        tau_new = tau * (1 - np.abs(cur)) if f["tau"] == "paper" else tau - tau * (np.exp(cur) / np.exp(cur).sum())
        opt.zero_grad(); loss = bce_loss(batch_forward(q, P, X), y); loss.backward(); opt.step()
        zero()
        nxt = grad_of(q, P, X, y, f["grad"], np.random.RandomState(seed * 100 + t))
        buf = buf + np.abs(cur - nxt) if f["buf"] == "sumdiff" else np.abs(buf - nxt)
        if t != 0 and t % WIN == 0:
            idx = buf < tau_new
            mask = (mask | idx) if f["keep"] == "cumul" else idx
            zero(); buf = nxt * 0 if f["buf"] == "absminus" else nxt.copy()
        tau = tau_new
        hist.append(float((P.detach().numpy().flatten() == 0).mean()))
    with torch.no_grad():
        val = accuracy_from_probs(batch_forward(q, P, Xv_), yv_)
    out = dict(arm=arm, seed=seed, val_acc=val, sparsity=hist[-1], sp_hist=hist)
    os.makedirs("repro/out_abl", exist_ok=True)
    json.dump(out, open(f"repro/out_abl/{arm}_{seed}.json", "w")); print("RESULT", arm, seed, val, hist[-1])

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
