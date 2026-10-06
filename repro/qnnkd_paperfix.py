"""Root-cause runs for the QNNKD MNIST 1-vs-5 mismatch (repo qnnkd.py still unmodified).
Differences from qnnkd_mnist15.py, each switchable:
  MODE  binary : PCA -> per-component min-max to [0,1] -> >0.5   (what qnnkd_mnist15.py did)
        rawpca : pixels/255 -> PCA (fit on train) -> >0.5 on the raw PCA scores (no rescaling)
  INIT  default: MQLayer default init, Normal(0, 0.01)
        paper  : U[-0.1pi, 0.1pi] as stated in the paper (Sec. III), set from outside
Logs full train/test accuracy and agreement with the one-bit rule "predict 5 iff PC1 bit == 0".
Usage: qnnkd_paperfix.py SEED MODE INIT KD(0|1) LAYERS...   (KD=1 also trains the 10-layer teacher)
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import qnnkd_mnist15 as base                       # data reader, constants, lr decay, fit()
import mindspore as ms
from mindspore import Tensor
import qnnkd

OUT = os.path.join(os.path.dirname(__file__), "out_fix")


def load(mode):
    if mode == "binary":
        return base.load("binary")
    # Made in the root venv (this venv has no sklearn): sklearn 1.7.2 PCA(15) fit on train X/255,
    # bits = transform(X/255) > 0.5. sklearn's sign convention matters because 0.5 is not symmetric.
    d = np.load(os.path.join(os.path.dirname(__file__), "rawpca_sklearn.npz"))
    return d["Xtr"], d["ytr"], d["Xte"], d["yte"]


def init_paper(model, rng):
    w = model.qlayer.weight
    w.set_data(Tensor(rng.uniform(-0.1 * np.pi, 0.1 * np.pi, w.shape).astype(np.float32)))


def predict(model, X, bs=512):
    return np.concatenate([np.argmax(model(Tensor(X[i:i + bs], ms.float32)).asnumpy(), 1)
                           for i in range(0, len(X), bs)])


def report(model, Xtr, ytr, Xte, yte):
    ptr, pte = predict(model, Xtr), predict(model, Xte)
    rule_tr, rule_te = (Xtr[:, 0] == 0).astype(int), (Xte[:, 0] == 0).astype(int)
    return {"train": float((ptr == ytr).mean()), "test": float((pte == yte).mean()),
            "train_n": int((ptr == ytr).sum()), "test_n": int((pte == yte).sum()),
            "agree_pc1_train": float((ptr == rule_tr).mean()), "agree_pc1_test": float((pte == rule_te).mean())}


ITERS, DECAY = int(os.environ.get("ITERS", base.ITERS)), int(os.environ.get("DECAY", 100))


def fit(model, teacher, X, y, seed):
    """base.fit with the step budget and decay period overridable (paper: 350 steps, x0.5 every 100)."""
    np.random.seed(seed); ms.set_seed(seed)
    tr = qnnkd.QNNTrainer(model, teacher=teacher, lr=base.LR, temperature=base.TEMP, alpha=base.ALPHA)
    lr = tr._step.optimizer.learning_rate
    for it in range(0, ITERS, 50):
        if it and it % DECAY == 0:
            lr.set_data(Tensor(float(lr.asnumpy()) * 0.5, lr.dtype))
        tr.train(X, y, teacher_x=X if teacher else None, n_iters=min(50, ITERS - it),
                 batch_size=base.BS, log_every=10**9)
    return tr


def main():
    seed, mode, init, kd = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
    layers = [int(a) for a in sys.argv[5:]]
    Xtr, ytr, Xte, yte = load(mode)
    rule = lambda X, y: int(((X[:, 0] == 0).astype(int) == y).sum())
    print(f"data {Xtr.shape} {Xte.shape}; PC1-bit rule train {rule(Xtr, ytr)}/{len(ytr)} "
          f"test {rule(Xte, yte)}/{len(yte)}; paper flat 11055 / 1865", flush=True)
    rng = np.random.default_rng(seed)
    tag = f"{mode}_{init}_kd{int(kd)}_s{seed}" + (f"_it{ITERS}d{DECAY}" if (ITERS, DECAY) != (350, 100) else "")
    out = {"seed": seed, "mode": mode, "init": init, "students": {}}
    t0 = time.time()
    teacher = None
    ckpt = os.environ.get("TEACHER_CKPT")         # pre-trained teacher from qnnkd_teacher.py
    if kd:
        np.random.seed(seed); ms.set_seed(seed)
        teacher = qnnkd.QNNClassifier(base.NQ, base.T_LAYERS, 2)
        if ckpt:
            qnnkd.load_model(teacher, ckpt); tag += "_" + os.path.basename(ckpt)[:-5]
        else:
            if init == "paper": init_paper(teacher, rng)
            fit(teacher, None, Xtr, ytr, seed)
        out["teacher"] = report(teacher, Xtr, ytr, Xte, yte)
        print("teacher", out["teacher"], f"{time.time()-t0:.0f}s", flush=True)
    for L in layers:
        r = {}
        arms = (True,) if kd and os.environ.get("KD_ONLY") else (False, True) if kd else (False,)
        for use_kd in arms:
            np.random.seed(seed + 1000 + L); ms.set_seed(seed + 1000 + L)
            st = qnnkd.QNNClassifier(base.NQ, L, 2)
            if init == "paper": init_paper(st, np.random.default_rng(seed + 1000 + L))
            fit(st, teacher if use_kd else None, Xtr, ytr, seed)
            r["kd" if use_kd else "plain"] = report(st, Xtr, ytr, Xte, yte)
        out["students"][L] = r
        print("student L", L, r, f"{time.time()-t0:.0f}s", flush=True)
        os.makedirs(OUT, exist_ok=True)
        json.dump(out, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
