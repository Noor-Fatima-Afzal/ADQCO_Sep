"""Paper-setup run of repro/ref/QNNKDMindSpore (qnnkd.py, unmodified) on MNIST 1-vs-5.
Paper (arXiv 2603.21586): 15 qubits, teacher 10 layers, students 4..8 layers, T=2, alpha=0.8,
Adam lr .01 (x0.5 every 100 steps), 350 iters, batch 64, 12163 train / 2027 test, PCA + 0.5
binarisation + phase (RX) encoding. Run with the QNNKD venv python (py3.10).
Usage: qnnkd_mnist15.py SEED MODE LAYERS [LAYERS...]   MODE = binary | cont
"""
import gzip, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ref", "QNNKDMindSpore"))
os.environ.setdefault("QNNKD_DEVICE", "CPU")
import mindspore as ms
from mindspore import Tensor
import qnnkd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MNIST", "raw")
NQ, T_LAYERS, TEMP, ALPHA, ITERS, BS, LR = 15, 10, 2.0, 0.8, 350, 64, 0.01


def idx(path):
    with gzip.open(path, "rb") as f:
        b = f.read()
    if "images" in path:
        return np.frombuffer(b, np.uint8, offset=16).reshape(-1, 784).astype(np.float64) / 255.0
    return np.frombuffer(b, np.uint8, offset=8)


def load(mode):
    Xtr, ytr = idx(f"{ROOT}/train-images-idx3-ubyte.gz"), idx(f"{ROOT}/train-labels-idx1-ubyte.gz")
    Xte, yte = idx(f"{ROOT}/t10k-images-idx3-ubyte.gz"), idx(f"{ROOT}/t10k-labels-idx1-ubyte.gz")
    def sub(X, y):
        m = (y == 1) | (y == 5); return X[m], (y[m] == 5).astype(np.int32)
    (Xtr, ytr), (Xte, yte) = sub(Xtr, ytr), sub(Xte, yte)
    mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    P = Vt[:NQ].T                                   # PCA fitted on train only
    Ztr, Zte = (Xtr - mu) @ P, (Xte - mu) @ P
    lo, hi = Ztr.min(0), Ztr.max(0)
    Ztr, Zte = [np.clip((Z - lo) / (hi - lo), 0, 1) for Z in (Ztr, Zte)]
    if mode == "binary":                             # paper text: threshold 0.5, RX(0|1)
        Ztr, Zte = (Ztr > .5).astype(np.float64), (Zte > .5).astype(np.float64)
    else:                                            # sensitivity variant: RX(pi * z)
        Ztr, Zte = Ztr * np.pi, Zte * np.pi
    return Ztr, ytr, Zte, yte


def decay_lr(tr, it):
    if it and it % 100 == 0:
        p = tr._step.optimizer.learning_rate
        p.set_data(Tensor(float(p.asnumpy()) * 0.5, p.dtype))


def fit(model, teacher, X, y, seed):
    np.random.seed(seed); ms.set_seed(seed)
    tr = qnnkd.QNNTrainer(model, teacher=teacher, lr=LR, temperature=TEMP, alpha=ALPHA)
    for c in range(ITERS // 50):
        decay_lr(tr, c * 50)
        tr.train(X, y, teacher_x=X if teacher else None, n_iters=50, batch_size=BS, log_every=10**9)
    return tr


def main():
    seed, mode, student_layers = int(sys.argv[1]), sys.argv[2], [int(a) for a in sys.argv[3:]]
    Xtr, ytr, Xte, yte = load(mode)
    print(f"data {Xtr.shape} {Xte.shape} class balance {ytr.mean():.3f}", flush=True)
    np.random.seed(seed); ms.set_seed(seed)
    out = {"seed": seed, "mode": mode, "students": {}}
    t0 = time.time()
    teacher = qnnkd.QNNClassifier(NQ, T_LAYERS, 2)
    ttr = fit(teacher, None, Xtr, ytr, seed)
    out["teacher_test"] = ttr.evaluate(Xte, yte); out["teacher_train"] = ttr.evaluate(Xtr[:2000], ytr[:2000])
    print("teacher", out["teacher_test"], f"{time.time()-t0:.0f}s", flush=True)
    for L in student_layers:
        r = {}
        for kd in (False, True):
            np.random.seed(seed + 1000 + L); ms.set_seed(seed + 1000 + L)
            st = qnnkd.QNNClassifier(NQ, L, 2)
            tr = fit(st, teacher if kd else None, Xtr, ytr, seed)
            r["kd" if kd else "plain"] = tr.evaluate(Xte, yte)
        out["students"][L] = r; print("student L", L, r, f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "out_kd"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "out_kd", f"{mode}_s{seed}.json"), "w"))


if __name__ == "__main__":
    main()
