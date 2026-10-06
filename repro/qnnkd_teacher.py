"""Pre-train the 15-qubit 10-layer QNNKD teacher on raw-PCA bits for longer than the students' 350 steps.
The paper gives a teacher at 95.6% test / 95.1% train but no teacher schedule; with the student schedule
(350 steps, lr .01 x0.5 / 100) the teacher stays on the 92% one-bit PC1 plateau. Here: ITERS steps,
lr .01 halved every DECAY steps, test accuracy logged every 100 steps, checkpoint saved at the end.
Usage: qnnkd_teacher.py SEED ITERS DECAY
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import qnnkd_paperfix as fix
import qnnkd_mnist15 as base
import mindspore as ms
from mindspore import Tensor
import qnnkd


def main():
    seed, iters, decay = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    Xtr, ytr, Xte, yte = fix.load("rawpca")
    np.random.seed(seed); ms.set_seed(seed)
    t = qnnkd.QNNClassifier(base.NQ, base.T_LAYERS, 2)
    fix.init_paper(t, np.random.default_rng(seed))
    tr = qnnkd.QNNTrainer(t, lr=base.LR)
    lr = tr._step.optimizer.learning_rate
    hist, t0 = [], time.time()
    for it in range(0, iters, 100):
        if it and it % decay == 0:
            lr.set_data(Tensor(float(lr.asnumpy()) * 0.5, lr.dtype))
        tr.train(Xtr, ytr, n_iters=100, batch_size=base.BS, log_every=10**9)
        r = fix.report(t, Xtr[:3000], ytr[:3000], Xte, yte)
        r.update(iter=it + 100, lr=float(lr.asnumpy()), sec=round(time.time() - t0))
        hist.append(r); print("teacher", r, flush=True)
    tag = f"teacher_s{seed}_{iters}_{decay}"
    qnnkd.save_model(t, os.path.join(fix.OUT, tag + ".ckpt"))
    json.dump(hist, open(os.path.join(fix.OUT, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
