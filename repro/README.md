# Reproduction checks: conclusions

## QAdaPrune (arXiv 2408.13352)

- The intern's version is not faithful to the paper. It forces 25% sparsity with an extra "top-up" step, uses
  a pruning window of 5 instead of 2, and uses a different gradient signal and threshold rule.
- At the paper's settings it removes about 91% of the parameters, and accuracy drops to 46.6% (93.2% without
  pruning; 6 seeds, MNIST-2).
- Fixing the gradient or the threshold rule, or following the authors' released code, still removes 57–95% of
  the parameters, at 40–60% accuracy. Freezing instead of deleting keeps 93.3%, but removes nothing.
- The paper's claim (≥25% removed with no accuracy loss) cannot be checked: the authors' code has no MNIST
  experiment and does not run as shipped.

Scripts: `compare.py`, `ablate.py`.

## GATE (arXiv 2603.19805)

- The authors' Corral script runs unmodified, apart from one stray argument removed (`run_gate_corral.py`).
  `gate_corral_extra.py` adds unpruned and classical baselines.

## Distilling QNNs (arXiv 2603.21586), MNIST 1 vs 5

| | Paper | Mine (authors' code) |
|---|---|---|
| Students without teacher, 4–7 layers | 92.0% | 92.0% (1864 vs 1865 of 2027 test images) |
| Student without teacher, 8 layers | 95.1% | 92.0% with the stated 350 steps; 93.6% with 1200 steps |
| Teacher, 10 layers | 95.6% | 92.0% with 350 steps; 93.7% with 2000 steps |
| Teacher's gain for a 6-layer student | +1.3 points | +0.4 points (3 seeds, within seed noise) |

- The paper's flat 92% is a shortcut: the model predicts from a single bit (the first PCA component). This
  only appears with the paper's data recipe: threshold the raw PCA values at 0.5, with no rescaling.
- The stated 350 training steps are too few for any model, the teacher included, to get past that shortcut.
  The paper's teacher and 8-layer results need more training than the paper states.
- The benefit of the teacher is not confirmed. It is within seed-to-seed variation, and the paper's figure is a
  single run.
- To settle it, the authors' data preparation and teacher-training scripts are needed. The paper says they are
  available from the first author on request.

Scripts: `qnnkd_paperfix.py`, `qnnkd_teacher.py`, data `rawpca_sklearn.npz`, results `results_fix/`.
`qnnkd_mnist15.py` is the first attempt, with the wrong data recipe. `qnnkd_seeds.txt` is the repo's toy
benchmark, where the teacher gives no gain.
