# Deep Learning Assignment 01 — Fashion-MNIST MLP

This repo contains a Colab notebook (`DL_ASS01_23F_0602_0559.ipynb`) that builds and evaluates
multi-layer perceptrons on Fashion-MNIST, covering a from-scratch NumPy MLP, activation/optimizer
comparisons, loss-function comparisons, overfitting analysis, regularization techniques, and a
final hyperparameter search whose best configuration is retrained and evaluated on the test set.

## Final Result

| Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|
| **89.64%** | 0.8961 | 0.8964 | 0.8957 |

Produced by the best of 12 randomly-searched configs (selected by 5-fold CV), retrained on the
full 60,000-image training set. See **Reproducing the Final Result** below for exact settings.

## Requirements

- Python 3.9+
- A CUDA-capable GPU is recommended (the notebook was run on a Colab T4) but not required —
  it falls back to CPU automatically.

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

## Repo / Data Layout

No manual data download is needed. The notebook downloads Fashion-MNIST automatically via
`torchvision.datasets.FashionMNIST(root='./data', download=True, ...)` on first run, into a
`./data` folder created alongside the notebook.

## How to Run

1. Clone the repo and open `DL_ASS01_23F_0602_0559.ipynb` in Jupyter, VS Code, or Google Colab.
2. Run all cells **in order, top to bottom** — later cells reuse variables (`X_train`, `y_train`,
   `X_val`, `y_val`, `X_test`, `y_test`, `FlexibleMLP`, `device`, etc.) defined in earlier cells.
3. A fixed seed (`SEED = 42`, applied via a `set_seed()` helper to `random`, `numpy`, and `torch`/
   `torch.cuda`) is set at the top of the notebook and re-applied before each model is built, so
   re-running end-to-end reproduces the same numbers shown below (minor floating-point/CUDA
   nondeterminism aside).

### Notebook sections (run in this order)

| Cell | What it does |
|---|---|
| 1 | Loads Fashion-MNIST, flattens/normalizes images, does a stratified 80/20 train/val split (48,000 / 12,000) plus the 10,000-image test set. |
| 2 | From-scratch NumPy 2-layer MLP (forward/backward/step) — sanity-checked against PyTorch autograd gradients. |
| 3 | Compares activation functions (sigmoid, tanh, ReLU, leaky ReLU) on gradient magnitude and dead-unit rate. |
| 4 | Compares Cross-Entropy vs. MSE loss for classification. |
| 5 | Compares optimizers (SGD, SGD+Momentum, RMSProp, Adam) with per-optimizer tuned learning rates. |
| 6 | Trains an intentionally large/over-parameterized MLP on a 2,000-sample subset to induce overfitting. |
| 7 | Compares regularization techniques (L2, L1, dropout, batch norm, early stopping, data augmentation, more training data) against the overfitting baseline. |
| 8 | Random search over 12 hyperparameter configs, ranked by 5-fold CV accuracy; retrains the best config on the full training set and reports final test metrics + confusion matrix. |

## Reproducing the Final Result

The final numbers come entirely from the **last cell** of the notebook. To reproduce them:

1. Run every prior cell first (they define `X_train`, `y_train`, `X_full`, `y_full`, `X_test`,
   `y_test`, `device`, and `set_seed`).
2. Run the final cell as-is. It will:
   - Generate 12 candidate configs from the search space
     `lr ∈ {0.0005, 0.001, 0.005, 0.01}`, `width ∈ {128, 256, 512}`, `dropout ∈ {0.2, 0.3, 0.5}`
     using `set_seed()` for reproducible sampling.
   - Evaluate each config with 5-fold cross-validation (10 epochs/fold) on the training data.
   - Select the top config — **width=128, dropout=0.2, lr=0.001, Adam optimizer**, architecture
     `784 → 128 (BatchNorm, ReLU, Dropout) → 128 (BatchNorm, ReLU, Dropout) → 10`.
   - Retrain that config on the **full 60,000-image training set** for **25 epochs**.
   - Run a single evaluation pass on the **10,000-image held-out test set** and print accuracy,
     macro precision/recall/F1, plus a confusion matrix.

No manual hyperparameter entry is required — the notebook selects and reruns the best config
automatically. Expect a runtime of several minutes on GPU (CV step is the most expensive part:
12 configs × 5 folds × 10 epochs).

## Key Finding

Among all overfitting-mitigation techniques tested in Part 6 (L2, L1, dropout, batch norm, early
stopping, data augmentation, and more training data), **increasing the training set size from
2,000 to 20,000 samples** gave the largest improvement in both validation accuracy (88.99% vs.
82.07% baseline) and generalization gap (0.077 vs. 0.136 baseline) — outperforming every explicit
regularizer tried.

## Notes on Reproducibility

- All random number generators (Python `random`, NumPy, PyTorch CPU/CUDA) are seeded via a shared
  `set_seed(SEED=42)` call, re-invoked before each model instantiation.
- Exact metric values may vary slightly (±0.1–0.3%) across different GPU/CUDA versions due to
  non-deterministic CUDA kernels; results are deterministic on CPU.
