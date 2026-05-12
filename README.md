# ldp-evaluation

Benchmark suite for Local Differential Privacy (LDP) mechanisms across downstream machine learning tasks. Evaluates pure ε-LDP and (ε,δ)-LDP mechanisms with and without the **PA** (Pre/Post-processing Adaptive) transform — an anisotropic noise shaping technique that rotates the latent space so that task-sensitive directions receive proportionally less noise.

---

## Author

Youngmok Ha  
Imperial College London
y.ha25@imperial.ac.uk

---

## Repository Structure

```
.
├── mechanisms/
│   └── mechanisms.py                          # All LDP mechanism implementations (shared)
├── cifar10-resnet20-classification-accuracy/  # Task 1: CIFAR-10 classification
├── mnist-vae-mlp-classification-accuracy/     # Task 2: MNIST classification
└── user-level-linear-regression-accuracy/     # Task 3: User-level linear regression
```

---

## Evaluation Tasks

### Task 1 — CIFAR-10 Classification (ResNet-20 Features)

**Directory:** `cifar10-resnet20-classification-accuracy/`

Applies LDP to 64-dim penultimate-layer features of a ResNet-20 trained on CIFAR-10, then classifies the decoded vector. Accuracy is averaged over all 10,000 test samples.

- Classifier types: linear FC head or MLP (relu / gelu / sigmoid / tanh / leaky_relu)
- Evaluation grid: ε ∈ {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10}
- Supports ablation study and distribution-shift evaluation (CIFAR-10-C)
- See [`cifar10-resnet20-classification-accuracy/README.md`](cifar10-resnet20-classification-accuracy/README.md)

### Task 2 — MNIST Classification (VAE Latent Space)

**Directory:** `mnist-vae-mlp-classification-accuracy/`

Applies LDP to VAE latent vectors (D ∈ {16, 32, 64}), decodes, and classifies with a jointly trained MLP. Accuracy is averaged over all 10,000 test samples.

- Evaluation grid: ε ∈ {0.5, …, 10}, D ∈ {16, 32, 64}
- Broader mechanism set including Duchi, Harmony, and Piecewise mechanisms
- See [`mnist-vae-mlp-classification-accuracy/README.md`](mnist-vae-mlp-classification-accuracy/README.md)

### Task 3 — User-Level Linear Regression (London Smart Meters)

**Directory:** `user-level-linear-regression-accuracy/`

User-level ε-LDP evaluation on household energy data: each private household participates in at most one aggregation across all rounds, giving a strict ε-per-household guarantee. Utility measured as RMSE against actual next-day energy consumption.

- Window size W=16 days, single-output regression (energy mean by default)
- Evaluation grid: ε ∈ {0.5, 1.0, …, 10.0} (step 0.5)
- See [`user-level-linear-regression-accuracy/README.md`](user-level-linear-regression-accuracy/README.md)

---

## LDP Mechanisms (`mechanisms/mechanisms.py`)

All mechanisms satisfy ε-LDP (pure) or (ε, δ)-LDP with δ = 1e-5. Clipping radii ρ are set to the 90th percentile of the corresponding norm over the public training set.

| Name | Type | Description |
|------|------|-------------|
| `NoNoise` | — | No perturbation; upper bound |
| `Laplace(L1)` | ε-LDP | L1 clip + Laplace noise |
| `AGM` | (ε,δ)-LDP | L2 clip + Gaussian (Balle & Wang 2018) |
| `PrivUnit2(Opt)` | ε-LDP | Spherical step-function on S^{d-1} |
| `PrivUnitG(MC)` | ε-LDP | Gaussian ambient-space step-function (MC) |
| `PrivUnitG(Paper)` | ε-LDP | PrivUnitG with paper-exact parameters |
| `CW(Laplace)+PA` | ε-LDP | PA + coordinate-wise i.n.i.d. Laplace |
| `CW(AGM)+PA` | (ε,δ)-LDP | PA + coordinate-wise i.n.i.d. Gaussian |
| `PLAN(Pub)` | (ε,δ)-LDP | Variance-aware scaling + Gaussian |
| `PLAN(Paper)` | (ε,δ)-LDP | PLAN Algorithm 1 (Aumüller et al. 2024) |
| `Inst-Opt` | (ε,δ)-LDP | Hadamard rotation + median shift + optimal L2 clip |
| `Task-Aware` | ε-LDP | Cholesky whitening + water-filling (linear downstream models only) |
| `*+PA` | same | Any mechanism above with PA anisotropic pre/post-processing |

### PA Transform

**Row-space extraction** (`compute_jacobian_row_space`):
- Stack per-sample Jacobians `J_i ∈ ℝ^{K×D}` into `B ∈ ℝ^{(n·K)×D}`.
- SVD with gap-based rank threshold → `W_eff ∈ ℝ^{r×D}` (effective rank `r`).

**SV-weighted λ allocation** (Lagrange optimum for `min Σ s_i² λ_i` s.t. `Σ 1/√λ_i = C`):
```
1/√λ_i ∝ s_i^{2/3}   (row space, i = 1…r)
1/√λ_i = 1/√λ_N      (null space, i = r+1…d,  λ_N = 1000)
```

**ANR** (Anisotropic Noise Reshaping) rotates the representation space by the Jacobian row space of the downstream task model so that task-sensitive directions receive proportionally less noise.

```
Pre-process  (Encode):  \bar{z}     =  clip( (z − μ) @ U ⊙ (1/√λ) )
         [add noise in \bar{Z}-space]
Post-process (Decode):  z_dec = (\bar{z}_noisy ⊙ √λ) @ U.T + μ
```

---

## Quick Start

Each task is self-contained. Install dependencies and follow the step-by-step instructions in each subdirectory's README.

```bash
# Prerequisites: Python 3.10, PyTorch (CUDA or CPU)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Task 1: CIFAR-10
cd cifar10-resnet20-classification-accuracy
pip install -r requirements.txt
python train_resnet20.py
python eval_cifar_classification.py

# Task 2: MNIST
cd mnist-vae-mlp-classification-accuracy
pip install -r requirements.txt
python train_vae.py --dim 64
python train_classifier.py --dim 64
python eval_mlp_classification.py --dim 64

# Task 3: London Smart Meters
cd user-level-linear-regression-accuracy
pip install -r requirements.txt
python download_data.py
python prepare_data.py --window 16 --target energy_mean
python train_regressor.py --window 16 --target energy_mean
python eval_regression.py --window 16 --target energy_mean
```

---

## References
- **PA**: Ha, Y., Schlegel, V., Sun, Y., & Bharath, A. A. (2026). Jacobian-Guided Anisotropic Noise Reshaping for Utility Enhancement Under Local Differential Privacy. *arXiv*.
- **CW**: Muthukrishnan, G., & Kalyani, S. (2025). Differential Privacy With Higher Utility by Exploiting Coordinate-Wise Disparity. *IEEE TIFS*.
- **PLAN**: Aumüller, M., Lebeda, C. J., Nelson, B., & Pagh, R. (2024). PLAN: Variance-Aware Private Mean Estimation. *PETs*.
- **PrivUnitG**: Asi, H., Feldman, V., & Talwar, K. (2022). Optimal Algorithms for Mean Estimation under Local Differential Privacy. *ICML*.
- **Inst-Opt**: Huang, Z., Liang, Y., & Yi, K. (2021). Instance-optimal Mean Estimation Under Differential Privacy. *NeurIPS*.
- **AGM**: Balle, B., & Wang, Y.-X. (2018). Improving the Gaussian Mechanism for Differential Privacy. *ICML*.
- **PrivUnit2**: Bhowmick, A. et al. (2018). Protection Against Reconstruction. *arXiv:1812.00984*.
- **Task-Aware**: Cheng, X. et al. (2022). Locally Differentially Private Functional Statistics. *ICML*.
- **Duchi**: Duchi, J., Jordan, M. I., & Wainwright, M. J. (2013). Local Privacy and Statistical Minimax Rates. *FOCS*.
