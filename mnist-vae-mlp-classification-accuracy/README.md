# LDP Evaluation in VAE Latent Space (MNIST)

Benchmarks LDP mechanisms on MNIST classification accuracy: apply LDP to a VAE latent vector, decode, and measure how often the classifier still predicts the correct class.

---

## Overview

The experiment proceeds in three stages:

1. **VAE training** (`train_vae.py`) — a standard VAE is trained on the **public dataset** (MNIST training set, 80/20 train/validation split). The encoder checkpoint is saved for use in Stage 2. The VAE is treated as a **public model**.
2. **Classifier training** (`train_classifier.py`) — an MLP (`D → 10 → ReLU → 32 → ReLU → 10`) is trained jointly end-to-end with the VAE encoder using cross-entropy loss. After joint training, latent vectors (Z_tr, Z_val, Z_te) are extracted from the jointly trained encoder and saved so that the LDP evaluation uses consistent representations. The Jacobian row space (rank ≤ 10, bounded by the first hidden width) identifies the *task-relevant subspace* exploited by ANR mechanisms. Both the encoder and classifier are treated as **public models**.
3. **LDP evaluation** (`eval_mlp_classification.py`) — for each test sample, apply an LDP mechanism to the latent vector, decode, and classify. Accuracy is averaged over the full test set (`n_repeat = len(test_set)`).

---

## Protocol

**Per sample:**
- Latent vector `z_i ∈ ℝ^D` from the jointly trained VAE encoder (posterior mean).
- Apply LDP: `z̃_i = mech.encode(z_i, ε, rng=rng_i)`, then decode: `ẑ_i = mech.decode(z̃_i)`.
- Correct if `clf(ẑ_i) == y_i` (ground-truth label).

**Sample accuracy** = fraction of test samples correctly classified after LDP perturbation.

---

## File Structure

```
.
├── train_vae.py                    # Stage 1: train VAE, save encoder checkpoint
├── train_classifier.py             # Stage 2: joint encoder+MLP training, extracts latent vectors
├── eval_mlp_classification.py      # Stage 3: LDP eval
├── aggregate_seeds.py              # Aggregate per-seed result files
├── run_seeds.sh                    # Run eval over multiple seeds
├── ../mechanisms/mnist_classification_mechanisms.py  # LDP mechanism implementations
├── requirements.txt                # Python dependencies
├── checkpoints/
│   ├── encoder_d{D}.pt             # Jointly trained VAE encoder weights
│   └── latent_mlp_d{D}.pt          # Jointly trained MLP classifier weights
└── data/MNIST/latent/{D}/
    ├── Z_tr.pt, Z_val.pt, Z_te.pt  # Latent vectors (float32)
    └── y_tr.pt, y_val.pt, y_te.pt  # Labels
```

---

## Models

### VAE (`train_vae.py`)

| Component | Architecture |
|-----------|-------------|
| Encoder | `784 → 512 → ReLU → 512 → ReLU → (μ, log σ²)` each `→ D` |
| Decoder | `D → 512 → ReLU → 512 → ReLU → 784 → Sigmoid` |

- Loss: ELBO = BCE reconstruction + KL divergence
- Latent representation: posterior mean `μ` (no reparameterization at inference)
- **Public dataset**: MNIST training set, split 80 % train / 20 % validation (seed 42)
- Early stopping: patience=10 epochs on validation loss; best weights restored
- Validation loss logged every 5 epochs; max epochs=100
- Optimizer: Adam, lr=1e-3, batch=512
- `D ∈ {16, 32, 64}`

### Classifier (`train_classifier.py`)

| Component | Architecture |
|-----------|-------------|
| MLP | `D → 10 → ReLU → 32 → ReLU → 10` (logits) |

- Initializes encoder from `encoder_d{D}.pt`, trains encoder + MLP end-to-end with cross-entropy loss
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4, cosine LR schedule, 100 epochs
- After training, extracts Z_tr, Z_val, Z_te from the jointly trained encoder and saves to `data/MNIST/latent/{D}/`
- Saves updated `encoder_d{D}.pt` and `latent_mlp_d{D}.pt`

The first hidden width (10) bounds the Jacobian row space to rank ≤ 10, which ANR mechanisms exploit to identify the task-sensitive subspace.

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

## Key Hyperparameters

| Constant | Value | Location | Role |
|----------|-------|----------|------|
| `PERCENTILE` | 90.0 | `mnist_classification_mechanisms.py` | Clipping radius `ρ` percentile over `Z_tr` |
| `LAMBDA_N` | 1000.0 | `mnist_classification_mechanisms.py` | Null-space noise amplifier (finite λ required for `PrivUnit2(Opt)+PA`) |
| `DELTA` | 1e-5 | `mnist_classification_mechanisms.py` | δ for all (ε,δ)-LDP mechanisms |
| `MLP_HIDDEN` | 10 | `train_classifier.py` | Classifier hidden width → Jacobian rank ≤ 10 |
| `VAE_EPOCHS` | 100 | `train_vae.py` | Max VAE training epochs (early stopping with patience=10) |
| `VAL_RATIO` | 0.2 | `train_vae.py` | Validation fraction of training set |

---

## Evaluation Grid

- `ε ∈ {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10}` — feature privacy budget
- `D ∈ {16, 32, 64}` — VAE latent dimension
- `n_repeat = len(test_set)` (10,000) — one evaluation per test sample, averaged

---

## How to Run

### Prerequisites

```bash
# Python 3.10

# PyTorch with CUDA 11.8
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# CPU-only
pip install torch==2.7.1+cpu torchvision==0.22.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Other dependencies
pip install -r requirements.txt
```

### Step-by-step

```bash
# 1. Train VAE (saves encoder checkpoint)
python train_vae.py --dim 16
python train_vae.py --dim 32
python train_vae.py --dim 64

# 2. Train encoder and classifier jointly (extracts latent vectors automatically)
python train_classifier.py --dim 16
python train_classifier.py --dim 32
python train_classifier.py --dim 64

# 3. Run LDP evaluation
python eval_mlp_classification.py --dim 16
python eval_mlp_classification.py --dim 32
python eval_mlp_classification.py --dim 64

# Run with multiple seeds
bash run_seeds.sh --dim 16
bash run_seeds.sh --dim 32
bash run_seeds.sh --dim 64

# Aggregate seed results
python aggregate_seeds.py results_seeds/
```

Evaluate a subset of mechanisms:

```bash
python eval_mlp_classification.py --dim 16 --mechs "NoNoise" "PrivUnit2(Opt)+PA" "PrivUnit2(Opt)"
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
