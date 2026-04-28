# LDP Evaluation in VAE Latent Space (MNIST)

Benchmarks 21 Local Differential Privacy (LDP) mechanisms on the task of privately aggregating VAE latent vectors, evaluated by how well a server can reconstruct the per-group mean after noise.

---

## Overview

The experiment proceeds in three stages:

1. **VAE training** (`train_vae.py`) — a standard VAE is trained on the **public dataset** (MNIST training set, 80/20 train/validation split). The encoder checkpoint is saved for use in Stage 2. The VAE is treated as a **public model**.
2. **Classifier training** (`train_classifier.py`) — an MLP (`D → 10 → ReLU → 32 → ReLU → 10`) is trained jointly end-to-end with the VAE encoder using cross-entropy loss. After joint training, latent vectors (Z_tr, Z_val, Z_te) are extracted from the jointly trained encoder and saved so that the LDP evaluation uses consistent representations. The Jacobian row space (rank ≤ 10, bounded by the first hidden width) identifies the *task-relevant subspace* exploited by ANR mechanisms. Both the encoder and classifier are treated as **public models**.
3. **LDP evaluation** (`eval_mlp_classification.py`) — `K_max = len(test_set) // n_repeat` clients pre-assigned (each holds one sample per repeat with no privacy composition); `K` clients are randomly sampled per repeat, each reporting their assigned sample. 

---

## Protocol

**Client side (per client `i`):**
- Latent vector `z_i ∈ ℝ^D` from the jointly trained VAE encoder (posterior mean).
- Each client is assigned one seed that drives both perturbations sequentially:
  1. **Label perturbation** `ỹ_i = RR(y_i, ε_label)` — Randomized Response. Consumes exactly 2 random values (1 float for the flip test, 1 int for the random offset) regardless of flip outcome, so the encoding RNG state is deterministic.
  2. **Feature perturbation** `z̃_i = mech.encode(z_i, ε_feat, rng=rng_i)` — satisfies ε_feat-LDP, using the remaining RNG state after RR.

**Server side:**
- Group encoded vectors by noisy label `ỹ`.
- For each group `k`: `z̄_enc = mean(z̃_i | ỹ_i = k)`, then `z̄_dec = mech.decode(z̄_enc)`.
- **Group accuracy** = fraction of groups `k` where `clf(z̄_dec) == clf(z̄)`.
  - `z̄` is the unperturbed group mean (oracle reference).
  - `clf(z̄_dec) == clf(z̄)` checks that the decoded aggregate predicts the same class as the noiseless aggregate, not ground truth.

**Randomized Response** keep probability for `K` classes:
```
p(keep) = e^{ε_label} / (e^{ε_label} + K − 1)
```

---

## File Structure

```
.
├── train_vae.py                    # Stage 1: train VAE, save encoder checkpoint
├── train_classifier.py             # Stage 2: joint encoder+MLP training, extracts latent vectors
├── eval_mlp_classification.py      # Stage 3: LDP eval 
├── ../mechanisms/standard_vae_mechanisms.py  # All LDP mechanism implementations
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

## LDP Mechanisms

All mechanisms satisfy ε-LDP (pure) or (ε, δ)-LDP with `δ = 1e-5`.  
Clipping radii `ρ` are set to the 90th percentile of the corresponding norm over the public training set `Z_tr`.

> **Note on `Laplace(L∞)`**: L∞ clipping constrains each coordinate to `[-ρ, ρ]`, giving a joint L1 sensitivity of `2ρd`. The noise scale is therefore `2ρd/ε` per coordinate to satisfy ε-LDP for the full vector — the same sensitivity accounting used by `ANR-SV(L∞,Lap)`.

| Name | Type | Description |
|------|------|-------------|
| `NoNoise` | — | No perturbation; upper bound on group accuracy |
| `Laplace(L1)` | ε-LDP | L1 clip to `ρ`, Laplace noise scale `2ρ/ε` (L1 sensitivity = `2ρ`) |
| `Laplace(L∞)` | ε-LDP | L∞ clip to `ρ`, Laplace noise `2ρd/ε` per coordinate (L1 sensitivity = `2ρd`) |
| `AGM` | (ε,δ)-LDP | L2 clip to `ρ`, AGM noise `σ = _agm_sigma(ε, δ, 2ρ)` (Balle & Wang 2018) |
| `Duchi` | ε-LDP | L∞ clip → scale to `[-1,1]` → Duchi mechanism (Duchi et al. 2013) |
| `Harmony` | ε-LDP | Per-dim min-max normalization to `[0,1]` → Harmony (Nguyen et al. 2016) |
| `Piecewise` | ε-LDP | L∞ clip → scale to `[-1,1]` → Piecewise mechanism (Wang et al. 2019) |
| `ANR-SV(L1,Lap)` | ε-LDP | ANR-SV transform → L1 clip → Laplace |
| `ANR-SV(L∞,Lap)` | ε-LDP | ANR-SV transform → L∞ clip → Laplace (L1 sensitivity = `2ρd`) |
| `ANR-SV(L2,AGM)` | (ε,δ)-LDP | ANR-SV transform → L2 clip → AGM noise |
| `ANR-SV+Duchi` | ε-LDP | ANR-SV transform → L∞ clip → Duchi in transformed space |
| `ANR-SV+Harmony` | ε-LDP | ANR-SV transform → per-dim min-max normalization to `[0,1]` → Harmony in transformed space |
| `ANR-SV+Piecewise` | ε-LDP | ANR-SV transform → L∞ clip → Piecewise in transformed space |
| `ANR-SV+PrivUnit2` | ε-LDP | ANR-SV transform → normalize to `S^{d-1}` → PrivUnit2 step-function |
| `ANR-SV+PrivUnitG` | ε-LDP | ANR-SV transform → normalize to `S^{d-1}` → PrivUnitG step-function |
| `ANR-SV-CW(Lap)` | ε-LDP | ANR-SV transform → coordinate-wise i.n.i.d. Laplace, budget split `∝ λ_i^{1/3}` [Muthukrishnan & Kalyani, TIFS 2025] |
| `ANR-SV-CW(AGM)` | (ε,δ)-LDP | ANR-SV transform → coordinate-wise i.n.i.d. Gaussian, budget split minimizing `Σ σ_i²` [Muthukrishnan & Kalyani, TIFS 2025] |
| `PLAN` | (ε,δ)-LDP | Variance-aware scaling → L2 clip → classical Gaussian noise [Aumüller et al., PETs 2024] |
| `Shifted-CM` | (ε,δ)-LDP | Hadamard rotation → per-dim median shift → optimal-C L2 clip → AGM noise [Huang et al., NeurIPS 2021] |
| `PrivUnit2` | ε-LDP | Optimal spherical step-function on `S^{d-1}` (Bhowmick et al. 2018) |
| `PrivUnitG` | ε-LDP | Gaussian ambient-space step-function (Asi et al., ICML 2022) |
| ~~`TASK(Cheng22)`~~ | — | Excluded from `eval_mlp_classification.py`: assumes a linear classifier; not applicable to MLP |

### ANR-SV Transform

**ANR** (Anisotropic Noise Randomization) rotates the latent space by the Jacobian row space of the downstream classifier so that task-sensitive directions receive proportionally less noise.

**Row-space extraction** (`compute_jacobian_row_space`):
- Stack per-sample Jacobians `J_i ∈ ℝ^{K×D}` into `B ∈ ℝ^{(n·K)×D}`.
- SVD with gap-based rank threshold → `W_eff ∈ ℝ^{r×D}` (effective rank `r`).

**SV-weighted λ allocation** (Lagrange optimum for `min Σ s_i² λ_i` s.t. `Σ 1/√λ_i = C`):
```
1/√λ_i ∝ s_i^{2/3}   (row space, i = 1…r)
1/√λ_i = 1/√λ_N      (null space, i = r+1…d,  λ_N = 1000)
```

**Encode / Decode**:
```
x     = (z − μ) @ U ⊙ (1/√λ)          # anisotropic scaling into row space
[add noise in x-space]
z_dec = (x_noisy ⊙ √λ) @ U.T + μ      # invert scaling and rotation
```

---

## Key Hyperparameters

| Constant | Value | Location | Role |
|----------|-------|----------|------|
| `PERCENTILE` | 90.0 | `standard_vae_mechanisms.py` | Clipping radius `ρ` percentile over `Z_tr` |
| `LAMBDA_N` | 1000.0 | `standard_vae_mechanisms.py` | Null-space noise amplifier (finite λ required for `ANR-SV+PrivUnit2`) |
| `DELTA` | 1e-5 | `standard_vae_mechanisms.py` | δ for all (ε,δ)-LDP mechanisms |
| `MLP_HIDDEN` | 10 | `train_classifier.py` | Classifier hidden width → Jacobian rank ≤ 10 |
| `VAE_EPOCHS` | 100 | `train_vae.py` | Max VAE training epochs (early stopping with patience=10) |
| `VAL_RATIO` | 0.2 | `train_vae.py` | Validation fraction of training set |

---

## Evaluation Grid

- `ε_feat ∈ {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10}` — feature privacy budget
- `ε_label ∈ {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10}` — label privacy budget
- `K ∈ {1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000}` — total clients; values above `K_max = len(test_set) // n_repeat` are skipped
- 20 random repeats per `(ε_feat, ε_label, K)` combination

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
python eval_mlp_classification.py --dim 16 --n_repeat 20
python eval_mlp_classification.py --dim 32 --n_repeat 20
python eval_mlp_classification.py --dim 64 --n_repeat 20

# Run all and save results
(python eval_mlp_classification.py --dim 16 && \
 python eval_mlp_classification.py --dim 32 && \
 python eval_mlp_classification.py --dim 64) > results.txt 2>&1
```

Evaluate a subset of mechanisms:

```bash
python eval_mlp_classification.py --dim 16 --mechs "NoNoise" "ANR-SV+PrivUnit2" "PrivUnit2"
```

---

## References

- **ANR-CW (coordinate-wise i.n.i.d. noise)**: Muthukrishnan, G., & Kalyani, S. (2025). Differential Privacy With Higher Utility by Exploiting Coordinate-Wise Disparity: Laplace Mechanism Can Beat Gaussian in High Dimensions. *IEEE Transactions on Information Forensics and Security*.

- **PLAN**: Aumüller, M., Lebeda, C. J., Nelson, B., & Pagh, R. (2024). PLAN: Variance-Aware Private Mean Estimation. *Proceedings on Privacy Enhancing Technologies*.

- **PrivUnitG**: Asi, H., Feldman, V., & Talwar, K. (2022). Optimal Algorithms for Mean Estimation under Local Differential Privacy. *ICML 2022*.

- **Shifted-CM**: Huang, Z., Liang, Y., & Yi, K. (2021). Instance-optimal Mean Estimation Under Differential Privacy. *NeurIPS 2021*.

- **Piecewise**: Wang, S., Huang, Z., Nie, T., Hu, Q., Wang, Y., & Skoglund, M. (2019). Local Differential Privacy for Data Collection and Analysis. *arXiv:1906.01777*.

- **AGM**: Balle, B., & Wang, Y.-X. (2018). Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising. *ICML 2018*.

- **PrivUnit2**: Bhowmick, A., Duchi, J., Freudiger, J., Kapoor, G., & Rogers, R. (2018). Protection Against Reconstruction and Its Applications in Private Federated Learning. *arXiv:1812.00984*.

- **Harmony**: Nguyen, T. T., Xiao, X., Yang, Y., Hui, S. C., Shin, H., & Shin, J. (2016). Collecting and Analyzing Data from Smart Device Users with Local Differential Privacy. *arXiv:1606.05053*.

- **Duchi**: Duchi, J., Jordan, M. I., & Wainwright, M. J. (2013). Local Privacy and Statistical Minimax Rates. *FOCS 2013*.
