# LDP Evaluation in ResNet-20 Feature Space (CIFAR-10)

Benchmarks 22 Local Differential Privacy (LDP) mechanisms on the task of privately aggregating ResNet-20 penultimate-layer feature vectors, evaluated by how well a server can reconstruct the aggregate after noise. Supports two evaluation settings (label-free and label-anchored) and two classifier types (linear FC and nonlinear MLP).

---

## Overview

The experiment proceeds in two stages:

1. **Feature extraction** (`train_resnet20.py`) — ResNet-20 is trained on CIFAR-10 (80/20 train/validation split). The 64-dim penultimate-layer features (Z_tr, Z_val, Z_te), the FC classifier head, and optional MLP classifier heads are saved. All are treated as **public models**.
2. **LDP evaluation** (`eval_cifar_classification.py`) — `K_max = len(test_set) // n_repeat` clients are pre-assigned (each holds one sample per repeat with no privacy composition); `K` clients are randomly sampled per repeat.

Two classifier types are supported via `--mlp` / `--activation`:
- **Linear FC** (default): ℝ^64 → ℝ^10, Jacobian row space rank ≤ 10 (constant across inputs).
- **MLP**: 64 → 10 → Act → Dropout(0.3) → 32 → Act → Dropout(0.3) → 10, where Act ∈ {relu, sigmoid, tanh, leaky_relu}. The first hidden width of 10 bounds the effective Jacobian rank; because the network is nonlinear, the Jacobian is aggregated over 500 training samples.

Two evaluation settings are supported via `--setting`:
- **label-anchored** (default): each client also privatizes its label via Randomized Response; the server groups reports by noisy label and evaluates per-group prediction preservation.
- **label-free**: clients transmit only the privatized feature vector; the server aggregates all reports into one global mean and evaluates global prediction preservation.

---

## Protocol

**Client pool:** `K_max = len(test_set) // n_repeat` clients are pre-assigned from 10,000 test samples. Each evaluation round samples `K ≤ K_max` clients without replacement. With the default `n_repeat=20`, `K_max = 500`; values of `K > K_max` in `K_list` are skipped.

**Client side (per client `i`):**
- Feature vector `z_i ∈ ℝ^{64}` from the ResNet-20 penultimate layer.

### Label-Free Setting (`--setting label_free`)

- Client sends `z̃_i = mech.encode(z_i, ε_feat, rng=rng_i)` only — no label transmitted.
- **Server:** `z̄_enc = mean(z̃_i)` over all `K` clients, then `z̄_dec = mech.decode(z̄_enc)`.
- **Prediction preservation rate** = fraction of rounds where `clf(z̄_dec) == clf(z̄)`, averaged over `n_repeat` rounds.
- Privacy cost per item: **ε_feat**.

### Label-Anchored Setting (`--setting label_anchored`, default)

- Each client is assigned one seed that drives both perturbations sequentially:
  1. **Label perturbation** `ỹ_i = RR(y_i, ε_label)` — Randomized Response. Consumes exactly 2 random values (1 float for the flip test, 1 int for the random offset) regardless of flip outcome, so the encoding RNG state is deterministic.
  2. **Feature perturbation** `z̃_i = mech.encode(z_i, ε_feat, rng=rng_i)` — satisfies ε_feat-LDP, using the remaining RNG state after RR.
- **Server:** group encoded vectors by noisy label `ỹ`. For each group `k`: `z̄_enc = mean(z̃_i | ỹ_i = k)`, then `z̄_dec = mech.decode(z̄_enc)`.
- **Prediction preservation rate** = fraction of groups `k` where `clf(z̄_dec) == clf(z̄)`, averaged over `n_repeat` rounds. Monitoring per-class granularity reveals *class collapse* — where noisy aggregates lose class boundaries.
- Privacy cost per item: **ε_feat + ε_label** (basic composition).

**Randomized Response** keep probability for 10 classes:
```
p(keep) = e^{ε_label} / (e^{ε_label} + 9)
```

In both settings, `z̄` is the unperturbed aggregate (oracle reference); the metric checks prediction agreement with the noiseless aggregate, not ground truth.

---

## File Structure

```
.
├── train_resnet20.py               # Stage 1: train ResNet-20, extract 64-dim features, train MLP heads
├── eval_cifar_classification.py    # Stage 2: LDP evaluation (label-free and label-anchored)
├── model.py                        # ResNet-20, FeatureClassifier, MLPClassifier definitions
├── ../mechanisms/cifar10_classification_mechanisms.py  # All LDP mechanism implementations
├── requirements.txt                # Python dependencies
├── checkpoints/
│   ├── resnet20_cifar10.pt         # Trained ResNet-20 weights
│   ├── feature_clf.pt              # Linear FC classifier head (64 → 10)
│   ├── mlp_clf_relu.pt             # MLP classifier, ReLU activation
│   ├── mlp_clf_sigmoid.pt          # MLP classifier, Sigmoid activation
│   ├── mlp_clf_tanh.pt             # MLP classifier, Tanh activation
│   └── mlp_clf_leaky_relu.pt       # MLP classifier, LeakyReLU activation
└── data/CIFAR10/latent/
    ├── Z_tr.pt, Z_val.pt, Z_te.pt  # 64-dim feature vectors (float32)
    └── y_tr.pt, y_val.pt, y_te.pt  # Labels
```

---

## Models

### ResNet-20 (`train_resnet20.py`)

| Component | Architecture |
|-----------|-------------|
| Backbone | 3 stages × 3 BasicBlocks, channels 16 → 32 → 64 |
| Feature | GlobalAvgPool → 64-dim vector |
| Classifier | Linear 64 → 10 (logits) |

- **Public dataset**: CIFAR-10 training set, split 80 % train / 20 % validation (seed 42)
- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4, MultiStepLR (×0.1 at epoch 100, 150), 200 epochs
- Best validation accuracy checkpoint restored after training
- After training, 64-dim features extracted and saved to `data/CIFAR10/latent/`

The default classifier is a linear map ℝ^64 → ℝ^10, so its Jacobian is the weight matrix W ∈ ℝ^{10×64} with rank ≤ 10, which ANR mechanisms exploit to identify the task-sensitive subspace.

### MLP Classifier (`--mlp --activation <act>`)

| Component | Architecture |
|-----------|-------------|
| Layer 1 | Linear 64 → 10 |
| Layer 2 | Linear 10 → 32 |
| Layer 3 | Linear 32 → 10 (logits) |
| Activation | relu / sigmoid / tanh / leaky_relu (inserted after layers 1 and 2) |
| Dropout | 0.3 after each activation |

The first hidden width of 10 constrains the effective Jacobian rank to ≤ 10 — the same task-sensitive bound as the linear classifier. Because the network is nonlinear, `compute_jacobian_row_space` aggregates Jacobians over 500 training samples to build the row-space estimate.

> `TASK(Cheng22)` is automatically removed from the mechanism set when `--mlp` is used (it is designed for linear MSE settings only).

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
| `Duchi` | ε-LDP | clip to `[0, ρ]` → scale to `[0,1]` → Duchi mechanism (Duchi et al. 2013) |
| `Harmony` | ε-LDP | Per-dim min-max normalization to `[0,1]` → Harmony (Nguyen et al. 2016) |
| `Piecewise` | ε-LDP | clip to `[0, ρ]` → scale to `[0,1]` → Piecewise mechanism (Wang et al. 2019) |
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
| `TASK(Cheng22)` | ε-LDP | Cholesky whitening + water-filling noise allocation for linear classifiers (Cheng et al., ICML 2022) |

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
| `PERCENTILE` | 90.0 | `cifar10_classification_mechanisms.py` | Clipping radius `ρ` percentile over `Z_tr` |
| `LAMBDA_N` | 1000.0 | `cifar10_classification_mechanisms.py` | Null-space noise amplifier (finite λ required for `ANR-SV+PrivUnit2`) |
| `DELTA` | 1e-5 | `cifar10_classification_mechanisms.py` | δ for all (ε,δ)-LDP mechanisms |
| `VAL_RATIO` | 0.2 | `train_resnet20.py` | Validation fraction of training set |
| `EPOCHS` | 200 | `train_resnet20.py` | ResNet-20 training epochs |

---

## Evaluation Grid

- `ε_feat ∈ {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10}` — feature privacy budget
- `ε_label ∈ {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10}` — label privacy budget (label-anchored only)
- `K ∈ {1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000}` — total clients; values above `K_max = len(test_set) // n_repeat` are skipped
- 20 random repeats per combination (default `--n_repeat 20`)

**Label-free** sweeps `ε_feat × K` (no `ε_label` loop).  
**Label-anchored** sweeps `ε_feat × ε_label × K`.

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
# 1. Train ResNet-20 and extract 64-dim features
python train_resnet20.py

# Or skip training with pretrained weights
python train_resnet20.py --weights path/to/resnet20.pth

# Also train MLP heads (optional, one per activation)
python train_resnet20.py --mlp --activation relu
python train_resnet20.py --mlp --activation sigmoid
python train_resnet20.py --mlp --activation tanh
python train_resnet20.py --mlp --activation leaky_relu

# 2. Run LDP evaluation

# Label-anchored (default)
python eval_cifar_classification.py --n_repeat 20

# Label-free
python eval_cifar_classification.py --setting label_free

# Both settings in one run
python eval_cifar_classification.py --setting both

# MLP classifier (nonlinear Jacobian)
python eval_cifar_classification.py --mlp --activation relu
python eval_cifar_classification.py --mlp --activation relu --setting label_free

# Save results
python eval_cifar_classification.py --setting both > results.txt 2>&1
```

Evaluate a subset of mechanisms:

```bash
python eval_cifar_classification.py --mechs "NoNoise" "ANR-SV+PrivUnit2" "PrivUnit2"
python eval_cifar_classification.py --setting label_free --mechs "NoNoise" "ANR-SV(L1,Lap)" "PrivUnit2"
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
