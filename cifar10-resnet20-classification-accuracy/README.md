# LDP Evaluation in ResNet-20 Feature Space (CIFAR-10)

Benchmarks LDP mechanisms on classification accuracy: for each test sample, apply an LDP mechanism to the 64-dim ResNet-20 penultimate-layer feature vector, decode, and classify. Accuracy is averaged over the full test set.

---

## Overview

The experiment proceeds in two stages:

1. **Feature extraction** (`train_resnet20.py`) — ResNet-20 is trained on CIFAR-10 (80/20 train/validation split). The 64-dim penultimate-layer features (Z_tr, Z_val, Z_te), the FC classifier head, and optional MLP classifier heads are saved.
2. **LDP evaluation** (`eval_cifar_classification.py`) — For each test sample, encode the feature vector with an LDP mechanism, decode, and check whether the classifier prediction matches the true label. Accuracy is averaged over all test samples.

Two classifier types are supported via `--mlp` / `--activation`:
- **Linear FC** (default): ℝ^64 → ℝ^10, Jacobian row space rank ≤ 10 (constant across inputs).
- **MLP**: 64 → 10 → Act → Dropout(0.3) → 32 → Act → Dropout(0.3) → 10, where Act ∈ {relu, sigmoid, tanh, leaky_relu}. Jacobian is aggregated over 500 training samples.

---

## Protocol

For each test sample `i`:

**Client side:**
- Encode: `z̃_i = mech.encode(z_i, ε_feat, rng=rng_i)`

**Server side:**
- Decode: `ẑ_i = mech.decode(z̃_i)`
- Evaluate: `clf(ẑ_i) == y_i`

**Accuracy** = fraction of test samples correctly classified after encode–decode, averaged over the full test set (`n_repeat = len(test_set) = 10,000`).

---

## File Structure

```
.
├── train_resnet20.py               # Stage 1: train ResNet-20, extract 64-dim features, train MLP heads
├── eval_cifar_classification.py    # Stage 2: LDP evaluation (main + ablation)
├── model.py                        # ResNet-20, FeatureClassifier, MLPClassifier definitions
├── ../mechanisms/mechanisms.py     # All LDP mechanism implementations
├── requirements.txt                # Python dependencies
├── checkpoints/
│   ├── resnet20_cifar10.pt         # Trained ResNet-20 weights
│   ├── feature_clf.pt              # Linear FC classifier head (64 → 10)
│   ├── mlp_clf_relu.pt
│   ├── mlp_clf_sigmoid.pt
│   ├── mlp_clf_tanh.pt
│   └── mlp_clf_leaky_relu.pt
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

- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4, MultiStepLR (×0.1 at epoch 100, 150), 200 epochs
- Best validation accuracy checkpoint restored after training

### MLP Classifier (`--mlp --activation <act>`)

| Component | Architecture |
|-----------|-------------|
| Layer 1 | Linear 64 → 10 |
| Layer 2 | Linear 10 → 32 |
| Layer 3 | Linear 32 → 10 (logits) |
| Activation | relu / sigmoid / tanh / leaky_relu (after layers 1 and 2) |
| Dropout | 0.3 after each activation |

> `Task-Aware` is automatically removed from the mechanism set when `--mlp` is used.

---

## LDP Mechanisms

All mechanisms satisfy ε-LDP (pure) or (ε, δ)-LDP with `δ = 1e-5`.  
Clipping radii `ρ` are set to the 90th percentile of the corresponding norm over `Z_tr`.

### Main registry (`build_mechs`, default)

| Name | Type | Description |
|------|------|-------------|
| `NoNoise` | — | No perturbation; upper bound on accuracy |
| `Laplace(L1)` | ε-LDP | L1 clip to `ρ`, Laplace noise scale `2ρ/ε` |
| `Laplace+PA` | ε-LDP | PA transform → L1 clip → Laplace |
| `AGM` | (ε,δ)-LDP | L2 clip → Gaussian (Balle & Wang 2018) |
| `AGM+PA` | (ε,δ)-LDP | PA transform → L2 clip → Gaussian |
| `PrivUnit2(Opt)` | ε-LDP | Spherical step-function on `S^{d-1}` |
| `PrivUnit2(Opt)+PA` | ε-LDP | PA transform → normalize → PrivUnit2 |
| `PrivUnitG(MC)` | ε-LDP | Gaussian ambient-space step-function (MC) |
| `PrivUnitG(Paper)` | ε-LDP | PrivUnitG with paper-exact parameters |
| `PrivUnitG(MC)+PA` | ε-LDP | PA transform → normalize → PrivUnitG(MC) |
| `PrivUnitG(Paper)+PA` | ε-LDP | PA transform → normalize → PrivUnitG(Paper) |
| `CW(Laplace)+PA` | ε-LDP | PA transform → coordinate-wise i.n.i.d. Laplace |
| `CW(AGM)+PA` | (ε,δ)-LDP | PA transform → coordinate-wise i.n.i.d. Gaussian |
| `PLAN(Pub)` | (ε,δ)-LDP | Variance-aware scaling → L2 clip → Gaussian |
| `PLAN(Paper)` | (ε,δ)-LDP | PLAN Algorithm 1 (Aumüller et al., private μ̃ and C) |
| `Inst-Opt` | (ε,δ)-LDP | Hadamard rotation + median shift + optimal L2 clip (d must be power of 2) |
| `Task-Aware` | ε-LDP | Cholesky whitening + water-filling noise (linear classifiers only) |

### Ablation registry (`build_mechs_ablation`, `--ablation`)

| Name | Ablation | Description |
|------|----------|-------------|
| `Laplace+PA` | full PA | Full PA pre+post processing (upper bound) |
| `PrivUnit2(Opt)+PA` | full PA | Full PA pre+post processing (upper bound) |
| `PrivUnitG(MC)+PA` | full PA | Full PA pre+post processing (upper bound) |
| `Laplace(L1)` | baseline | L1 clip → Laplace (no PA transform) |
| `Laplace+PA/NoPostProc` | NoPostProc | PA Pre-processing, identity Post-processing |
| `PrivUnit2(Opt)+PA/NoPostProc` | NoPostProc | PA Pre-processing, identity Post-processing |
| `PrivUnitG(MC)+PA/NoPostProc` | NoPostProc | PA Pre-processing, identity Post-processing |
| `Laplace+PA/NoReshaping` | NoReshaping | SVD rotation only, no anisotropic scaling |
| `PrivUnit2(Opt)/NoReshaping` | NoReshaping | SVD rotation only, no anisotropic scaling |
| `PrivUnitG(MC)/NoReshaping` | NoReshaping | SVD rotation only, no anisotropic scaling |
| `Laplace(L1)/NoPreProc` | NoPreProc | Raw-space encode, PA Post-processing |
| `PrivUnit2/NoPreProc` | NoPreProc | Raw-space encode, PA Post-processing |
| `PrivUnitG/NoPreProc` | NoPreProc | Raw-space encode, PA Post-processing |
| `PrivUnit2(L1,Opt)` | L1Clip | L1 clip in raw space + PrivUnit2 (no PA) |
| `PrivUnitG(L1,MC)` | L1Clip | L1 clip in raw space + PrivUnitG (no PA) |

### PA (Pre/Post-processing Adaptive) Transform

Rotates the latent space by the Jacobian row space of the downstream classifier so that task-sensitive directions receive proportionally less noise.

**Row-space extraction** (`compute_jacobian_row_space`):
- Stack per-sample Jacobians into `B ∈ ℝ^{(n·K)×D}`, SVD with gap-based rank threshold → `W_eff ∈ ℝ^{r×D}`.

**Encode / Decode:**
```
x     = (z − μ) @ U ⊙ (1/√λ)
[add noise in x-space]
z_dec = (x_noisy ⊙ √λ) @ U.T + μ
```

---

## Key Hyperparameters

| Constant | Value | Role |
|----------|-------|------|
| `PERCENTILE` | 90.0 | Clipping radius `ρ` percentile over `Z_tr` |
| `LAMBDA_N` | 1000.0 | Null-space noise amplifier |
| `DELTA` | 1e-5 | δ for all (ε,δ)-LDP mechanisms |

---

## Evaluation Grid

- `ε ∈ {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10}`
- `n_repeat = len(test_set) = 10,000` (all test samples)

---

## How to Run

### Prerequisites

```bash
# PyTorch with CUDA 11.8
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# CPU-only
pip install torch==2.7.1+cpu torchvision==0.22.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

### Step-by-step

```bash
# 1. Train ResNet-20 and extract 64-dim features
python train_resnet20.py

# Or with pretrained weights
python train_resnet20.py --weights path/to/resnet20.pth

# Train MLP heads (optional)
python train_resnet20.py --mlp --activation relu
python train_resnet20.py --mlp --activation tanh
python train_resnet20.py --mlp --activation sigmoid
python train_resnet20.py --mlp --activation leaky_relu

# 2. Run LDP evaluation (main registry)
python eval_cifar_classification.py

# MLP classifier
python eval_cifar_classification.py --mlp --activation relu

# Subset of mechanisms
python eval_cifar_classification.py --mechs "NoNoise" "Laplace+PA" "PrivUnit2(Opt)+PA"

# 3. Run ablation study
python eval_cifar_classification.py --ablation

# Subset of ablation mechanisms
python eval_cifar_classification.py --ablation --mechs "Laplace+PA/NoPostProc" "Laplace+PA/NoReshaping"

# 4. CIFAR-10-C distribution shift evaluation
# Download dataset (~2.8 GB)
bash download_cifar10c.sh

# Extract features for all corruptions at severity 1
python extract_cifar10c_features.py --c10c_dir data/CIFAR10-C/CIFAR-10-C --severity 1

# Extract specific corruptions
python extract_cifar10c_features.py --c10c_dir data/CIFAR10-C/CIFAR-10-C --severity 3 \
    --corrupt_types gaussian_noise shot_noise

# Evaluate on corrupted test set
python eval_cifar_classification.py --te_dir data/CIFAR10-C/latent/gaussian_noise_s1
python eval_cifar_classification.py --ablation --te_dir data/CIFAR10-C/latent/gaussian_noise_s1

# Save results
python eval_cifar_classification.py > results.txt 2>&1
python eval_cifar_classification.py --ablation > results_ablation.txt 2>&1
```

---

## References

- **PA (ANR-CW)**: Muthukrishnan, G., & Kalyani, S. (2025). Differential Privacy With Higher Utility by Exploiting Coordinate-Wise Disparity. *IEEE TIFS*.
- **PLAN**: Aumüller, M., Lebeda, C. J., Nelson, B., & Pagh, R. (2024). PLAN: Variance-Aware Private Mean Estimation. *PETs*.
- **PrivUnitG**: Asi, H., Feldman, V., & Talwar, K. (2022). Optimal Algorithms for Mean Estimation under Local Differential Privacy. *ICML*.
- **Inst-Opt**: Huang, Z., Liang, Y., & Yi, K. (2021). Instance-optimal Mean Estimation Under Differential Privacy. *NeurIPS*.
- **AGM**: Balle, B., & Wang, Y.-X. (2018). Improving the Gaussian Mechanism for Differential Privacy. *ICML*.
- **PrivUnit2(Opt)**: Bhowmick, A. et al. (2018). Protection Against Reconstruction. *arXiv:1812.00984*.
- **Task-Aware**: Cheng, X. et al. (2022). Locally Differentially Private Functional Statistics. *ICML*.
