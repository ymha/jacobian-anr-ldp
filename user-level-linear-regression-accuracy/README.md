# User-Level LDP Evaluation for Linear Regression (London Smart Meters)

Benchmarks Local Differential Privacy (LDP) mechanisms on the task of privately collecting household energy feature vectors for linear regression under a **user-level privacy guarantee**: each household contributes to at most one aggregation across the entire evaluation, so the total privacy cost is strictly ε per household. Utility is measured as RMSE against actual ground-truth labels (denormalized).

---

## Overview

The experiment proceeds in three stages:

1. **Data preparation** (`prepare_data.py`) — loads the London Smart Meters daily dataset, builds sliding windows of size W (past W days → next day), splits households into **public** (first `N_PUBLIC_HH` by LCLid, used for training) and **private** (the rest, used for evaluation), normalizes features to `[0, 1]` using public-household train statistics, and saves per-date household metadata.
2. **Regressor training** (`train_regressor.py`) — fits OLS linear regression on the public training set. The weight vector `β` is saved as a **public model** and later used by PA (Proposed Approach) to identify the task-sensitive direction.
3. **LDP evaluation** (`eval_regression.py`) — evaluates user-level LDP with K=1: private households are partitioned into `N_rounds` disjoint groups; within each group, one household per evaluation date encodes its feature vector and the server predicts using the decoded output. No household appears in more than one aggregation.

---

## Protocol

### User-level LDP guarantee

Before evaluation begins, the private household pool is **partitioned once** into `N_rounds` disjoint subsets of `group_size = ⌊n_private / N_rounds⌋` households each. Within each round:

- A fixed random priority order (`hh_perm`) is assigned to households in the round.
- For each eligible date (at least W days from the previously used date), the first unused household in `hh_perm` that has data on that date is selected.
- A household that participates on one date is never used again.

Because each household appears at most once across all rounds and all dates, the total privacy cost is **ε per household**, regardless of `N_rounds` or `n_dates`.

### Per-round LDP exchange (K=1)

**Client side (household `i` on date `t`):**
- Feature vector `z_i ∈ ℝ^D`: normalized energy consumption over the past W days.
- Sends: `encode(z_i, ε)` — satisfies ε-LDP.

**Server side:**
- Decode: `z̄_dec = mech.decode(encode(z_i))`.
- Predict: `ŷ = predict_fn_denorm(z̄_dec)`.
- Measure: `RMSE(ŷ, y_real)` against the actual next-day energy label.

### Evaluation protocol walk-through

**Step 1 — Build reportable-household index per date (`prepare_data.py`)**

For every (household, date) pair, a sliding window of size W is attempted. If the household has W consecutive days of data ending on the day before `date`, the window is valid and a row `(Z, y, date, hh_id)` is added to `Z_te`. This produces `date_to_hhs[date]` = the set of households that can report on `date`.

**Step 2 — Shuffle and partition households (`eval_regression.py`)**

All private households are randomly shuffled and split into `n_rounds` disjoint groups:
```
group_size  = ⌊n_private / n_rounds⌋
Round r     = shuffled_hh[r * group_size : (r+1) * group_size]
```
Leftover households (`n_private % n_rounds`) are discarded.

**Step 3 — Assign a fixed priority order within each round**

Within each round `r`, households are permuted by a seeded RNG (`hh_perm`). This order determines which household is picked first on any given date.

**Step 4 — Iterate dates chronologically to select one household per date**

For each round, dates are traversed in ascending order. A date is eligible if:
1. It is at least W days after the previously selected date (gap constraint).
2. At least one household in the round that (a) has data on this date and (b) has not yet been used can be found in `hh_perm` order.

If either condition fails the date is skipped. When a household is found, it is marked as used and the sample is recorded. This repeats until `n_dates` samples are collected or dates are exhausted.

**Step 5 — LDP evaluation per sample**

Each selected household encodes its feature vector with the chosen mechanism at privacy budget ε. The server decodes and predicts in original energy units. Squared error is accumulated against the actual label across all samples and rounds; RMSE is reported per mechanism per ε.

---

## File Structure

```
.
├── prepare_data.py            # Stage 1: build sliding windows, split, normalize, save
├── train_regressor.py         # Stage 2: fit OLS on public HHs, save β and intercept
├── eval_regression.py         # Stage 3: user-level LDP evaluation across mechanisms and ε
├── ../mechanisms/mechanisms.py         # All LDP mechanism implementations
├── requirements.txt
├── checkpoints/
│   ├── regressor_w{W}_{target}.npz   # β (shape D) and intercept (single-output)
│   └── regressor_w{W}_multi.npz      # B (D×5), W (5×D), intercepts (multi-output)
└── data/london/
    ├── window{W}_{target}/    # single-output: one target stat
    │   ├── Z_tr.pt, Z_te.pt   # Normalized feature windows (public HHs / private HHs)
    │   ├── y_tr.pt, y_te.pt   # Labels: next-day energy stat (original scale)
    │   ├── z_min.npy          # Per-dimension train min (from public HHs)
    │   ├── z_scale.npy        # Per-dimension train range (from public HHs)
    │   ├── ids_te.npy         # Household integer code per test row (private HHs only)
    │   └── dates_te.npy       # Date per test row (used to group by date)
    └── window{W}_multi/       # multi-output: Z=energy_mean, Y=all 5 targets
        ├── Z_tr.pt, Z_te.pt
        ├── y_tr.pt, y_te.pt   # Labels shape: (N, 5)
        ├── z_min.npy
        ├── z_scale.npy
        ├── ids_te.npy
        └── dates_te.npy
```

---

## Models

### Data (`prepare_data.py`)

- **Source**: `daily_dataset.csv` — per-household daily energy statistics.
- **Filter**: rows where `energy_count == 48` (complete 30-min readings only).
- **Window**: `[day_t, …, day_{t+W-1}]` → predict `day_{t+W}`.
- **Household split**: households sorted by `LCLid`; first `N_PUBLIC_HH` (default: 3000) are **public**, the rest are **private**.
- **Train set**: public households × dates before `2013-01-01`.
- **Test set**: private households × dates on/after `2013-01-01`.
- **Feature normalization**: per-dimension min-max to `[0, 1]` using public-household train statistics.
- **Modes**:
  - Single-output (default): one of `{energy_mean, energy_median, energy_max, energy_min, energy_std}`.

### Regressor (`train_regressor.py`)

OLS with intercept via `np.linalg.lstsq`. Trained on normalized public-household features and original-scale labels.

```
[Z | 1] · [β; b] ≈ y
ŷ = Z · β + b
```

- **Single-output**: `β ∈ ℝ^D`, `b ∈ ℝ`.

The weight vector `β` (or matrix `W`) defines the task-sensitive direction used by ANR mechanisms.

### Prediction scale (`eval_regression.py`)

Predictions are always in original energy units (kWh). `predict_fn_denorm` inverts the feature normalization before applying the regression weights:

```
ŷ = (z * z_scale + z_min) @ β_orig + intercept_orig
```

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
| `WINDOW` | 16 | `prepare_data.py`, `train_regressor.py` | Sliding window size (days) |
| `WINDOW` | 16 | `eval_regression.py` | Default window for evaluation |
| `SPLIT_DATE` | `2013-01-01` | `prepare_data.py` | Train/test date cutoff |
| `N_PUBLIC_HH` | 3000 | `prepare_data.py`, `eval_regression.py` | Households used for public training; private HHs have `id ≥ N_PUBLIC_HH` |
| `PERCENTILE` | 90.0 | `mechanisms.py` | Clipping radius `ρ` percentile |
| `LAMBDA_N` | 1000.0 | `mechanisms.py` | Null-space noise amplifier |
| `DELTA` | 1e-5 | `mechanisms.py` | δ for all (ε,δ)-LDP mechanisms |

---

## Evaluation Grid

- `ε ∈ {0.5, 1.0, …, 10.0}` (step 0.5) — privacy budget
- `K = 1` — fixed; one household per aggregation
- `--n_rounds` (default: 20) — number of disjoint household partitions; with ~2547 private households and 20 rounds, `group_size = 127`
- `--n_dates` (default: 20) — target number of evaluation samples per round

Each evaluation sample uses one household that has never participated before, selected on a date at least W days from any other used date in the same round. RMSE is averaged over all collected samples across all rounds.

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

### Step 0 — Download data

The dataset is [London Smart Meters](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) on Kaggle. Set up your API key first:

```bash
pip install kaggle
# place ~/.kaggle/kaggle.json with your credentials
# https://www.kaggle.com/docs/api#authentication
```

Then download and prepare in one command:

```bash
python download_data.py                  # download + prepare all targets
python download_data.py --skip-download  # skip download if csv already present
```

### Step-by-step 

```bash
# 1. Prepare data (public/private household split)
# Default CSV: ./raw_data/daily_dataset.csv  Override with --data_csv <path>
python prepare_data.py --window 16 --target energy_mean

# 2. Train regressor on public households
python train_regressor.py --window 16 --target energy_mean

# 3. Run user-level LDP evaluation on private households
python eval_regression.py --window 16 --target energy_mean --n_dates 130 --n_rounds 20
```

### Evaluate a subset of mechanisms

```bash
python eval_regression.py --window 16 --target energy_mean \
    --mechs "NoNoise" "Laplace+PA" "Laplace(L1)"
```

---

## References

- **Coordinate-Wise (CW) (i.n.i.d. noise)**: Muthukrishnan, G., & Kalyani, S. (2025). Differential Privacy With Higher Utility by Exploiting Coordinate-Wise Disparity: Laplace Mechanism Can Beat Gaussian in High Dimensions. *IEEE Transactions on Information Forensics and Security*.

- **PLAN**: Aumüller, M., Lebeda, C. J., Nelson, B., & Pagh, R. (2024). PLAN: Variance-Aware Private Mean Estimation. *Proceedings on Privacy Enhancing Technologies*.

- **PrivUnitG**: Asi, H., Feldman, V., & Talwar, K. (2022). Optimal Algorithms for Mean Estimation under Local Differential Privacy. *ICML 2022*.

- **Task-Aware**: Cheng, J., Tang, A., & Chinchali, S. (2022). Task-aware Privacy Preservation for Multi-dimensional Data. *ICML 2022*. PMLR 162.

- **Inst-Opt**: Huang, Z., Liang, Y., & Yi, K. (2021). Instance-optimal Mean Estimation Under Differential Privacy. *NeurIPS 2021*.

- **AGM**: Balle, B., & Wang, Y.-X. (2018). Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising. *ICML 2018*.

- **PrivUnit2**: Bhowmick, A., Duchi, J., Freudiger, J., Kapoor, G., & Rogers, R. (2018). Protection Against Reconstruction and Its Applications in Private Federated Learning. *arXiv:1812.00984*. Optimal parameters from Asi, H., Feldman, V., & Talwar, K. (2022). Optimal Algorithms for Mean Estimation under Local Differential Privacy. *ICML 2022*.

