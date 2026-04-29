# User-Level LDP Evaluation for Linear Regression (London Smart Meters)

Benchmarks Local Differential Privacy (LDP) mechanisms on the task of privately aggregating household energy feature vectors for linear regression under a **user-level privacy guarantee**: each household contributes to at most one aggregation across the entire evaluation, so the total privacy cost is strictly ε per household. MSE measures only the prediction distortion introduced by LDP, isolating it from model error.

---

## Overview

The experiment proceeds in three stages:

1. **Data preparation** (`prepare_data.py`) — loads the London Smart Meters daily dataset, builds sliding windows of size W (past W days → next day), splits households into **public** (first `N_PUBLIC_HH` by LCLid, used for training) and **private** (the rest, used for evaluation), normalizes features to `[0, 1]` using public-household train statistics, and saves per-date household metadata.
2. **Regressor training** (`train_regressor.py`) — fits OLS linear regression on the public training set. The weight vector `β` is saved as a **public model** and later used by ANR mechanisms to identify the task-sensitive direction.
3. **LDP evaluation** (`eval_regression.py`) — evaluates user-level LDP: private households are partitioned into `N_rounds` disjoint groups; within each group, non-overlapping batches of K households aggregate their encoded feature vectors per evaluation date. No household appears in more than one aggregation.

---

## Protocol

### User-level LDP guarantee

Before evaluation begins, the private household pool is **partitioned once** into `N_rounds` disjoint subsets of `group_size = ⌊n_private / N_rounds⌋` households each. Within each round:

- Households are further split into non-overlapping batches of exactly K.
- Each batch is assigned to exactly one evaluation date (at least W days from all other assigned dates in the same round).
- A household that participates in one batch is never used again.

Because each household appears in at most one aggregation across all rounds and all dates, the total privacy cost is **strictly ε per household**, regardless of `N_rounds` or `n_dates`.

**Consequence for large K**: a round of `group_size` households can supply at most `⌊group_size / K⌋` batches. When this is fewer than `n_dates`, the available batches are all used (the round is never skipped entirely).

### Per-round LDP exchange

**Client side (per household `i` on date `t`):**
- Feature vector `z_i ∈ ℝ^D`: normalized energy consumption over the past W days.
- Sends: `encode(z_i, ε)` — satisfies ε-LDP.

**Server side:**
- Aggregate: `z̄_enc = mean(encode(z_i))`, then `z̄_dec = mech.decode(z̄_enc)`.
- Predict: `ŷ = predict_fn(z̄_dec)`.

### MSE evaluation

Each client also appends raw `z_i` alongside the encoded vector — for evaluation only, not part of the LDP protocol. The server computes two predictions:

```
y_pred = predict_fn(decode(mean(encode(z_i))))   # LDP path
y_true = predict_fn(mean(z_i))                   # noiseless path (MSE baseline)
```

`MSE(y_pred, y_true)` measures LDP-induced prediction distortion only. By construction, `NoNoise` gives exactly 0.

### Evaluation protocol walk-through

The evaluation proceeds in the following order:

**Step 1 — Build reportable-household index per date (`prepare_data.py`)**

For every (household, date) pair, a sliding window of size W is attempted. If the household has W consecutive days of data ending on the day before `date`, the window is valid and a row `(Z, date, hh_id)` is added to `Z_te`. This produces `date_to_hhs[date]` = the set of households that can report on `date` (i.e., have the required prior W days).

**Step 2 — Shuffle and partition households (`eval_regression.py`)**

All private households are randomly shuffled and split into `n_rounds` disjoint groups:
```
group_size  = ⌊n_private / n_rounds⌋
Round r     = shuffled_hh[r * group_size : (r+1) * group_size]
```
Leftover households (`n_private % n_rounds`) are discarded.

**Step 3 — Assign a fixed priority order within each round**

Within each round `r`, households are permuted by a seeded RNG (`hh_perm`). This order determines which households are picked first on any given date.

**Step 4 — Iterate dates chronologically to form batches**

For each round, dates are traversed in ascending order. A date is eligible if:
1. It is at least W days after the previously selected date (gap constraint).
2. At least K households in the round that (a) have data on this date and (b) have not yet been used can be found in `hh_perm` order.

If either condition fails the date is skipped and the candidate households remain available for future dates. When K households are found, they are marked as used and the batch is recorded. This repeats until `n_dates` batches are collected or dates are exhausted.

**Step 5 — LDP evaluation per batch**

Each selected batch of K households encodes its feature vector with the chosen mechanism at privacy budget ε. The server averages the encoded vectors, decodes, and predicts. MSE is computed against the noiseless prediction and accumulated across all batches and rounds.

---

## File Structure

```
.
├── prepare_data.py            # Stage 1: build sliding windows, split, normalize, save
├── train_regressor.py         # Stage 2: fit OLS on public HHs, save β and intercept
├── eval_regression.py         # Stage 3: user-level LDP evaluation across mechanisms and ε/K grids
├── ../mechanisms/regression_mechanisms.py  # All LDP mechanism implementations
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
  - Multi-output (`--multi`): `Z = energy_mean × W`, `Y = all 5 targets`.

### Regressor (`train_regressor.py`)

OLS with intercept via `np.linalg.lstsq`. Trained on normalized public-household features and original-scale labels.

```
[Z | 1] · [β; b] ≈ y
ŷ = Z · β + b
```

- **Single-output**: `β ∈ ℝ^D`, `b ∈ ℝ`.
- **Multi-output**: `B ∈ ℝ^{D×5}`, `intercepts ∈ ℝ^5`, task matrix `W = B.T ∈ ℝ^{5×D}`.

The weight vector `β` (or matrix `W`) defines the task-sensitive direction used by ANR mechanisms.

### Prediction scale (`eval_regression.py`)

By default, predictions are normalized to `[0, 1]` using label train statistics so that MSE is scale-invariant across targets. Use `--denormalize` to report MSE in original energy units.

---

## LDP Mechanisms

All mechanisms satisfy ε-LDP (pure) or (ε, δ)-LDP with `δ = 1e-5`.  
Clipping radii `ρ` are set to the 90th percentile of the corresponding norm over `Z_tr`.

| Name | Type | Description |
|------|------|-------------|
| `NoNoise` | — | No perturbation; gives MSE = 0 by construction |
| `Laplace(L1)` | ε-LDP | L1 clip to `ρ`, Laplace noise scale `2ρ/ε` |
| `Laplace(L∞)` | ε-LDP | L∞ clip to `ρ`, Laplace noise `2ρd/ε` per coordinate |
| `AGM` | (ε,δ)-LDP | L2 clip to `ρ`, AGM noise (Balle & Wang 2018) |
| `Duchi` | ε-LDP | L∞ clip → scale to `[-1,1]` → Duchi mechanism |
| `Piecewise` | ε-LDP | L∞ clip → scale to `[-1,1]` → Piecewise mechanism |
| `ANR-SV(L1,Lap)` | ε-LDP | ANR-SV transform → L1 clip → Laplace |
| `ANR-SV(L∞,Lap)` | ε-LDP | ANR-SV transform → L∞ clip → Laplace |
| `ANR-SV(L2,AGM)` | (ε,δ)-LDP | ANR-SV transform → L2 clip → AGM noise |
| `ANR-SV+Duchi` | ε-LDP | ANR-SV transform → Duchi in transformed space |
| `ANR-SV+Piecewise` | ε-LDP | ANR-SV transform → Piecewise in transformed space |
| `ANR-SV+PrivUnit2` | ε-LDP | ANR-SV transform → normalize to `S^{d-1}` → PrivUnit2 |
| `ANR-SV+PrivUnitG` | ε-LDP | ANR-SV transform → normalize to `S^{d-1}` → PrivUnitG |
| `ANR-SV-CW(Lap)` | ε-LDP | ANR-SV transform → coordinate-wise i.n.i.d. Laplace, budget split `∝ λ_i^{1/3}` |
| `ANR-SV-CW(AGM)` | (ε,δ)-LDP | ANR-SV transform → coordinate-wise i.n.i.d. Gaussian, budget split minimizing `Σ σ_i²` |
| `PLAN` | (ε,δ)-LDP | Variance-aware scaling → L2 clip → Gaussian noise |
| `Shifted-CM` | (ε,δ)-LDP | Hadamard rotation → per-dim median shift → AGM noise |
| `PrivUnit2` | ε-LDP | Optimal spherical step-function on `S^{d-1}` |
| `PrivUnitG` | ε-LDP | Gaussian ambient-space step-function |
| `TASK(Cheng22)` | ε-LDP | Cholesky whitening → task-relevant rotation → water-filling → Laplace |

### ANR-SV Transform

**ANR** (Anisotropic Noise Randomization) rotates the feature space so that task-sensitive directions receive proportionally less noise.

In this linear regression setting, the **task-relevant subspace** is the regressor weight vector `β` (single-output, passed as `W = β.reshape(1, -1)`) or the full task matrix `W = B.T` (multi-output), replacing the Jacobian row space used in the VAE/classifier setting.

**SV-weighted λ allocation** (Lagrange optimum for `min Σ s_i² λ_i` s.t. `Σ 1/√λ_i = C`):
```
1/√λ_i ∝ s_i^{2/3}   (row space, i = 1…r)
1/√λ_i = 1/√λ_N      (null space, i = r+1…d,  λ_N = 1000)
```

**Encode / Decode**:
```
x     = (z − μ) @ U ⊙ (1/√λ)          # anisotropic scaling
[add noise in x-space]
z_dec = (x_noisy ⊙ √λ) @ U.T + μ      # invert scaling
```

---

## Key Hyperparameters

| Constant | Value | Location | Role |
|----------|-------|----------|------|
| `WINDOW` | 16 | `prepare_data.py`, `train_regressor.py` | Sliding window size (days) |
| `WINDOW` | 16 | `eval_regression.py` | Default window for evaluation |
| `SPLIT_DATE` | `2013-01-01` | `prepare_data.py` | Train/test date cutoff |
| `N_PUBLIC_HH` | 3000 | `prepare_data.py`, `eval_regression.py` | Households used for public training; private HHs have `id ≥ N_PUBLIC_HH` |
| `PERCENTILE` | 90.0 | `regression_mechanisms.py` | Clipping radius `ρ` percentile |
| `LAMBDA_N` | 1000.0 | `regression_mechanisms.py` | Null-space noise amplifier |
| `DELTA` | 1e-5 | `regression_mechanisms.py` | δ for all (ε,δ)-LDP mechanisms |

---

## Evaluation Grid

- `ε ∈ {0.5, 1.0, …, 10.0}` (step 0.5) — privacy budget
- `k ∈ {1, 2, 5, 10, 20, 50, 100}` — households per aggregation batch; capped at `group_size = ⌊n_private / n_rounds⌋`
- `--n_rounds` (default: 20) — number of disjoint household partitions; with ~2547 private households and 20 rounds, `group_size ≈ 127`, so the maximum K is 127
- `--n_dates` (default: 20) — target number of evaluation batches per round; the actual count may be fewer when `⌊group_size / K⌋ < n_dates`

Each evaluation batch uses K households that have never participated before, assigned to a date at least W days from any other used date in the same round. MSE is averaged over all collected batches across all rounds.

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

### Step-by-step (single-output)

```bash
# 1. Prepare data (public/private household split)
python prepare_data.py --window 16 --target energy_mean

# 2. Train regressor on public households
python train_regressor.py --window 16 --target energy_mean

# 3. Run user-level LDP evaluation on private households
python eval_regression.py --window 16 --target energy_mean --n_dates 20 --n_rounds 20
```

### Step-by-step (multi-output)

```bash
# 1. Prepare data
python prepare_data.py --window 16 --multi

# 2. Train regressor
python train_regressor.py --window 16 --multi

# 3. Run LDP evaluation
python eval_regression.py --window 16 --multi --n_dates 20 --n_rounds 20
```

### Evaluate a subset of mechanisms

```bash
python eval_regression.py --window 16 --target energy_mean \
    --mechs "NoNoise" "ANR-SV(L1,Lap)" "Laplace(L1)"
```

### Measure utility in original energy scale

```bash
python eval_regression.py --window 16 --target energy_mean --denormalize
```

---

## References

- **ANR-CW (coordinate-wise i.n.i.d. noise)**: Muthukrishnan, G., & Kalyani, S. (2025). Differential Privacy With Higher Utility by Exploiting Coordinate-Wise Disparity: Laplace Mechanism Can Beat Gaussian in High Dimensions. *IEEE Transactions on Information Forensics and Security*.

- **PLAN**: Aumüller, M., Lebeda, C. J., Nelson, B., & Pagh, R. (2024). PLAN: Variance-Aware Private Mean Estimation. *Proceedings on Privacy Enhancing Technologies*.

- **PrivUnitG**: Asi, H., Feldman, V., & Talwar, K. (2022). Optimal Algorithms for Mean Estimation under Local Differential Privacy. *ICML 2022*.

- **TASK(Cheng22)**: Cheng, J., Tang, A., & Chinchali, S. (2022). Task-aware Privacy Preservation for Multi-dimensional Data. *ICML 2022*. PMLR 162.

- **Shifted-CM**: Huang, Z., Liang, Y., & Yi, K. (2021). Instance-optimal Mean Estimation Under Differential Privacy. *NeurIPS 2021*.

- **Piecewise**: Wang, S., Huang, Z., Nie, T., Hu, Q., Wang, Y., & Skoglund, M. (2019). Local Differential Privacy for Data Collection and Analysis. *arXiv:1906.01777*.

- **AGM**: Balle, B., & Wang, Y.-X. (2018). Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising. *ICML 2018*.

- **PrivUnit2**: Bhowmick, A., Duchi, J., Freudiger, J., Kapoor, G., & Rogers, R. (2018). Protection Against Reconstruction and Its Applications in Private Federated Learning. *arXiv:1812.00984*.

- **Duchi**: Duchi, J., Jordan, M. I., & Wainwright, M. J. (2013). Local Privacy and Statistical Minimax Rates. *FOCS 2013*.
