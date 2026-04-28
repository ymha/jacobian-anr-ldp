"""
LDP evaluation for linear regression on London Smart Meters data.

User-level LDP protocol:
  The private household pool is partitioned into N_rounds disjoint subsets before
  evaluation. Each round r is assigned an exclusive subset of ⌊n_private/N_rounds⌋
  households. Within round r, households are further partitioned into non-overlapping
  batches of exactly K. Each batch is assigned to one evaluation date (≥ D days from
  all other assigned dates in the round). A given household therefore appears in at
  most one aggregation across the entire evaluation, so the privacy cost is strictly
  ε per household regardless of N_rounds or the number of evaluation dates.

  Consequence for large K: a round with group_size households can supply at most
  ⌊group_size / K⌋ non-overlapping batches. When this is smaller than n_dates,
  the round contributes fewer evaluation points (never skipped entirely).

MSE evaluation:
  Each client appends raw z_i alongside the encoded vector for evaluation only.
  Server computes two predictions:
    y_pred = predict_fn(decode(mean(encode(z_i))))   [LDP path]
    y_true = predict_fn(mean(z_i))                   [noiseless path, MSE baseline]
  Metric: MSE(y_pred, y_true)
  → NoNoise gives exactly 0; MSE measures only LDP-induced prediction distortion.

Usage:
  python eval_regression.py --window 16 --target energy_mean --n_dates 20 --n_rounds 20
  python eval_regression.py --window 16 --multi
  python eval_regression.py --window 16 --mechs "NoNoise" "ANR-SV(L1,Lap)" "Laplace(L1)"
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "mechanisms"))
from regression_mechanisms import build_regression_mechs

WINDOW      = 16
N_PUBLIC_HH = 3000
DATA_BASE   = "./data/london"
CKPT_DIR    = "./checkpoints"
TARGETS     = ["energy_mean", "energy_median", "energy_max", "energy_min", "energy_std"]


def _load_data(data_dir: str):
    def _pt(name):
        return torch.load(f"{data_dir}/{name}", weights_only=True).numpy().astype(np.float64)
    return (
        _pt("Z_tr.pt"),
        _pt("Z_te.pt"),
        _pt("y_tr.pt"),
        np.load(f"{data_dir}/z_min.npy").astype(np.float64),
        np.load(f"{data_dir}/z_scale.npy").astype(np.float64),
        np.load(f"{data_dir}/dates_te.npy"),
        np.load(f"{data_dir}/ids_te.npy"),
    )


def evaluate(mechs, Z_te, dates_te, ids_te, predict_fn,
             eps_list, K_list, n_dates, n_rounds, seed=0):
    """Evaluate LDP distortion under the user-level LDP round protocol.

    Households in each round are split into non-overlapping batches of K before
    evaluation. Each batch is assigned to exactly one evaluation date (≥ D days
    from all other dates in the round). No household appears in more than one
    aggregation, guaranteeing a privacy cost of strictly ε per household.

    A round provides at most ⌊group_size / K⌋ evaluation points. When this is
    less than n_dates, the available points are used (round is never skipped).
    """
    names  = list(mechs.keys())
    col_w  = 18
    D      = Z_te.shape[1]

    unique_hh  = np.unique(ids_te)
    n_hh       = len(unique_hh)
    group_size = n_hh // n_rounds

    # Partition households into n_rounds disjoint groups (fixed across all eps/K)
    shuffled_hh = np.random.default_rng(seed).permutation(unique_hh)

    # Precompute per-round data (independent of K): data slices and date→HH map.
    round_data: list = []
    for r in range(n_rounds):
        round_hh = shuffled_hh[r * group_size : (r + 1) * group_size]
        mask     = np.isin(ids_te, round_hh)
        Z_r      = Z_te[mask]
        dates_r  = dates_te[mask]
        ids_r    = ids_te[mask]
        date_to_hhs: dict = {
            date: frozenset(int(h) for h in ids_r[dates_r == date])
            for date in np.unique(dates_r)
        }
        round_data.append((round_hh, Z_r, dates_r, ids_r, date_to_hhs))

    # Pre-build evaluation pairs (date, Z_K) per round per K.
    # Iterates dates chronologically (≥ D gap). On each eligible date, selects
    # the first K unused round-members that have a record that day (priority
    # determined by hh_perm, a fixed per-round permutation). Selected households
    # are marked used and never drawn again. This guarantees each household
    # appears in at most one aggregation across the entire evaluation.
    round_eligible: dict[int, list] = {}
    for K in K_list:
        if K > group_size:
            continue
        per_round = []
        for r in range(n_rounds):
            round_hh, Z_r, dates_r, ids_r, date_to_hhs = round_data[r]
            # Fixed priority order (seed includes K to decorrelate across K values)
            hh_perm = [int(h) for h in
                       np.random.default_rng([seed, r, int(K)]).permutation(round_hh)]

            selected: list = []
            used_hhs: set  = set()
            last_date      = None

            for date in sorted(date_to_hhs):
                gap = (np.inf if last_date is None
                       else float((date - last_date) / np.timedelta64(1, "D")))
                if gap < D:
                    continue

                hhs_today = date_to_hhs[date]
                # First K unused HHs present today, in fixed permutation order
                batch = [h for h in hh_perm
                         if h in hhs_today and h not in used_hhs][:K]
                if len(batch) < K:
                    continue

                used_hhs.update(batch)

                d_mask  = dates_r == date
                Z_day   = Z_r[d_mask]
                ids_day = ids_r[d_mask]
                b_mask  = np.isin(ids_day, batch)
                Z_b     = Z_day[b_mask]
                ids_b   = ids_day[b_mask]
                order   = np.argsort(ids_b)        # deterministic row order
                selected.append((date, Z_b[order]))
                last_date = date

                if len(selected) >= n_dates:
                    break

            per_round.append(selected)
        round_eligible[K] = per_round

    for eps in eps_list:
        print(f"\neps={eps}  D={D}  n_rounds={n_rounds}  group_size={group_size}")
        print(f"{'K':>6}  " + "".join(f"{n:>{col_w}}" for n in names))
        print("-" * (8 + len(names) * col_w))

        for K in K_list:
            if K > group_size:
                continue

            vals = {n: [] for n in names}

            for r, eligible in enumerate(round_eligible[K]):
                # Use however many batches are available (≤ n_dates)
                for s in range(min(n_dates, len(eligible))):
                    _, Z_samp = eligible[s]   # K×D, households pre-assigned; no sampling

                    for name, mech in mechs.items():
                        messages = [
                            np.hstack([
                                mech.encode(Z_samp[i:i+1], eps,
                                            rng=np.random.default_rng([seed, r, s, i])),
                                Z_samp[i:i+1],
                            ])
                            for i in range(K)
                        ]

                        msg_bar   = np.vstack(messages).mean(axis=0, keepdims=True)
                        z_enc_bar = msg_bar[:, :D]
                        z_raw_bar = msg_bar[:, D:]

                        y_pred = predict_fn(mech.decode(z_enc_bar))
                        y_true = predict_fn(z_raw_bar)
                        v      = float(np.mean((y_pred - y_true) ** 2))
                        if not np.isnan(v):
                            vals[name].append(v)

            n_evals = sum(len(v) for v in vals.values()) // max(len(vals), 1)
            print(f"{K:>6}  " + "".join(
                f"{np.mean(v):>{col_w}.4f}" if v else f"{'nan':>{col_w}}"
                for v in vals.values()
            ) + f"  (n={n_evals})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window",       type=int, default=WINDOW)
    p.add_argument("--target",       type=str, default="energy_mean", choices=TARGETS)
    p.add_argument("--multi",        action="store_true")
    p.add_argument("--n_dates",      type=int, default=20,
                   help="evaluation dates per round (must be ≤ eligible dates in each round)")
    p.add_argument("--n_rounds",     type=int, default=20,
                   help="number of disjoint household partitions")
    p.add_argument("--n_public_hh",  type=int, default=N_PUBLIC_HH,
                   help="households with id < this threshold are treated as public; "
                        "test data is filtered to private households (id ≥ threshold)")
    p.add_argument("--mechs",        nargs="*", default=None)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--denormalize",  action="store_true",
                   help="report MSE in original energy scale instead of normalized")
    args = p.parse_args()

    if args.multi:
        data_dir = f"{DATA_BASE}/window{args.window}_multi"
        Z_tr, Z_te, y_tr, z_min, z_scale, dates_te, ids_te = _load_data(data_dir)

        ckpt       = np.load(f"{CKPT_DIR}/regressor_w{args.window}_multi.npz")
        B          = ckpt["B"]
        W          = ckpt["W"]
        intercepts = ckpt["intercepts"]
        B_orig          = B / z_scale[:, None]
        intercepts_orig = intercepts - z_min @ B_orig

        y_min_y   = y_tr.min(axis=0)
        y_scale_y = np.maximum(y_tr.max(axis=0) - y_min_y, 1e-8)

        if args.denormalize:
            predict_fn = lambda z: ((z * z_scale + z_min) @ B_orig).flatten() + intercepts_orig
        else:
            predict_fn = lambda z: ((z @ B).flatten() + intercepts - y_min_y) / y_scale_y

        print(f"Z_te={Z_te.shape}  W={W.shape}")
    else:
        data_dir = f"{DATA_BASE}/window{args.window}_{args.target}"
        Z_tr, Z_te, y_tr, z_min, z_scale, dates_te, ids_te = _load_data(data_dir)

        ckpt      = np.load(f"{CKPT_DIR}/regressor_w{args.window}_{args.target}.npz")
        beta      = ckpt["beta"]
        intercept = float(ckpt["intercept"][0])
        W         = beta.reshape(1, -1)
        beta_orig      = beta / z_scale
        intercept_orig = intercept - float(z_min @ beta_orig)

        y_min_y   = float(y_tr.min())
        y_scale_y = float(np.maximum(y_tr.max() - y_min_y, 1e-8))

        if args.denormalize:
            predict_fn = lambda z: float(((z * z_scale + z_min) @ beta_orig).flat[0]) + intercept_orig
        else:
            predict_fn = lambda z: (float((z @ beta).flat[0]) + intercept - y_min_y) / y_scale_y

        print(f"Z_te={Z_te.shape}  β_orig={beta_orig}")

    # Filter to private households only (ids ≥ n_public_hh).
    # When data was prepared with prepare_data.py (which already applies this split),
    # this is a no-op. For legacy data containing all households, this enforces the
    # public/private split required by the user-level LDP evaluation protocol.
    if ids_te.min() < args.n_public_hh:
        priv = ids_te >= args.n_public_hh
        Z_te, dates_te, ids_te = Z_te[priv], dates_te[priv], ids_te[priv]
        print(f"Filtered to private HHs (ids≥{args.n_public_hh}): "
              f"{np.unique(ids_te).size} households, {len(Z_te):,} rows")

    mechs = build_regression_mechs(args.window, W, Z_tr)
    if args.mechs:
        mechs = {k: v for k, v in mechs.items() if k in args.mechs}

    group_size = np.unique(ids_te).size // args.n_rounds
    print(f"Private households: {np.unique(ids_te).size}  "
          f"n_rounds={args.n_rounds}  group_size={group_size}  max K={group_size}")

    eps_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    # K must be ≤ group_size (= ⌊n_private / n_rounds⌋).
    # With the default setup (2547 HHs, 20 rounds) group_size = 127.
    K_list   = [1, 2, 5, 10, 20, 50, 100]

    evaluate(mechs, Z_te, dates_te, ids_te, predict_fn,
             eps_list, K_list, args.n_dates, args.n_rounds, args.seed)


if __name__ == "__main__":
    main()
