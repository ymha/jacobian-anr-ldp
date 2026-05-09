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

Accuracy evaluation (K=1):
  RMSE of denormalized LDP predictions vs actual labels.

Usage:
  python eval_regression.py --window 16 --target energy_mean --n_dates 20 --n_rounds 20
  python eval_regression.py --window 16 --multi
  python eval_regression.py --window 16 --mechs "NoNoise" "Laplace+PA" "Laplace(L1)"
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "mechanisms"))
from mechanisms import build_mechs as build_regression_mechs

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
        _pt("y_te.pt"),
        np.load(f"{data_dir}/z_min.npy").astype(np.float64),
        np.load(f"{data_dir}/z_scale.npy").astype(np.float64),
        np.load(f"{data_dir}/dates_te.npy"),
        np.load(f"{data_dir}/ids_te.npy"),
    )


def evaluate_accuracy(mechs, Z_te, y_te, dates_te, ids_te,
                         predict_fn_denorm, eps_list, n_dates, n_rounds, seed=0):
    """K=1 only: RMSE of denormalized LDP predictions vs actual labels."""
    names  = list(mechs.keys())
    col_w  = 18
    D      = Z_te.shape[1]

    unique_hh  = np.unique(ids_te)
    n_hh       = len(unique_hh)
    group_size = n_hh // n_rounds

    shuffled_hh = np.random.default_rng(seed).permutation(unique_hh)

    round_data = []
    for r in range(n_rounds):
        round_hh = shuffled_hh[r * group_size : (r + 1) * group_size]
        mask     = np.isin(ids_te, round_hh)
        Z_r      = Z_te[mask]
        y_r      = y_te[mask]
        dates_r  = dates_te[mask]
        ids_r    = ids_te[mask]
        date_to_hhs = {
            date: frozenset(int(h) for h in ids_r[dates_r == date])
            for date in np.unique(dates_r)
        }
        round_data.append((round_hh, Z_r, y_r, dates_r, ids_r, date_to_hhs))

    K = 1
    per_round = []
    for r in range(n_rounds):
        round_hh, Z_r, y_r, dates_r, ids_r, date_to_hhs = round_data[r]
        hh_perm = [int(h) for h in
                   np.random.default_rng([seed, r, K]).permutation(round_hh)]

        selected  = []
        used_hhs  = set()
        last_date = None

        for date in sorted(date_to_hhs):
            gap = (np.inf if last_date is None
                   else float((date - last_date) / np.timedelta64(1, "D")))
            if gap < D:
                continue

            hhs_today = date_to_hhs[date]
            batch = [h for h in hh_perm
                     if h in hhs_today and h not in used_hhs][:K]
            if len(batch) < K:
                continue

            used_hhs.update(batch)

            d_mask  = dates_r == date
            Z_day   = Z_r[d_mask]
            y_day   = y_r[d_mask]
            ids_day = ids_r[d_mask]
            b_mask  = np.isin(ids_day, batch)
            Z_b     = Z_day[b_mask]
            y_b     = y_day[b_mask]
            order   = np.argsort(ids_day[b_mask])
            selected.append((date, Z_b[order], y_b[order]))
            last_date = date

            if len(selected) >= n_dates:
                break

        per_round.append(selected)

    print(f"\n[K=1 Accuracy: RMSE vs real labels, denormalized]")
    print(f"n_rounds={n_rounds}  group_size={group_size}")
    print(f"{'eps':>6}  " + "".join(f"{n:>{col_w}}" for n in names))
    print("-" * (8 + len(names) * col_w))

    for eps in eps_list:
        vals = {n: [] for n in names}

        for r, eligible in enumerate(per_round):
            for s, (_, Z_b, y_b) in enumerate(eligible[:n_dates]):
                y_real = y_b[0]
                Z_i    = Z_b[0:1]

                for name, mech in mechs.items():
                    z_enc  = mech.encode(Z_i, eps, rng=np.random.default_rng([seed, r, s, 0]))
                    y_pred = predict_fn_denorm(mech.decode(z_enc))
                    err = float(np.mean(
                        (np.atleast_1d(y_pred) - np.atleast_1d(y_real)) ** 2
                    ))
                    if not np.isnan(err):
                        vals[name].append(err)

        n_evals = sum(len(v) for v in vals.values()) // max(len(vals), 1)
        print(f"{eps:>6.1f}  " + "".join(
            f"{np.sqrt(np.mean(v)):>{col_w}.4f}" if v else f"{'nan':>{col_w}}"
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
    args = p.parse_args()

    if args.multi:
        data_dir = f"{DATA_BASE}/window{args.window}_multi"
        Z_tr, Z_te, y_tr, y_te, z_min, z_scale, dates_te, ids_te = _load_data(data_dir)

        ckpt       = np.load(f"{CKPT_DIR}/regressor_w{args.window}_multi.npz")
        B          = ckpt["B"]
        W          = ckpt["W"]
        intercepts = ckpt["intercepts"]
        B_orig          = B / z_scale[:, None]
        intercepts_orig = intercepts - z_min @ B_orig

        predict_fn_denorm = lambda z: ((z * z_scale + z_min) @ B_orig).flatten() + intercepts_orig

        print(f"Z_te={Z_te.shape}  W={W.shape}")
    else:
        data_dir = f"{DATA_BASE}/window{args.window}_{args.target}"
        Z_tr, Z_te, y_tr, y_te, z_min, z_scale, dates_te, ids_te = _load_data(data_dir)

        ckpt      = np.load(f"{CKPT_DIR}/regressor_w{args.window}_{args.target}.npz")
        beta      = ckpt["beta"]
        intercept = float(ckpt["intercept"][0])
        W         = beta.reshape(1, -1)
        beta_orig      = beta / z_scale
        intercept_orig = intercept - float(z_min @ beta_orig)

        predict_fn_denorm = lambda z: float(((z * z_scale + z_min) @ beta_orig).flat[0]) + intercept_orig

        print(f"Z_te={Z_te.shape}  β_orig={beta_orig}")

    # Filter to private households only (ids ≥ n_public_hh).
    # When data was prepared with prepare_data.py (which already applies this split),
    # this is a no-op. For legacy data containing all households, this enforces the
    # public/private split required by the user-level LDP evaluation protocol.
    if ids_te.min() < args.n_public_hh:
        priv = ids_te >= args.n_public_hh
        Z_te, y_te, dates_te, ids_te = Z_te[priv], y_te[priv], dates_te[priv], ids_te[priv]
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

    evaluate_accuracy(mechs, Z_te, y_te, dates_te, ids_te,
                         predict_fn_denorm, eps_list,
                         args.n_dates, args.n_rounds, args.seed)


if __name__ == "__main__":
    main()
