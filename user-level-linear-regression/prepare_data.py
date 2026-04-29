"""
Prepare London Smart Meters data for LDP linear regression.

Split policy:
  Households sorted by LCLid; first N_PUBLIC_HH are public (train), rest are private (test).
  Training set : public households  × dates < split_date
  Test set     : private households × dates ≥ split_date

Single-output mode (default):
  Z = past WINDOW days of one energy stat → y = next day of same stat
  Output: data/london/window{W}_{target}/

Multi-output mode (--multi):
  Z = past WINDOW days of energy_mean → Y = next day [mean, median, max, min, std]
  Output: data/london/window{W}_multi/
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

DATA_CSV      = None
OUT_BASE      = "./data/london"
WINDOW        = 16
SPLIT_DATE    = "2013-01-01"
N_PUBLIC_HH   = 3000
TARGETS       = ["energy_mean", "energy_median", "energy_max", "energy_min", "energy_std"]


def _add_run_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["LCLid", "day"]).reset_index(drop=True)
    same_hh   = df["LCLid"] == df["LCLid"].shift(1)
    consec    = df["day"].diff().dt.days == 1
    df["run"] = (~(same_hh & consec)).cumsum()
    return df


def make_windows(
    df: pd.DataFrame, window: int, feat_col: str, label_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows within each consecutive household run.

    Returns Z (N, window), Y (N,) or (N, C), dates (N,), hh_ids (N,).
    hh_ids is an integer code identifying the household for each row.
    Y is 1-D when len(label_cols) == 1.
    """
    df = _add_run_col(df)
    hh_codes = {hh: i for i, hh in enumerate(df["LCLid"].unique())}
    Zs, Ys, ds, ids = [], [], [], []

    for _, grp in df.groupby("run", sort=False):
        if len(grp) <= window:
            continue
        feat = grp[feat_col].values.astype(np.float32)
        wins = sliding_window_view(feat, window + 1)
        n    = len(wins)
        Zs.append(wins[:, :window])
        Ys.append(grp[label_cols].values.astype(np.float32)[window:])
        ds.append(grp["day"].values[window:])
        ids.append(np.full(n, hh_codes[grp["LCLid"].iloc[0]], dtype=np.int32))

    Z = np.vstack(Zs)
    Y = np.vstack(Ys)
    if Y.shape[1] == 1:
        Y = Y.ravel()
    return Z, Y, np.concatenate(ds), np.concatenate(ids)


def process(
    df: pd.DataFrame, window: int, split_date: str,
    feat_col: str, label_cols: list[str], out_dir: str,
    n_public_hh: int = N_PUBLIC_HH,
):
    tag = os.path.basename(out_dir)
    print(f"\n[{tag}] building windows ...")
    Z, Y, dates, hh_ids = make_windows(df, window, feat_col, label_cols)
    print(f"  Total windows: {len(Z):,}  Y shape: {np.shape(Y)}")

    n_total_hh = hh_ids.max() + 1
    print(f"  Public HHs: {n_public_hh:,}  Private HHs: {n_total_hh - n_public_hh:,}")

    # Training: public households (ids < n_public_hh) × dates before split_date
    # Test:    private households (ids ≥ n_public_hh) × dates on/after split_date
    tr = (hh_ids < n_public_hh) & (dates < np.datetime64(split_date))
    te = (hh_ids >= n_public_hh) & (dates >= np.datetime64(split_date))
    print(f"  Train: {tr.sum():,}  Test: {te.sum():,}")

    Z_tr, Z_te = Z[tr],  Z[te]
    y_tr, y_te = Y[tr],  Y[te]
    ids_te     = hh_ids[te]
    dates_te   = dates[te]

    z_min   = Z_tr.min(axis=0)
    z_scale = np.maximum(Z_tr.max(axis=0) - z_min, 1e-8)
    Z_tr_n  = (Z_tr - z_min) / z_scale
    Z_te_n  = (Z_te - z_min) / z_scale

    print(f"  Test households: {np.unique(ids_te).size:,}  "
          f"Test dates: {np.unique(dates_te).size:,}")

    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.from_numpy(Z_tr_n), os.path.join(out_dir, "Z_tr.pt"))
    torch.save(torch.from_numpy(y_tr),   os.path.join(out_dir, "y_tr.pt"))
    torch.save(torch.from_numpy(Z_te_n), os.path.join(out_dir, "Z_te.pt"))
    torch.save(torch.from_numpy(y_te),   os.path.join(out_dir, "y_te.pt"))
    np.save(os.path.join(out_dir, "z_min.npy"),   z_min)
    np.save(os.path.join(out_dir, "z_scale.npy"), z_scale)
    np.save(os.path.join(out_dir, "dates_te.npy"), dates_te)
    np.save(os.path.join(out_dir, "ids_te.npy"),   ids_te)
    print(f"  Saved → {out_dir}/  Z_tr={Z_tr_n.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",       type=int, default=WINDOW)
    parser.add_argument("--split_date",   type=str, default=SPLIT_DATE)
    parser.add_argument("--data_csv",     type=str, required=True,
                        help="path to daily_dataset.csv")
    parser.add_argument("--n_public_hh",  type=int, default=N_PUBLIC_HH,
                        help="number of households (sorted by LCLid) used as public training data")
    parser.add_argument("--target",       type=str, default=None, choices=TARGETS,
                        help="single target; omit to process all targets")
    parser.add_argument("--multi",        action="store_true",
                        help="multi-output mode: Z=energy_mean, Y=all 5 targets")
    args = parser.parse_args()

    print(f"Loading {args.data_csv} ...")
    df = pd.read_csv(args.data_csv, parse_dates=["day"],
                     usecols=["LCLid", "day", "energy_count"] + TARGETS)
    print(f"  Loaded {len(df):,} rows, {df['LCLid'].nunique():,} households")
    df = df[df["energy_count"] == 48].drop(columns="energy_count").copy()
    print(f"  After energy_count==48 filter: {len(df):,} rows")

    if args.multi:
        out_dir = os.path.join(OUT_BASE, f"window{args.window}_multi")
        process(df, args.window, args.split_date,
                feat_col="energy_mean", label_cols=TARGETS, out_dir=out_dir,
                n_public_hh=args.n_public_hh)
    else:
        for t in ([args.target] if args.target else TARGETS):
            out_dir = os.path.join(OUT_BASE, f"window{args.window}_{t}")
            process(df, args.window, args.split_date,
                    feat_col=t, label_cols=[t], out_dir=out_dir,
                    n_public_hh=args.n_public_hh)

    print("\nDone.")


if __name__ == "__main__":
    main()
