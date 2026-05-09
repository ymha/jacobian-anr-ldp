"""
Train linear regression on London power windowed data.

Single-output mode (default / --target):
  fits OLS per target, saves β ∈ ℝ^D and intercept

Multi-output mode (--multi):
  fits OLS simultaneously for all 5 targets
  saves B ∈ ℝ^{D×5}, intercepts ∈ ℝ^5, and W = B.T ∈ ℝ^{5×D}

Usage:
  python train_regressor.py --window 16            # all single-output targets
  python train_regressor.py --window 16 --multi    # multi-output (W is 5x16)
"""

import os
import argparse

import numpy as np
import torch

DATA_BASE = "./data/london"
CKPT_DIR  = "./checkpoints"
TARGETS   = ["energy_mean", "energy_median", "energy_max", "energy_min", "energy_std"]


def _load_arrays(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _pt(name):
        return torch.load(os.path.join(data_dir, name), weights_only=True).numpy().astype(np.float64)
    return _pt("Z_tr.pt"), _pt("y_tr.pt"), _pt("Z_te.pt"), _pt("y_te.pt")


def _ols(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Z_aug, coeffs) from OLS on [Z | 1] · coeffs ≈ y."""
    Z_aug = np.hstack([Z, np.ones((len(Z), 1))])
    coeffs, _, _, _ = np.linalg.lstsq(Z_aug, y, rcond=None)
    return Z_aug, coeffs


def train_single(window: int, target: str):
    data_dir = os.path.join(DATA_BASE, f"window{window}_{target}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    Z_tr, y_tr, Z_te, y_te = _load_arrays(data_dir)
    Z_tr_aug, coeffs = _ols(Z_tr, y_tr)
    Z_te_aug = np.hstack([Z_te, np.ones((len(Z_te), 1))])
    beta, intercept = coeffs[:-1], coeffs[-1]

    print(f"[{target}]  Train MSE: {np.mean((Z_tr_aug @ coeffs - y_tr)**2):.6f}"
          f"  Test MSE: {np.mean((Z_te_aug @ coeffs - y_te)**2):.6f}")

    ckpt_path = os.path.join(CKPT_DIR, f"regressor_w{window}_{target}.npz")
    np.savez(ckpt_path, beta=beta, intercept=np.array([intercept]))
    print(f"  Saved → {ckpt_path}")


def train_multi(window: int):
    """Multi-output OLS: Z ∈ ℝ^D → Y ∈ ℝ^5.  Saves B (D×5) and W = B.T (5×D)."""
    data_dir = os.path.join(DATA_BASE, f"window{window}_multi")
    os.makedirs(CKPT_DIR, exist_ok=True)

    Z_tr, Y_tr, Z_te, Y_te = _load_arrays(data_dir)
    print(f"[multi]  Z_tr={Z_tr.shape}  Y_tr={Y_tr.shape}")

    Z_tr_aug, Coeffs = _ols(Z_tr, Y_tr)
    Z_te_aug  = np.hstack([Z_te, np.ones((len(Z_te), 1))])
    B, intercepts = Coeffs[:-1], Coeffs[-1]
    W = B.T

    train_mse = np.mean((Z_tr_aug @ Coeffs - Y_tr) ** 2, axis=0)
    test_mse  = np.mean((Z_te_aug @ Coeffs - Y_te) ** 2, axis=0)

    print("  Train MSE per target: " + "  ".join(f"{t}={v:.6f}" for t, v in zip(TARGETS, train_mse)))
    print("  Test  MSE per target: " + "  ".join(f"{t}={v:.6f}" for t, v in zip(TARGETS, test_mse)))
    print(f"  W shape: {W.shape}  (= B.T, task matrix for ANR-SV)")

    ckpt_path = os.path.join(CKPT_DIR, f"regressor_w{window}_multi.npz")
    np.savez(ckpt_path, B=B, W=W, intercepts=intercepts)
    print(f"  Saved → {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--target", type=str, default=None, choices=TARGETS)
    parser.add_argument("--multi",  action="store_true")
    args = parser.parse_args()

    if args.multi:
        train_multi(args.window)
    else:
        for t in ([args.target] if args.target else TARGETS):
            train_single(args.window, t)


if __name__ == "__main__":
    main()
