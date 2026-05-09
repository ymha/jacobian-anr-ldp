"""Aggregate per-seed regression result files into mean ± std tables.

Handles two table types written by eval_regression.py:
  1. Main MSE table  — header line: "eps=X.X  D=..."   rows keyed by K
  2. K=1 RMSE table — header line: "[K=1 Accuracy...]" rows keyed by eps

Usage:
  python aggregate_seeds.py results_seeds/
  python aggregate_seeds.py results_seeds/ --out aggregated.txt
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_result_file(path: Path) -> tuple[dict, dict]:
    """Parse one result file.

    Returns:
        mse_results:  {(eps, K): {mech_name: mse_value}}
        rmse_results: {eps:      {mech_name: rmse_value}}  (K=1 accuracy section)
    """
    mse_results:  dict = {}
    rmse_results: dict = {}
    lines = path.read_text().splitlines()

    section  = "mse"   # "mse" | "rmse"
    cur_eps  = None
    header   = None

    for line in lines:
        # Switch to K=1 RMSE section
        if "[K=1 Accuracy" in line:
            section = "rmse"
            cur_eps = None
            header  = None
            continue

        # Main MSE table header: "eps=0.5  D=16  n_rounds=..."
        if section == "mse":
            m = re.match(r"eps=([\d.]+)", line)
            if m:
                cur_eps = float(m.group(1))
                header  = None
                continue

        # Column header row (starts with "K" or "eps" then mechanism names)
        if re.match(r"\s*(K|eps)\s+\S", line):
            header = line.split()   # ['K', 'NoNoise', ...] or ['eps', ...]
            continue

        if header is None or re.match(r"^-+$", line.strip()):
            continue

        parts = line.split()
        if not parts:
            continue

        # Strip trailing "(n=NNN)" token if present
        if parts and parts[-1].startswith("(n="):
            parts = parts[:-1]

        if section == "mse":
            if cur_eps is None:
                continue
            try:
                K = int(parts[0])
            except (ValueError, IndexError):
                continue
            key = (cur_eps, K)
            mse_results[key] = {}
            for mech_name, val_str in zip(header[1:], parts[1:]):
                try:
                    mse_results[key][mech_name] = float(val_str)
                except ValueError:
                    pass

        else:  # rmse section — row key is eps value
            try:
                eps = float(parts[0])
            except (ValueError, IndexError):
                continue
            rmse_results[eps] = {}
            for mech_name, val_str in zip(header[1:], parts[1:]):
                try:
                    rmse_results[eps][mech_name] = float(val_str)
                except ValueError:
                    pass

    return mse_results, rmse_results


def _print_table(per_key: dict, all_mechs: list, n_seeds: int,
                 row_label: str, fmt: str) -> str:
    col_w = 26
    lines = []
    prev_group = object()  # sentinel — never equals None or any eps value
    header_sep = "-" * (8 + len(all_mechs) * col_w)

    for key in sorted(per_key.keys()):
        if isinstance(key, tuple):
            eps, K = key
            group = eps
            group_header = f"\neps={eps}  seeds={n_seeds}"
            row_prefix   = f"{K:>6}  "
        else:
            eps = key
            group = None
            row_prefix = f"{eps:>6.1f}  "
            group_header = None

        if group != prev_group:
            header_line = f"{row_label:>6}  " + "".join(f"{m:>{col_w}}" for m in all_mechs)
            if group_header:
                lines.append(group_header + "\n" + header_line + "\n" + header_sep)
            else:
                lines.append(f"seeds={n_seeds}\n" + header_line + "\n" + header_sep)
            prev_group = group

        row = row_prefix
        for mech in all_mechs:
            vals = per_key[key].get(mech, [])
            if vals:
                mean, std = np.mean(vals), np.std(vals, ddof=1)
                row += f"{fmt.format(mean, std)}".rjust(col_w)
            else:
                row += "nan".rjust(col_w)
        lines.append(row)

    return "\n".join(lines)


def aggregate(result_dir: Path, out_path: Path | None):
    files = sorted(result_dir.glob("result_seed*.txt"))
    if not files:
        sys.exit(f"No result_seed*.txt files found in {result_dir}")

    print(f"Found {len(files)} seed files: {[f.name for f in files]}")

    mse_per_key:  dict = defaultdict(lambda: defaultdict(list))
    rmse_per_key: dict = defaultdict(lambda: defaultdict(list))
    mse_mechs:  list = []
    rmse_mechs: list = []

    for f in files:
        mse, rmse = parse_result_file(f)
        for key, mechs in mse.items():
            for mech, val in mechs.items():
                mse_per_key[key][mech].append(val)
                if mech not in mse_mechs:
                    mse_mechs.append(mech)
        for key, mechs in rmse.items():
            for mech, val in mechs.items():
                rmse_per_key[key][mech].append(val)
                if mech not in rmse_mechs:
                    rmse_mechs.append(mech)

    parts = []

    if mse_per_key:
        parts.append("=== MSE (normalized, LDP vs noiseless) ===")
        parts.append(_print_table(mse_per_key, mse_mechs, len(files),
                                  row_label="K", fmt="{:.4f} ± {:.4f}"))

    if rmse_per_key:
        parts.append("\n=== K=1 RMSE vs real labels (denormalized) ===")
        parts.append(_print_table(rmse_per_key, rmse_mechs, len(files),
                                  row_label="eps", fmt="{:.4f} ± {:.4f}"))

    output = "\n".join(parts) + "\n"
    print(output)

    if out_path:
        out_path.write_text(output)
        print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_dir", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="output file (default: print only)")
    args = p.parse_args()
    aggregate(args.result_dir, args.out)


if __name__ == "__main__":
    main()
