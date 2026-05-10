"""Aggregate per-seed result files into mean ± std tables.

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


def parse_result_file(path: Path) -> dict:
    """Parse one result file. Returns {eps_feat: {mech_name: accuracy}}."""
    results = {}
    lines = path.read_text().splitlines()

    header = None

    for line in lines:
        stripped = line.strip()

        # Header row starts with "eps_feat"
        if re.match(r"eps_feat\b", stripped):
            # Handle merged column names, e.g. "PrivUnitG(MC)+PAPrivUnitG(Paper)+PA"
            fixed = stripped.replace("PrivUnitG(MC)+PAPrivUnitG(Paper)+PA",
                                     "PrivUnitG(MC)+PA PrivUnitG(Paper)+PA")
            header = fixed.split()  # ['eps_feat', 'NoNoise', ...]
            continue

        if header is None or re.match(r"^-+$", stripped):
            continue

        parts = stripped.split()
        if not parts:
            continue

        # Data rows: first token is a float (eps value)
        try:
            eps = float(parts[0])
        except ValueError:
            continue

        results[eps] = {}
        for mech_name, val_str in zip(header[1:], parts[1:]):
            try:
                results[eps][mech_name] = float(val_str)
            except ValueError:
                pass

    return results


def aggregate(result_dir: Path, out_path: Path | None):
    files = sorted(result_dir.glob("result_seed*.txt"))
    if not files:
        sys.exit(f"No result_seed*.txt files found in {result_dir}")

    print(f"Found {len(files)} seed files.")

    # per_eps[eps][mech] = [acc_seed0, acc_seed1, ...]
    per_eps: dict = defaultdict(lambda: defaultdict(list))
    all_mechs: list = []

    for f in files:
        parsed = parse_result_file(f)
        for eps, mechs in parsed.items():
            for mech, acc in mechs.items():
                per_eps[eps][mech].append(acc)
                if mech not in all_mechs:
                    all_mechs.append(mech)

    col_w = 22  # wide enough for "0.1234 ± 0.0056"
    lines_out = []

    header_line = (
        f"\nseeds={len(files)}\n"
        f"{'eps_feat':>10}  " + "".join(f"{m:>{col_w}}" for m in all_mechs) + "\n"
        + "-" * (12 + len(all_mechs) * col_w)
    )
    lines_out.append(header_line)

    for eps in sorted(per_eps.keys()):
        row = f"{eps:>10.1f}  "
        for mech in all_mechs:
            vals = per_eps[eps].get(mech, [])
            if vals:
                row += f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}".rjust(col_w)
            else:
                row += "nan".rjust(col_w)
        lines_out.append(row)

    output = "\n".join(lines_out) + "\n"
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

