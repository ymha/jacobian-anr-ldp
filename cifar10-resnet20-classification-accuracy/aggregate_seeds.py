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
    """Parse one result file. Returns nested dict:
    {(eps_feat, eps_label, K): {mech_name: accuracy}}
    """
    results = {}
    lines = path.read_text().splitlines()

    eps_feat = eps_label = None
    header = None

    for line in lines:
        m = re.match(r"eps_feat=([\d.]+).*eps_label=([\d.]+)", line)
        if m:
            eps_feat  = float(m.group(1))
            eps_label = float(m.group(2))
            header    = None
            continue

        if eps_feat is None:
            continue

        # Header row: "     K    NoNoise    PrivUnit2(Opt)+PA  ..."
        if re.match(r"\s*K\s+\S", line):
            header = line.split()  # ['K', 'NoNoise', ...]
            continue

        if header is None or re.match(r"^-+$", line.strip()):
            continue

        parts = line.split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue

        try:
            K = int(parts[0])
        except ValueError:
            continue

        key = (eps_feat, eps_label, K)
        results[key] = {}
        for mech_name, val_str in zip(header[1:], parts[1:]):
            try:
                results[key][mech_name] = float(val_str)
            except ValueError:
                pass

    return results


def aggregate(result_dir: Path, out_path: Path | None):
    files = sorted(result_dir.glob("result_seed*.txt"))
    if not files:
        sys.exit(f"No result_seed*.txt files found in {result_dir}")

    print(f"Found {len(files)} seed files: {[f.name for f in files]}")

    # per_key[key][mech] = [acc_seed0, acc_seed1, ...]
    per_key: dict = defaultdict(lambda: defaultdict(list))
    all_mechs: list = []

    for f in files:
        parsed = parse_result_file(f)
        for key, mechs in parsed.items():
            for mech, acc in mechs.items():
                per_key[key][mech].append(acc)
                if mech not in all_mechs:
                    all_mechs.append(mech)

    col_w   = 26  # wide enough for "0.9000 ± 0.0123"
    lines_out = []

    prev_eps = None
    for key in sorted(per_key.keys()):
        eps_feat, eps_label, K = key
        if (eps_feat, eps_label) != prev_eps:
            header_line = (
                f"\neps_feat={eps_feat}  eps_label={eps_label}  "
                f"seeds={len(files)}\n"
                f"{'K':>6}  " + "".join(f"{m:>{col_w}}" for m in all_mechs) + "\n"
                + "-" * (8 + len(all_mechs) * col_w)
            )
            lines_out.append(header_line)
            prev_eps = (eps_feat, eps_label)

        row = f"{K:>6}  "
        for mech in all_mechs:
            vals = per_key[key].get(mech, [])
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
