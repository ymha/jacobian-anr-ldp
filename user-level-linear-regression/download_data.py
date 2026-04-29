"""
Download London Smart Meters dataset from Kaggle and prepare data.

Usage:
  python download_data.py                    # download + prepare all targets
  python download_data.py --skip-download    # prepare only (csv already present)
  python download_data.py --window 7         # different window size
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_DATASET = "jeanmidev/smart-meters-in-london"
CSV_NAME       = "daily_dataset.csv"
RAW_DIR        = Path("./raw_data")


def download_kaggle(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{KAGGLE_DATASET.split('/')[-1]}.zip"

    print(f"Downloading {KAGGLE_DATASET} from Kaggle ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest)],
            check=True,
        )
    except FileNotFoundError:
        sys.exit(
            "Error: 'kaggle' command not found.\n"
            "  pip install kaggle\n"
            "  Then set up ~/.kaggle/kaggle.json with your API key:\n"
            "  https://www.kaggle.com/docs/api#authentication"
        )
    except subprocess.CalledProcessError:
        sys.exit(
            "Kaggle download failed. Make sure:\n"
            "  1. ~/.kaggle/kaggle.json exists with valid credentials\n"
            "  2. You have accepted the dataset terms on Kaggle"
        )

    zip_files = list(dest.glob("*.zip"))
    if not zip_files:
        sys.exit(f"No zip file found in {dest} after download.")

    print(f"Extracting {zip_files[0]} ...")
    with zipfile.ZipFile(zip_files[0]) as zf:
        zf.extractall(dest)
    print("Extraction complete.")


def find_csv(search_dir: Path) -> Path:
    matches = list(search_dir.rglob(CSV_NAME))
    if not matches:
        sys.exit(
            f"Could not find {CSV_NAME} under {search_dir}.\n"
            "Try running without --skip-download."
        )
    if len(matches) > 1:
        print(f"Multiple matches found; using {matches[0]}")
    return matches[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true",
                   help="skip Kaggle download (csv already in raw_data/)")
    p.add_argument("--window",        type=int, default=16)
    p.add_argument("--split-date",    type=str, default="2013-01-01")
    p.add_argument("--n-public-hh",   type=int, default=3000)
    p.add_argument("--target",        type=str, default=None,
                   help="single target; omit to prepare all targets + multi")
    args = p.parse_args()

    if not args.skip_download:
        download_kaggle(RAW_DIR)

    csv_path = find_csv(RAW_DIR)
    print(f"Using CSV: {csv_path}")

    base_cmd = [
        sys.executable, "prepare_data.py",
        "--data_csv",    str(csv_path),
        "--window",      str(args.window),
        "--split_date",  args.split_date,
        "--n_public_hh", str(args.n_public_hh),
    ]

    targets = [args.target] if args.target else None

    if targets:
        for t in targets:
            print(f"\nPreparing target: {t}")
            subprocess.run(base_cmd + ["--target", t], check=True)
    else:
        print("\nPreparing all single-output targets ...")
        subprocess.run(base_cmd, check=True)
        print("\nPreparing multi-output ...")
        subprocess.run(base_cmd + ["--multi"], check=True)

    print("\nData preparation complete.")
    print("Next steps:")
    print(f"  python train_regressor.py --window {args.window}")
    print(f"  python train_regressor.py --window {args.window} --multi")
    print(f"  python eval_regression.py --window {args.window} --target energy_mean")


if __name__ == "__main__":
    main()
