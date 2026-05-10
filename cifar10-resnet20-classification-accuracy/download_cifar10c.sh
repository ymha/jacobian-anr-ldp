#!/usr/bin/env bash
# Download and extract CIFAR-10-C dataset.
# Usage: bash download_cifar10c.sh

set -euo pipefail

OUT_DIR="data/CIFAR10-C"
mkdir -p "$OUT_DIR"

echo "Downloading CIFAR-10-C (~2.8 GB)..."
wget -c --show-progress \
    https://zenodo.org/record/2535967/files/CIFAR-10-C.tar \
    -O "$OUT_DIR/CIFAR-10-C.tar"

echo "Extracting..."
tar -xf "$OUT_DIR/CIFAR-10-C.tar" -C "$OUT_DIR"

echo "Done. Files in $OUT_DIR/CIFAR-10-C/:"
ls "$OUT_DIR/CIFAR-10-C/"
