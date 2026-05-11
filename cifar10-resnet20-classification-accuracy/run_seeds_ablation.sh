#!/usr/bin/env bash
# Run eval_cifar_classification.py with multiple seeds and save per-seed outputs.
#
# Usage:
#   bash run_seeds.sh                    # seeds 0..4, MLP(relu), n_repeat=10000
#   bash run_seeds.sh --seeds "0 1 2"   # custom seeds
#   bash run_seeds.sh --out_dir my_results

set -euo pipefail

SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
OUT_DIR="results_seeds_ablation"
EXTRA_ARGS=()

# Parse our own flags; collect everything else as EXTRA_ARGS for the python script.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)   SEEDS="$2";   shift 2 ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$OUT_DIR"

for seed in $SEEDS; do
    out="$OUT_DIR/result_seed${seed}.txt"
    echo "=== seed=${seed} → ${out} ==="
    python eval_cifar_classification.py --seed "$seed" --mlp --activation relu --ablation --te_dir data/CIFAR10-C/latent/brightness_s5 "${EXTRA_ARGS[@]}" | tee "$out"
done

echo ""
echo "All seeds done. Results in: $OUT_DIR/"
echo "Run:  python aggregate_seeds.py $OUT_DIR/"

