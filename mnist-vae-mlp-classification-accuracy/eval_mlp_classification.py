"""
LDP evaluation in VAE latent space (MNIST) — sample accuracy.

Protocol: for each test sample, apply LDP mechanism to the latent vector,
decode, and classify. Sample accuracy = fraction of test samples where
clf(z_dec) == y_true. Averaged over the full test set (n_repeat = len(test_set)).

Prerequisites:
  python train_classifier.py --dim <dim>

Usage:
  python eval_mlp_classification.py --dim 16
  python eval_mlp_classification.py --dim 32
  python eval_mlp_classification.py --dim 64
  python eval_mlp_classification.py --dim 16 --seed 1
  python eval_mlp_classification.py --dim 16 --mechs "NoNoise" "PrivUnit2(Opt)+PA"
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mechanisms.mechanisms import build_mechs, compute_jacobian_row_space, DELTA
from train_classifier import load_classifier

NUM_CLASSES = 10
CKPT_DIR    = "./checkpoints"
LATENT_DIR  = "./data/MNIST/latent"


@torch.no_grad()
def _predict(clf, z: np.ndarray, device: str) -> int:
    return F.softmax(clf(torch.from_numpy(z).float().to(device)), dim=-1).argmax(1).cpu().item()


def sample_accuracy(clf, mech, Z_te: torch.Tensor, y_te: np.ndarray,
                    client_ids: np.ndarray, client_seeds: np.ndarray,
                    eps: float, device: str) -> float:
    Z_np = Z_te.numpy()
    correct, total = 0, 0
    for client_id, seed in zip(client_ids, client_seeds):
        rng   = np.random.default_rng(int(seed))
        z_enc = mech.encode(Z_np[client_id:client_id + 1], eps, rng=rng)
        z_dec = mech.decode(z_enc)
        correct += _predict(clf, z_dec, device) == int(y_te[client_id])
        total   += 1
    return correct / total if total else float("nan")


def evaluate(clf, mechs: dict, Z_te: torch.Tensor, y_te: np.ndarray,
             eps_list: list[float], n_repeat: int, seed: int = 0):
    device  = next(clf.parameters()).device
    n_data  = len(y_te)
    names   = list(mechs.keys())
    col_w   = max(18, max(len(n) for n in names) + 2)

    assign_rng  = np.random.default_rng([seed, 0])
    client_data = assign_rng.permutation(n_data)[:n_repeat]

    print(f"n_repeat={n_repeat}  D={Z_te.shape[1]}  delta={DELTA}")
    print(f"{'eps':>10}  " + "".join(f"{n:>{col_w}}" for n in names))
    print("-" * (12 + len(names) * col_w))

    for eps in eps_list:
        sums = {n: [] for n in names}
        for s in range(n_repeat):
            rng          = np.random.default_rng([seed, s])
            client_ids   = client_data[s:s + 1]
            client_seeds = rng.integers(0, 2**31, size=1)
            for name, mech in mechs.items():
                m = sample_accuracy(clf, mech, Z_te, y_te, client_ids, client_seeds,
                                    eps, device)
                if not np.isnan(m):
                    sums[name].append(m)

        print(f"{eps:>10}  " + "".join(
            f"{np.mean(v):>{col_w}.4f}" if v else f"{'nan':>{col_w}}"
            for v in sums.values()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim",   type=int, default=16)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--mechs", nargs="*", default=None,
                   help="subset of mechanism names to evaluate (default: all)")
    args = p.parse_args()

    eps_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0]

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    ld       = f"{LATENT_DIR}/{args.dim}"
    Z_tr     = torch.load(f"{ld}/Z_tr.pt", map_location="cpu", weights_only=True).float()
    Z_te     = torch.load(f"{ld}/Z_te.pt", map_location="cpu", weights_only=True).float()
    y_te     = torch.load(f"{ld}/y_te.pt", map_location="cpu", weights_only=True).numpy()
    clf      = load_classifier(args.dim, device)
    n_repeat = len(y_te)

    print(f"n_repeat={n_repeat}  D={args.dim}")

    B     = compute_jacobian_row_space(clf, Z_tr, n_samples=500)
    mechs = build_mechs(args.dim, B, Z_tr.numpy())
    mechs.pop("Task-Aware", None)
    if args.mechs:
        mechs = {k: v for k, v in mechs.items() if k in args.mechs}

    evaluate(clf, mechs, Z_te, y_te, eps_list, n_repeat, seed=args.seed)


if __name__ == "__main__":
    main()
