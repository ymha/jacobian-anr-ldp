"""
LDP evaluation in ResNet-20 feature space (CIFAR-10).

For each test sample: apply LDP mechanism to 64-dim feature vector, decode, classify.
Accuracy is averaged over the full test set (n_repeat = len(test_set)).

Prerequisites:
  python train_resnet20.py

Usage:
  python eval_cifar_classification.py
  python eval_cifar_classification.py --mechs "NoNoise" "PrivUnit2(Opt)+PA"
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mechanisms.mechanisms import build_mechs, build_mechs_ablation, compute_jacobian_row_space, DELTA
from model import FeatureClassifier, MLPClassifier, ACTIVATIONS

NUM_CLASSES = 10
LATENT_DIM  = 64
CKPT_DIR    = "./checkpoints"
LATENT_DIR  = "./data/CIFAR10/latent"


def load_classifier(device: str, mlp: bool = False, activation: str = "relu"):
    if mlp:
        ckpt = torch.load(f"{CKPT_DIR}/mlp_clf_{activation}.pt", map_location=device, weights_only=True)
        clf  = MLPClassifier(activation=activation).to(device)
    else:
        ckpt = torch.load(f"{CKPT_DIR}/feature_clf.pt", map_location=device, weights_only=True)
        clf  = FeatureClassifier(nn.Linear(LATENT_DIM, NUM_CLASSES)).to(device)
    clf.load_state_dict(ckpt["state_dict"])
    for p in clf.parameters():
        p.requires_grad_(False)
    clf.eval()
    return clf


@torch.no_grad()
def _predict(clf, z: np.ndarray, device: str) -> int:
    return F.softmax(clf(torch.from_numpy(z).float().to(device)), dim=-1).argmax(1).cpu().item()


def sample_accuracy(clf, mech, Z_te: torch.Tensor, y_te: np.ndarray,
                    client_ids: np.ndarray, client_seeds: np.ndarray,
                    eps: float, device: str) -> float:
    Z_np = Z_te.numpy()
    correct, total = 0, 0
    for client_id, seed in zip(client_ids, client_seeds):
        rng     = np.random.default_rng(int(seed))
        z_enc   = mech.encode(Z_np[client_id:client_id + 1], eps, rng=rng)
        z_dec   = mech.decode(z_enc)
        correct += _predict(clf, z_dec, device) == int(y_te[client_id])
        total   += 1
    return correct / total if total else float("nan")


def evaluate(clf, mechs: dict, Z_te: torch.Tensor, y_te: np.ndarray,
             eps_list: list, n_repeat: int, seed: int = 0):
    device  = next(clf.parameters()).device
    n_data  = len(y_te)
    names   = list(mechs.keys())
    col_w   = max(18, max(len(n) for n in names) + 2)

    assign_rng  = np.random.default_rng([seed, 0])
    client_data = assign_rng.permutation(n_data)[:n_repeat]

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
    p.add_argument("--seed",       type=int,  default=0)
    p.add_argument("--mlp",        action="store_true",
                   help="use MLP classifier head instead of linear FC")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS),
                   help="MLP activation function (default: relu)")
    p.add_argument("--ablation",   action="store_true",
                   help="use ablation mechanism registry instead of main registry")
    p.add_argument("--mechs",      nargs="*", default=None,
                   help="subset of mechanism names to evaluate (default: all)")
    p.add_argument("--te_dir",     default=None,
                   help="directory containing Z_te.pt and y_te.pt "
                        "(default: ./data/CIFAR10/latent; use for CIFAR-10-C corrupted sets)")
    args = p.parse_args()

    eps_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0]

    te_dir   = args.te_dir or LATENT_DIR
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    Z_tr     = torch.load(f"{LATENT_DIR}/Z_tr.pt", map_location="cpu", weights_only=True).float()
    Z_te     = torch.load(f"{te_dir}/Z_te.pt",     map_location="cpu", weights_only=True).float()
    y_te     = torch.load(f"{te_dir}/y_te.pt",     map_location="cpu", weights_only=True).numpy()
    clf      = load_classifier(device, mlp=args.mlp, activation=args.activation)
    n_repeat = len(y_te)

    clf_name = f"MLP({args.activation})" if args.mlp else "linear"
    print(f"n_repeat={n_repeat}  D={LATENT_DIM}  clf={clf_name}  ablation={args.ablation}")

    B     = compute_jacobian_row_space(clf, Z_tr, n_samples=500)
    mechs = build_mechs_ablation(LATENT_DIM, B, Z_tr.numpy()) if args.ablation \
            else build_mechs(LATENT_DIM, B, Z_tr.numpy())
    if not args.ablation and args.mlp:
        mechs.pop("Task-Aware", None)
    if args.mechs:
        mechs = {k: v for k, v in mechs.items() if k in args.mechs}

    evaluate(clf, mechs, Z_te, y_te, eps_list, n_repeat, seed=args.seed)


if __name__ == "__main__":
    main()
