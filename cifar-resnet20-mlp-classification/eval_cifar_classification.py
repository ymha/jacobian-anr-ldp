"""
LDP evaluation in ResNet-20 feature space (CIFAR-10).

The 'latent' representation is the 64-dim penultimate-layer feature vector from
ResNet-20 (after avg pool, before fc). The classifier 'clf' is the fc layer alone
(a linear map ℝ^64 → ℝ^10), so its Jacobian row space has rank ≤ 10 — the
task-relevant subspace that ANR mechanisms exploit.

Protocol: identical to eval_mlp_classification.py.
  K_max = len(test_set) // n_repeat clients pre-assigned from 10,000 test samples.
  For each (eps_feat, eps_label, K):
    Sample K clients; each applies RR to label + LDP mech to feature vector.
    Server groups by noisy label, aggregates, decodes.
    group_accuracy = fraction of groups where clf(z_dec) == clf(z_bar).

Prerequisites:
  python train_resnet20.py [--weights path/to/resnet20.pth]

Usage:
  python eval_cifar_classification.py
  python eval_cifar_classification.py --n_repeat 20
  python eval_cifar_classification.py --mechs "NoNoise" "ANR-SV+PrivUnit2" "PrivUnit2"
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mechanisms.cifar10_classification_mechanisms import build_latent_mechs, compute_jacobian_row_space, DELTA
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


def global_accuracy(clf, mech, Z_te: torch.Tensor,
                    client_ids: np.ndarray, client_seeds: np.ndarray,
                    eps_feat: float, device: str) -> float:
    """Label-free: encode K clients and aggregate into one global mean."""
    Z_np = Z_te.numpy()
    D    = Z_np.shape[1]
    rows = []

    for client_id, seed in zip(client_ids, client_seeds):
        rng   = np.random.default_rng(int(seed))
        z_raw = Z_np[client_id:client_id + 1]
        z_enc = mech.encode(z_raw, eps_feat, rng=rng)
        rows.append(np.concatenate([z_raw, z_enc], axis=1))

    combined  = np.vstack(rows)
    agg       = combined.mean(axis=0, keepdims=True)
    z_bar     = agg[:, :D]
    z_enc_bar = agg[:, D:]
    z_dec     = mech.decode(z_enc_bar)
    return float(_predict(clf, z_dec, device) == _predict(clf, z_bar, device))


def group_accuracy(clf, mech, Z_te: torch.Tensor, y_te: np.ndarray,
                   client_ids: np.ndarray, client_seeds: np.ndarray,
                   eps_feat: float, eps_label: float, device: str) -> float:
    """Encode K clients' samples and compute group accuracy.

    Each client uses their seed to: (1) apply RR to label, then (2) encode feature vector.
    RR consumes exactly 2 random values (1 float + 1 int) so encoding RNG state is
    deterministic regardless of whether a flip occurred.
    """
    Z_np    = Z_te.numpy()
    D       = Z_np.shape[1]
    p       = np.exp(eps_label) / (np.exp(eps_label) + NUM_CLASSES - 1)
    y_true  = y_te[client_ids]
    y_tilde = np.empty(len(client_ids), dtype=int)
    rows    = []

    for i, (client_id, seed) in enumerate(zip(client_ids, client_seeds)):
        rng    = np.random.default_rng(int(seed))
        u      = rng.random()
        offset = int(rng.integers(1, NUM_CLASSES))
        y_tilde[i] = int((y_true[i] + offset) % NUM_CLASSES) if u >= p else int(y_true[i])
        z_raw = Z_np[client_id:client_id + 1]
        z_enc = mech.encode(z_raw, eps_feat, rng=rng)
        rows.append(np.concatenate([z_raw, z_enc], axis=1))

    combined = np.vstack(rows)  # (K, 2D)
    correct, total = 0, 0
    for k in range(NUM_CLASSES):
        idx = np.where(y_tilde == k)[0]
        if not len(idx):
            continue
        agg       = combined[idx].mean(axis=0, keepdims=True)
        z_bar     = agg[:, :D]
        z_enc_bar = agg[:, D:]
        z_dec     = mech.decode(z_enc_bar)
        correct  += _predict(clf, z_dec, device) == _predict(clf, z_bar, device)
        total    += 1
    return correct / total if total else float("nan")


def evaluate_label_free(clf, mechs: dict, Z_te: torch.Tensor, y_te: np.ndarray,
                        eps_feat_list: list, K_list: list,
                        n_repeat: int, seed: int = 0):
    device  = next(clf.parameters()).device
    n_data  = len(y_te)
    K_max   = n_data // n_repeat
    names   = list(mechs.keys())
    col_w   = 18

    assign_rng  = np.random.default_rng([seed, 0])
    all_idx     = assign_rng.permutation(n_data)[:K_max * n_repeat]
    client_data = all_idx.reshape(K_max, n_repeat)

    for eps_feat in eps_feat_list:
        print(f"\n[label-free]  eps_feat={eps_feat}  delta={DELTA}  "
              f"D={Z_te.shape[1]}  K_max={K_max}  repeats={n_repeat}")
        print(f"{'K':>6}  " + "".join(f"{n:>{col_w}}" for n in names))
        print("-" * (8 + len(names) * col_w))

        for K in K_list:
            if K > K_max:
                continue
            sums = {n: [] for n in names}
            for s in range(n_repeat):
                rng          = np.random.default_rng([seed, s, K])
                selected     = rng.choice(K_max, size=K, replace=False)
                client_ids   = client_data[selected, s]
                client_seeds = rng.integers(0, 2**31, size=K)
                for name, mech in mechs.items():
                    sums[name].append(
                        global_accuracy(clf, mech, Z_te, client_ids, client_seeds,
                                        eps_feat, device))

            print(f"{K:>6}  " + "".join(
                f"{np.mean(v):>{col_w}.4f}" for v in sums.values()))


def evaluate(clf, mechs: dict, Z_te: torch.Tensor, y_te: np.ndarray,
             eps_feat_list: list, K_list: list,
             eps_label: float, n_repeat: int, seed: int = 0):
    device  = next(clf.parameters()).device
    n_data  = len(y_te)
    K_max   = n_data // n_repeat
    names   = list(mechs.keys())
    col_w   = 18

    assign_rng  = np.random.default_rng([seed, 0])
    all_idx     = assign_rng.permutation(n_data)[:K_max * n_repeat]
    client_data = all_idx.reshape(K_max, n_repeat)  # (K_max, n_repeat)

    for eps_feat in eps_feat_list:
        print(f"\neps_feat={eps_feat}  eps_label={eps_label}  delta={DELTA}  "
              f"D={Z_te.shape[1]}  K_max={K_max}  repeats={n_repeat}")
        print(f"{'K':>6}  " + "".join(f"{n:>{col_w}}" for n in names))
        print("-" * (8 + len(names) * col_w))

        for K in K_list:
            if K > K_max:
                continue
            sums = {n: [] for n in names}
            for s in range(n_repeat):
                rng          = np.random.default_rng([seed, s, K])
                selected     = rng.choice(K_max, size=K, replace=False)
                client_ids   = client_data[selected, s]
                client_seeds = rng.integers(0, 2**31, size=K)
                for name, mech in mechs.items():
                    m = group_accuracy(clf, mech, Z_te, y_te, client_ids, client_seeds,
                                       eps_feat, eps_label, device)
                    if not np.isnan(m):
                        sums[name].append(m)

            print(f"{K:>6}  " + "".join(
                f"{np.mean(v):>{col_w}.4f}" if v else f"{'nan':>{col_w}}"
                for v in sums.values()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_repeat",   type=int,  default=20)
    p.add_argument("--mlp",        action="store_true",
                   help="use MLP classifier head instead of linear FC")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS),
                   help="MLP activation function (default: relu)")
    p.add_argument("--mechs",      nargs="*", default=None,
                   help="subset of mechanism names to evaluate (default: all)")
    p.add_argument("--setting",    default="label_anchored",
                   choices=["label_anchored", "label_free", "both"])
    args = p.parse_args()

    eps_feat_list  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0]
    eps_label_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0]
    K_list         = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Z_tr   = torch.load(f"{LATENT_DIR}/Z_tr.pt", map_location="cpu", weights_only=True).float()
    Z_te   = torch.load(f"{LATENT_DIR}/Z_te.pt", map_location="cpu", weights_only=True).float()
    y_te   = torch.load(f"{LATENT_DIR}/y_te.pt", map_location="cpu", weights_only=True).numpy()
    clf    = load_classifier(device, mlp=args.mlp, activation=args.activation)

    K_max = len(y_te) // args.n_repeat
    clf_name = f"MLP({args.activation})" if args.mlp else "linear"
    print(f"Clients: {K_max}  (each with {args.n_repeat} samples)  D={LATENT_DIM}  clf={clf_name}")

    B     = compute_jacobian_row_space(clf, Z_tr, n_samples=500)
    mechs = build_latent_mechs(LATENT_DIM, B, Z_tr.numpy())
    if args.mlp:
        mechs.pop("TASK(Cheng22)", None)
    if args.mechs:
        mechs = {k: v for k, v in mechs.items() if k in args.mechs}

    if args.setting in ("label_free", "both"):
        evaluate_label_free(clf, mechs, Z_te, y_te, eps_feat_list, K_list, args.n_repeat)

    if args.setting in ("label_anchored", "both"):
        for eps_label in eps_label_list:
            evaluate(clf, mechs, Z_te, y_te, eps_feat_list, K_list, eps_label, args.n_repeat)


if __name__ == "__main__":
    main()
