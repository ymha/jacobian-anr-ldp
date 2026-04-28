"""
LDP evaluation in VAE latent space (MNIST).

Protocol:
  K_max = len(test_set) // n_repeat clients are pre-assigned from the 10,000
  test samples; each client holds one sample per repeat (n_repeat samples total).

  For each (eps_feat, eps_label, K):
    Sample K clients (without replacement) from the pool of K_max.
    Each client i:
      z_enc_i   = mech.encode(z_i, eps_feat)   [eps_feat-LDP]
      y_tilde_i = RR(y_i, eps_label)            [eps_label-LDP]
      sends [z_raw_i | z_enc_i] to server
    Server: group by y_tilde
            per group k: aggregate [z_raw | z_enc] → split → decode z_enc_bar
    Accuracy: fraction of groups where predict(clf, z_dec) == predict(clf, z_bar)

Usage:
  python eval_logistic_classification.py --dim 16
  python eval_logistic_classification.py --dim 32 --n_repeat 20
  python eval_logistic_classification.py --dim 64 --n_repeat 20
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mechanisms"))
from mnist_classification_mechanisms import build_latent_mechs, compute_jacobian_row_space, DELTA
from train_classifier import load_classifier

NUM_CLASSES = 10
CKPT_DIR    = "./checkpoints"
LATENT_DIR  = "./data/MNIST/latent"


@torch.no_grad()
def _predict(clf, z: np.ndarray, device: str) -> int:
    return F.softmax(clf(torch.from_numpy(z).float().to(device)), dim=-1).argmax(1).cpu().item()


def group_accuracy(clf, mech, Z_te: torch.Tensor, y_te: np.ndarray,
                   client_ids: np.ndarray, client_seeds: np.ndarray,
                   eps_feat: float, eps_label: float,
                   device: str) -> float:
    """Encode K clients' single samples and compute group accuracy.

    Each client uses their seed to: (1) apply RR to their label, then (2) encode z.
    RR always consumes exactly 2 random values (1 float + 1 int) so that the
    encoding rng state is deterministic regardless of whether a flip occurred.

    client_seeds: shape (K,) — same across mechanisms for fair comparison.
    """
    Z_np   = Z_te.numpy()
    D      = Z_np.shape[1]
    p      = np.exp(eps_label) / (np.exp(eps_label) + NUM_CLASSES - 1)
    y_true = y_te[client_ids]
    rows    = []
    y_tilde = np.empty(len(client_ids), dtype=int)
    for i, (client_id, seed) in enumerate(zip(client_ids, client_seeds)):
        client_rng = np.random.default_rng(int(seed))
        u      = client_rng.random()
        offset = int(client_rng.integers(1, NUM_CLASSES))
        y_tilde[i] = int((y_true[i] + offset) % NUM_CLASSES) if u >= p else int(y_true[i])
        z_raw = Z_np[client_id:client_id + 1]
        z_enc = mech.encode(z_raw, eps_feat, rng=client_rng)
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


def evaluate(clf, mechs: dict, Z_te: torch.Tensor, y_te: np.ndarray,
             eps_feat_list: list[float], K_list: list[int],
             eps_label: float, n_repeat: int, seed: int = 0):
    device  = next(clf.parameters()).device
    n_data  = len(y_te)
    K_max   = n_data // n_repeat   # number of clients
    names   = list(mechs.keys())
    col_w   = 18

    # client_data[c, s] = data index that client c submits in repeat s
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
                # independent rng per (s, K) — sample K clients without replacement from K_max
                rng          = np.random.default_rng([seed, s, K])
                selected     = rng.choice(K_max, size=K, replace=False)  # client IDs
                client_ids   = client_data[selected, s]                  # data indices
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
    p.add_argument("--dim",      type=int,  default=16)
    p.add_argument("--n_repeat", type=int,  default=20)
    p.add_argument("--mechs",    nargs="*", default=None,
                   help="subset of mechanism names to evaluate (default: all)")
    args = p.parse_args()

    eps_feat_list  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0]
    eps_label_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0]
    K_list         = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ld     = f"{LATENT_DIR}/{args.dim}"
    Z_tr   = torch.load(f"{ld}/Z_tr.pt", map_location="cpu", weights_only=True).float()
    Z_te   = torch.load(f"{ld}/Z_te.pt", map_location="cpu", weights_only=True).float()
    y_te   = torch.load(f"{ld}/y_te.pt", map_location="cpu", weights_only=True).numpy()
    clf    = load_classifier(args.dim, device)

    K_max = len(y_te) // args.n_repeat
    print(f"Clients: {K_max}  (each with {args.n_repeat} samples)")

    B     = compute_jacobian_row_space(clf, Z_tr, n_samples=500)
    mechs = build_latent_mechs(args.dim, B, Z_tr.numpy())
    if args.mechs:
        mechs = {k: v for k, v in mechs.items() if k in args.mechs}

    for eps_label in eps_label_list:
        evaluate(clf, mechs, Z_te, y_te, eps_feat_list, K_list, eps_label, args.n_repeat)


if __name__ == "__main__":
    main()
