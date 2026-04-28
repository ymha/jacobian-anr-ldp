"""
Train VAE encoder and MLP classifier jointly end-to-end on MNIST.

Initializes encoder from a pretrained VAE checkpoint, then fine-tunes encoder
and MLP together with cross-entropy loss. After training, re-extracts latent
vectors (Z_tr, Z_val, Z_te) from the updated encoder and saves all checkpoints.

Usage:
  python train_classifier.py --dim 16
  python train_classifier.py --dim 32
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from train_vae import Encoder, HIDDEN_DIM, load_mnist

CKPT_DIR    = "./checkpoints"
LATENT_DIR  = "./data/MNIST/latent"
NUM_CLASSES = 10
MLP_HIDDEN1 = 10   # first hidden layer; Jacobian rank bounded by min(h1, h2, NUM_CLASSES)=10
MLP_HIDDEN2 = 32   # second hidden layer


class LatentMLP(nn.Module):
    def __init__(self, dim: int, hidden1: int = MLP_HIDDEN1, hidden2: int = MLP_HIDDEN2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_classifier(dim: int, device: str) -> nn.Module:
    ckpt  = torch.load(os.path.join(CKPT_DIR, f"latent_mlp_d{dim}.pt"),
                       map_location=device, weights_only=True)
    model = LatentMLP(dim,
                      hidden1=ckpt.get("hidden1", MLP_HIDDEN1),
                      hidden2=ckpt.get("hidden2", MLP_HIDDEN2)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def _encode(encoder: nn.Module, X: torch.Tensor, device: str) -> torch.Tensor:
    return encoder(X.flatten(1).to(device))[0].cpu()


@torch.no_grad()
def _accuracy(encoder: nn.Module, clf: nn.Module,
              X: torch.Tensor, y: torch.Tensor, device: str) -> float:
    mu = encoder(X.flatten(1).to(device))[0]
    return accuracy_score(y.numpy(), clf(mu).argmax(1).cpu().numpy())


def train_joint(dim: int, epochs: int = 100, lr: float = 1e-3,
                batch_size: int = 256, seed: int = 42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    X_tr, y_tr, X_te, y_te = load_mnist()
    n_val   = int(len(X_tr) * 0.2)
    n_train = len(X_tr) - n_val
    tr_ds, val_ds = random_split(
        TensorDataset(X_tr, y_tr), [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))
    X_tr_s, y_tr_s = tr_ds[:]
    X_val_s, y_val_s = val_ds[:]

    enc_ckpt = torch.load(os.path.join(CKPT_DIR, f"encoder_d{dim}.pt"),
                          map_location=device, weights_only=True)
    encoder = Encoder(784, HIDDEN_DIM, dim).to(device)
    encoder.load_state_dict(enc_ckpt["state_dict"])

    clf       = LatentMLP(dim).to(device)
    opt       = optim.Adam(list(encoder.parameters()) + list(clf.parameters()),
                           lr=lr, weight_decay=1e-4)
    sch       = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(TensorDataset(X_tr_s, y_tr_s), batch_size, shuffle=True)

    print(f"\n{'='*55}")
    print(f"Joint D={dim}  train={n_train}  val={n_val}  test={len(X_te)}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        encoder.train(); clf.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            mu, _ = encoder(Xb.flatten(1))
            opt.zero_grad()
            criterion(clf(mu), yb).backward()
            opt.step()
        sch.step()

        if epoch % 20 == 0 or epoch == 1:
            encoder.eval(); clf.eval()
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"val_acc={_accuracy(encoder, clf, X_val_s, y_val_s, device):.4f}")

    encoder.eval(); clf.eval()
    print(f"Final test acc: {_accuracy(encoder, clf, X_te, y_te, device):.4f}")

    ld = os.path.join(LATENT_DIR, str(dim))
    os.makedirs(ld, exist_ok=True)
    for split, X, y in [("tr", X_tr_s, y_tr_s), ("val", X_val_s, y_val_s), ("te", X_te, y_te)]:
        torch.save(_encode(encoder, X, device), os.path.join(ld, f"Z_{split}.pt"))
        torch.save(y,                           os.path.join(ld, f"y_{split}.pt"))
    print(f"Latent vectors re-extracted → {ld}/")

    for p in list(encoder.parameters()) + list(clf.parameters()):
        p.requires_grad_(False)

    torch.save({"state_dict": encoder.state_dict(), "latent_dim": dim, "hidden_dim": HIDDEN_DIM},
               os.path.join(CKPT_DIR, f"encoder_d{dim}.pt"))
    torch.save({"state_dict": clf.state_dict(), "dim": dim,
                "hidden1": MLP_HIDDEN1, "hidden2": MLP_HIDDEN2},
               os.path.join(CKPT_DIR, f"latent_mlp_d{dim}.pt"))
    print(f"Saved → {os.path.join(CKPT_DIR, f'latent_mlp_d{dim}.pt')}")
    return encoder, clf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=16)
    args = p.parse_args()
    train_joint(args.dim)


if __name__ == "__main__":
    main()
