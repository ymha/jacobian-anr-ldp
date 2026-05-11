"""
Standard VAE training.

Trains the VAE on the public MNIST dataset (80/20 train/val split, seed 42)
and saves the encoder weights to checkpoints/encoder_d{D}.pt.

Latent vector extraction is handled by train_classifier.py --joint,
which fine-tunes the encoder and re-extracts Z_tr/Z_val/Z_te afterwards.
"""

import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision


def load_mnist():
    tf = transforms.ToTensor()
    tr = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=tf)
    te = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
    def _t(ds):
        return ds.data.unsqueeze(1).float() / 255.0, ds.targets.long()
    return *_t(tr), *_t(te)

CKPT_DIR   = "./checkpoints"
VAE_EPOCHS = 100
PATIENCE   = 10
HIDDEN_DIM = 512
BATCH_SIZE = 512
LR         = 1e-3
VAL_RATIO  = 0.2


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, z):
        return self.net(z)


class StandardVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x.flatten(1))
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x.flatten(1))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x, x_hat, mu, logvar):
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl    = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon


def train_vae(X_tr: torch.Tensor, X_val: torch.Tensor,
              latent_dim: int, verbose: bool = True) -> StandardVAE:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    n_train, n_val = len(X_tr), len(X_val)
    tr_loader  = DataLoader(TensorDataset(X_tr),  BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val), BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = StandardVAE(784, HIDDEN_DIM, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    if verbose:
        print(f"\n── Training Standard VAE  D={latent_dim} ──")
        print(f"   train={n_train}  val={n_val}  patience={PATIENCE}")

    best_val   = float("inf")
    best_state = None
    wait       = 0

    for epoch in range(1, VAE_EPOCHS + 1):
        model.train()
        total_loss = total_recon = 0.0
        for (x,) in tr_loader:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x)
            loss, recon = vae_loss(x, x_hat, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss  += loss.item()
            total_recon += recon.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device, non_blocking=True)
                x_hat, mu, logvar = model(x)
                val_loss += vae_loss(x, x_hat, mu, logvar)[0].item()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  [{epoch:3d}/{VAE_EPOCHS}]  "
                  f"train_loss/sample={total_loss/n_train:.2f}  "
                  f"recon/sample={total_recon/n_train:.2f}  "
                  f"val_loss/sample={val_loss/n_val:.2f}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  "
                          f"(best val_loss/sample={best_val/n_val:.2f})")
                break

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()
    D = args.dim

    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading MNIST...")
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  train: {X_tr.shape}  test: {X_te.shape}")

    # ── 1. Split train → 80 % train / 20 % val ───────────────────────────────
    n_val   = int(len(X_tr) * VAL_RATIO)
    n_train = len(X_tr) - n_val
    tr_ds, val_ds = random_split(
        TensorDataset(X_tr, y_tr), [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    X_tr_split,  y_tr_split  = tr_ds[:]
    X_val_split, y_val_split = val_ds[:]

    # ── 2. Train VAE ──────────────────────────────────────────────────────────
    vae = train_vae(X_tr_split, X_val_split, D, verbose=True)
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()

    # ── 3. Save Encoder ───────────────────────────────────────────────────────
    enc_path = os.path.join(CKPT_DIR, f"encoder_d{D}.pt")
    torch.save({"state_dict": vae.encoder.state_dict(),
                "latent_dim": D, "hidden_dim": HIDDEN_DIM}, enc_path)
    print(f"\nEncoder saved → {enc_path}")


if __name__ == "__main__":
    main()
