"""
Train ResNet-20 on CIFAR-10 and extract 64-dim penultimate-layer features.

Stage 1 (optional): train ResNet-20 from scratch.
Stage 2: extract features for train/val/test splits and save for LDP evaluation.

Usage:
  python train_resnet20.py                                          # train from scratch
  python train_resnet20.py --weights resnet20_cifar10.pth           # skip training
  python train_resnet20.py --weights ../resnet_20/resnet20_cifar10.pth
"""

import os
import argparse
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from model import ResNet20, FeatureClassifier, MLPClassifier, ACTIVATIONS

CKPT_DIR   = "./checkpoints"
LATENT_DIR = "./data/CIFAR10/latent"
LATENT_DIM = 64
MEAN       = (0.4914, 0.4822, 0.4465)
STD        = (0.2470, 0.2435, 0.2616)
VAL_RATIO  = 0.2
BATCH_SIZE = 128
EPOCHS     = 200


def _get_datasets(data_dir):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tr_aug  = datasets.CIFAR10(data_dir, train=True,  download=True,  transform=train_tf)
    tr_eval = datasets.CIFAR10(data_dir, train=True,  download=False, transform=eval_tf)
    te_ds   = datasets.CIFAR10(data_dir, train=False, download=True,  transform=eval_tf)
    return tr_aug, tr_eval, te_ds


@torch.no_grad()
def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += model(x.to(device)).argmax(1).eq(y.to(device)).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def _train(model, tr_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 150], gamma=0.1)

    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()
        sch.step()

        if epoch % 20 == 0 or epoch == 1:
            val_acc = _accuracy(model, val_loader, device)
            print(f"  [{epoch:3d}/{EPOCHS}] lr={sch.get_last_lr()[0]:.4f}  "
                  f"val_acc={val_acc:.2f}%  ({time.time()-t0:.1f}s)")
            if val_acc > best_acc:
                best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _extract(model, ds, indices, device, batch_size=512):
    model.eval()
    loader = DataLoader(Subset(ds, list(indices)), batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    Zs, ys = [], []
    for x, y in loader:
        Zs.append(model.features(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(Zs).float(), torch.cat(ys)


def _train_mlp(Z_tr: torch.Tensor, y_tr: torch.Tensor,
               Z_val: torch.Tensor, y_val: torch.Tensor,
               device: str, epochs: int = 100,
               activation: str = "relu") -> MLPClassifier:
    mlp = MLPClassifier(activation=activation).to(device)
    opt = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Z_tr, y_tr), batch_size=256, shuffle=True)

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        mlp.train()
        for z, y in loader:
            opt.zero_grad()
            criterion(mlp(z.to(device)), y.to(device)).backward()
            opt.step()
        sch.step()
        if epoch % 20 == 0 or epoch == epochs:
            mlp.eval()
            with torch.no_grad():
                acc = mlp(Z_val.to(device)).argmax(1).eq(y_val.to(device)).float().mean().item() * 100
            print(f"  [{epoch:3d}/{epochs}] val_acc={acc:.2f}%")
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(mlp.state_dict())

    mlp.load_state_dict(best_state)
    return mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default=None,
                        help="path to pretrained ResNet-20 state_dict; skips training")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--mlp",      action="store_true",
                        help="train MLP head on frozen features after extraction")
    parser.add_argument("--activation", default="relu", choices=list(ACTIVATIONS),
                        help="MLP activation function (default: relu)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LATENT_DIR, exist_ok=True)

    # 80/20 train/val split (reproducible)
    rng  = np.random.default_rng(args.seed)
    perm = rng.permutation(50000)
    n_val   = int(50000 * VAL_RATIO)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    tr_aug, tr_eval, te_ds = _get_datasets(args.data_dir)

    tr_loader  = DataLoader(Subset(tr_aug,  tr_idx),  BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(tr_eval, val_idx), BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    te_loader  = DataLoader(te_ds,                    BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = ResNet20().to(device)

    if args.weights:
        print(f"Loading weights from {args.weights}")
        state = torch.load(args.weights, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"val_acc={_accuracy(model, val_loader, device):.2f}%  "
              f"test_acc={_accuracy(model, te_loader, device):.2f}%")
    else:
        print(f"\nTraining ResNet-20 on CIFAR-10  (train={len(tr_idx)}  val={n_val})")
        model = _train(model, tr_loader, val_loader, device)
        te_acc = _accuracy(model, te_loader, device)
        print(f"Final test_acc={te_acc:.2f}%")
        ckpt_path = os.path.join(CKPT_DIR, "resnet20_cifar10.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved → {ckpt_path}")

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    # Save FeatureClassifier (fc head only — this is 'clf' in eval)
    clf = FeatureClassifier(model.fc)
    torch.save({"state_dict": clf.state_dict(), "latent_dim": LATENT_DIM},
               os.path.join(CKPT_DIR, "feature_clf.pt"))

    print("\nExtracting 64-dim features...")
    for split, ds, idx in [("tr",  tr_eval, tr_idx),
                            ("val", tr_eval, val_idx),
                            ("te",  te_ds,   range(len(te_ds)))]:
        Z, y = _extract(model, ds, idx, device)
        torch.save(Z, os.path.join(LATENT_DIR, f"Z_{split}.pt"))
        torch.save(y, os.path.join(LATENT_DIR, f"y_{split}.pt"))
        print(f"  {split}: Z{tuple(Z.shape)}  y{tuple(y.shape)}")

    print(f"\nFeatures  → {LATENT_DIR}/")
    print(f"Classifier → {CKPT_DIR}/feature_clf.pt")

    if args.mlp:
        print(f"\nTraining MLP head on frozen features (activation={args.activation})...")
        Z_tr_t  = torch.load(os.path.join(LATENT_DIR, "Z_tr.pt"),  weights_only=True).float()
        Z_val_t = torch.load(os.path.join(LATENT_DIR, "Z_val.pt"), weights_only=True).float()
        y_tr_t  = torch.load(os.path.join(LATENT_DIR, "y_tr.pt"),  weights_only=True)
        y_val_t = torch.load(os.path.join(LATENT_DIR, "y_val.pt"), weights_only=True)
        mlp = _train_mlp(Z_tr_t, y_tr_t, Z_val_t, y_val_t, device, activation=args.activation)
        ckpt_path = os.path.join(CKPT_DIR, f"mlp_clf_{args.activation}.pt")
        torch.save({"state_dict": mlp.state_dict(), "activation": args.activation}, ckpt_path)
        print(f"MLP classifier → {ckpt_path}")


if __name__ == "__main__":
    main()
