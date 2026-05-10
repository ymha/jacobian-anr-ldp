"""Extract ResNet-20 features from CIFAR-10-C corrupted test images.

CIFAR-10-C format (https://zenodo.org/record/2535967):
  Each corruption: {name}.npy  shape (50000, 32, 32, 3) uint8
  5 severity levels × 10000 images, stacked in severity order.
  labels.npy: shape (50000,), same 10000 labels repeated 5 times.

Outputs (per corruption × severity):
  data/CIFAR10-C/latent/{corruption}_s{severity}/Z_te.pt  (10000 × 64)
  data/CIFAR10-C/latent/{corruption}_s{severity}/y_te.pt  (10000,)

Usage:
  python extract_cifar10c_features.py --c10c_dir /path/to/CIFAR-10-C --severity 1
  python extract_cifar10c_features.py --c10c_dir /path/to/CIFAR-10-C --severity 1 --corrupt_types gaussian_noise shot_noise
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import ResNet20

CKPT_DIR = "./checkpoints"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

ALL_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
]


def to_tensor_normalized(images_np: np.ndarray) -> torch.Tensor:
    """(N, 32, 32, 3) uint8  →  (N, 3, 32, 32) float, normalized."""
    t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


@torch.no_grad()
def extract_features(model, images_np: np.ndarray, device: str, batch_size: int = 512):
    tensors = to_tensor_normalized(images_np)
    loader  = DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    Zs = []
    for (x,) in loader:
        Zs.append(model.features(x.to(device)).cpu())
    return torch.cat(Zs).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c10c_dir",      type=Path, required=True,
                   help="Directory containing CIFAR-10-C .npy files")
    p.add_argument("--severity",      type=int, default=1, choices=range(1, 6))
    p.add_argument("--corrupt_types", nargs="*", default=None,
                   help="Corruption types to process (default: all 19)")
    p.add_argument("--out_dir",       type=Path,
                   default=Path("./data/CIFAR10-C/latent"))
    p.add_argument("--batch_size",    type=int, default=512)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ResNet20().to(device)
    ckpt  = torch.load(f"{CKPT_DIR}/resnet20_cifar10.pt", map_location=device,
                       weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    labels_all = np.load(args.c10c_dir / "labels.npy")   # (50000,)
    s          = args.severity
    start, end = (s - 1) * 10000, s * 10000
    y_te       = torch.from_numpy(labels_all[start:end].astype(np.int64))

    for ctype in (args.corrupt_types or ALL_CORRUPTIONS):
        npy_path = args.c10c_dir / f"{ctype}.npy"
        if not npy_path.exists():
            print(f"  [skip] {npy_path} not found")
            continue

        images = np.load(npy_path)[start:end]   # (10000, 32, 32, 3)
        print(f"  {ctype}  severity={s} ...", end=" ", flush=True)
        Z_te    = extract_features(model, images, device, args.batch_size)

        out_dir = args.out_dir / f"{ctype}_s{s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(Z_te, out_dir / "Z_te.pt")
        torch.save(y_te, out_dir / "y_te.pt")
        print(f"saved → {out_dir}/  Z_te{tuple(Z_te.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
