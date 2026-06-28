"""Nebius Job N2 — fine-tune the compact 2D U-Net (browser model).

Trains on the processed MMOTU split produced by the preprocessing job, logs
Dice/IoU per epoch, and checkpoints the best model to the object-storage mount.
Augmentation uses Albumentations (flips, elastic, intensity, noise).

Runnable stub: the training loop, dataset, and checkpointing are real; point
`--data-dir` at the preprocessing output and `--out-dir` at a checkpoint mount.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
import cv2

# Allow `python jobs/finetune/train_unet.py` from repo root or container.
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.unet import build_model  # noqa: E402
from models.losses import build_loss  # noqa: E402
from models.metrics import MetricTracker  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EndoSeg U-Net fine-tuning job")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "/data/processed"))
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "/checkpoints"))
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_transforms(img_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    return A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


class MMOTUDataset(Dataset):
    """Reads (image, mask) pairs listed in splits.json for one split."""

    def __init__(self, data_dir: str, split: str, transforms: A.Compose) -> None:
        self.root = Path(data_dir)
        with open(self.root / "splits.json") as f:
            self.records = json.load(f)[split]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = cv2.cvtColor(cv2.imread(str(self.root / rec["image"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.root / rec["mask"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        out = self.transforms(image=img, mask=mask)
        image = torch.from_numpy(out["image"].transpose(2, 0, 1)).float()
        target = torch.from_numpy(out["mask"]).unsqueeze(0).float()
        return image, target


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    tracker = MetricTracker()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = loss_fn(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        tracker.update(logits.detach(), targets)
    n = max(len(loader.dataset), 1)
    metrics = tracker.compute()
    metrics["loss"] = total_loss / n
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MMOTUDataset(args.data_dir, "train", build_transforms(args.img_size, True))
    val_ds = MMOTUDataset(args.data_dir, "val", build_transforms(args.img_size, False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=1).to(device)
    loss_fn = build_loss("dice_bce")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} dice {train_metrics['dice']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} dice {val_metrics['dice']:.4f} iou {val_metrics['iou']:.4f}"
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            ckpt = out_dir / "unet_best.pt"
            torch.save({"model": model.state_dict(), "val_dice": best_dice, "epoch": epoch}, ckpt)
            print(f"  saved new best -> {ckpt} (dice {best_dice:.4f})")

    print(f"done. best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
