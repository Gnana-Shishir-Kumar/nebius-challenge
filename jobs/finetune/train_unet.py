"""Nebius Job N2 — fine-tune the compact 2D U-Net (browser model).

Trains the grayscale binary lesion-vs-background U-Net on MMOTU (OTU_2d):

  * Loads raw MMOTU image/mask pairs and binarizes masks (lesion vs background).
  * Preprocessing: resize 256x256, CLAHE on the grayscale image, normalize [0,1].
  * Augmentation (train only) via Albumentations: HFlip, ElasticTransform,
    RandomBrightnessContrast, GaussNoise.
  * Patient-disjoint 70/15/15 train/val/test split grouped by case prefix so the
    same case never leaks across splits.
  * Loss: 0.5 * BCE + 0.5 * Dice.  Optimizer: Adam(lr=1e-4) + ReduceLROnPlateau.
  * Logs Dice/IoU per epoch as a Markdown table that is easy to paste into the
    blog, and checkpoints the best (by val Dice) model to checkpoints/unet_best.pth.

Run a fast end-to-end sanity check (no real data required) with:

    python jobs/finetune/train_unet.py --smoke-test

The smoke test runs 2 batches/epoch for 2 epochs; if no dataset is found at
--data-dir it falls back to a tiny synthetic dataset so the full pipeline
(model -> loss -> backward -> metrics -> checkpoint) still executes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A

# Allow `python jobs/finetune/train_unet.py` from repo root or container (/app).
_here = Path(__file__).resolve().parent
for _candidate in (_here, *_here.parents):
    if (_candidate / "models" / "unet.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
from models.unet import build_model  # noqa: E402
from models.losses import build_loss  # noqa: E402
from models.metrics import MetricTracker, dice_score, iou_score  # noqa: E402

IMG_SIZE = 256
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description="EndoSeg U-Net fine-tuning job")
    p.add_argument("--data-dir", default="data/raw/mmotu",
                   help="MMOTU root (looks for images/ + annotations|masks/, "
                        "optionally under an OTU_2d/ subfolder).")
    p.add_argument("--out-dir", default="checkpoints",
                   help="Where to write unet_best.pth.")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default=default_device,
                   help="cuda or cpu (default: cuda if available else cpu).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="Fast sanity run: 2 batches/epoch, 2 epochs, synthetic "
                        "data if none is found.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset discovery + patient-disjoint split
# --------------------------------------------------------------------------- #
def _find_dirs(data_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate the images dir and the mask/annotation dir under ``data_dir``.

    Handles both ``<root>/images`` and the official ``<root>/OTU_2d/images``
    layout, with masks in either ``annotations/`` or ``masks/``.
    """
    candidates = [data_dir, data_dir / "OTU_2d", data_dir / "OTU_2d" / "OTU_2d"]
    for base in candidates:
        img_dir = base / "images"
        if not img_dir.is_dir():
            continue
        for mask_name in ("annotations", "masks"):
            mask_dir = base / mask_name
            if mask_dir.is_dir():
                return img_dir, mask_dir
    return None, None


def _case_key(stem: str) -> str:
    """Derive a case/patient key from a filename stem for disjoint splitting.

    MMOTU 2D filenames are mostly plain integers with no patient grouping, so we
    take the leading non-digit token when present (e.g. ``caseA_3`` -> ``caseA``)
    and otherwise fall back to the stem itself (each image is its own case). This
    keeps the split leakage-free whenever a grouping prefix actually exists.
    """
    head = stem.replace("-", "_").split("_")[0]
    stripped = head.rstrip("0123456789")
    return stripped if stripped else stem


def discover_pairs(data_dir: Path) -> list[tuple[Path, Path, str]]:
    """Return ``(image_path, mask_path, case_key)`` triples found under data_dir."""
    img_dir, mask_dir = _find_dirs(data_dir)
    if img_dir is None or mask_dir is None:
        return []

    masks_by_stem: dict[str, Path] = {}
    for m in mask_dir.iterdir():
        if m.suffix.lower() in IMAGE_EXTS:
            masks_by_stem.setdefault(m.stem, m)

    pairs: list[tuple[Path, Path, str]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        mask = masks_by_stem.get(img.stem)
        if mask is not None:
            pairs.append((img, mask, _case_key(img.stem)))
    return pairs


def discover_preprocessed_splits(data_dir: Path) -> dict[str, list[str]] | None:
    """Load a splits.json manifest written by jobs/preprocess/preprocess.py.

    Returns ``None`` if ``data_dir`` doesn't look like a preprocessed output
    directory (no splits.json), so callers can fall back to raw-pair discovery.
    """
    splits_path = data_dir / "splits.json"
    if not splits_path.is_file():
        return None
    with open(splits_path) as f:
        return json.load(f)


def split_by_case(
    pairs: list[tuple[Path, Path, str]],
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, list[tuple[Path, Path, str]]]:
    """Group by case key, then assign whole cases to train/val/test (70/15/15)."""
    groups: dict[str, list[tuple[Path, Path, str]]] = {}
    for item in pairs:
        groups.setdefault(item[2], []).append(item)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])

    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for k in keys:
        split = "test" if k in test_keys else "val" if k in val_keys else "train"
        out[split].extend(groups[k])

    if n == len(pairs):
        print("  note: no case-grouping prefix found in filenames -> split is "
              "effectively per-image (still deterministic).")
    return out


# --------------------------------------------------------------------------- #
# Transforms + datasets
# --------------------------------------------------------------------------- #
def build_augmentations() -> A.Compose:
    """Train-time geometric/intensity augmentation on uint8 grayscale images."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
    ])


class MMOTUDataset(Dataset):
    """Loads raw MMOTU pairs: grayscale + CLAHE + resize, binary mask, [0,1]."""

    def __init__(
        self,
        records: list[tuple[Path, Path, str]],
        img_size: int,
        augment: bool,
    ) -> None:
        self.records = records
        self.img_size = img_size
        self.augment = augment
        self.aug = build_augmentations() if augment else None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        return self.clahe.apply(img)  # uint8 HxW

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"could not read mask: {path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)  # binary lesion vs background

    def __getitem__(self, idx: int):
        img_path, mask_path, _ = self.records[idx]
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if self.aug is not None:
            out = self.aug(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = image.astype(np.float32) / 255.0           # normalize to [0,1]
        image_t = torch.from_numpy(image).unsqueeze(0)      # (1, H, W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1,H,W)
        return image_t, mask_t


class PreprocessedMMOTUDataset(Dataset):
    """Loads MMOTU pairs already preprocessed by jobs/preprocess/preprocess.py.

    Expects ``data_dir/images/{stem}.npy`` (float32 HxW, normalized [0,1]) and
    ``data_dir/masks/{stem}.npy`` (uint8 HxW, values {0,1}) for each stem.
    """

    def __init__(self, data_dir: Path, stems: list[str], augment: bool) -> None:
        self.data_dir = data_dir
        self.stems = stems
        self.augment = augment
        self.aug = build_augmentations() if augment else None

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        image = np.load(self.data_dir / "images" / f"{stem}.npy")  # float32 HxW [0,1]
        mask = np.load(self.data_dir / "masks" / f"{stem}.npy")    # uint8 HxW {0,1}

        if self.aug is not None:
            image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            out = self.aug(image=image_u8, mask=mask)
            image = out["image"].astype(np.float32) / 255.0
            mask = out["mask"]

        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1,H,W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)    # (1,H,W)
        return image_t, mask_t


class SyntheticDataset(Dataset):
    """Tiny random dataset used by --smoke-test when no real data is present."""

    def __init__(self, n: int, img_size: int) -> None:
        self.n = n
        self.img_size = img_size
        rng = np.random.default_rng(0)
        self._imgs = rng.random((n, 1, img_size, img_size), dtype=np.float32)
        masks = (rng.random((n, 1, img_size, img_size)) > 0.7).astype(np.float32)
        self._masks = masks

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.from_numpy(self._imgs[idx]), torch.from_numpy(self._masks[idx])


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, loss_fn, optimizer, device, train, max_batches=None):
    model.train(train)
    tracker = MetricTracker()
    total_loss, n_seen = 0.0, 0
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = loss_fn(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        n_seen += bs
        tracker.update(logits.detach(), targets)

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(n_seen, 1)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("requested cuda but it is unavailable -> falling back to cpu")
        device = "cpu"

    epochs = 2 if args.smoke_test else args.epochs
    max_batches = 2 if args.smoke_test else None
    num_workers = 0 if args.smoke_test else args.num_workers

    print(f"device: {device} | epochs: {epochs} | batch-size: {args.batch_size}"
          + (" | SMOKE TEST" if args.smoke_test else ""))

    # ---- data ----
    data_dir = Path(args.data_dir)
    preprocessed_splits = discover_preprocessed_splits(data_dir)
    pairs = [] if preprocessed_splits is not None else discover_pairs(data_dir)

    if preprocessed_splits is not None:
        print(f"found preprocessed splits.json under {data_dir}")
        for name in ("train", "val", "test"):
            print(f"  {name}: {len(preprocessed_splits.get(name, []))}")
        train_ds = PreprocessedMMOTUDataset(data_dir, preprocessed_splits["train"], augment=True)
        val_ds = PreprocessedMMOTUDataset(data_dir, preprocessed_splits["val"], augment=False)
    elif pairs:
        print(f"found {len(pairs)} image/mask pairs under {data_dir}")
        splits = split_by_case(pairs, seed=args.seed)
        for name in ("train", "val", "test"):
            print(f"  {name}: {len(splits[name])}")
        train_ds = MMOTUDataset(splits["train"], args.img_size, augment=True)
        val_ds = MMOTUDataset(splits["val"], args.img_size, augment=False)
    elif args.smoke_test:
        print(f"no dataset found at {data_dir} -> using synthetic data for smoke test")
        train_ds = SyntheticDataset(2 * args.batch_size, args.img_size)
        val_ds = SyntheticDataset(2 * args.batch_size, args.img_size)
    else:
        raise SystemExit(
            f"No MMOTU pairs found under {data_dir}. Run data/download_mmotu.sh "
            f"first, or pass --data-dir / use --smoke-test."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)

    # ---- model / loss / optim ----
    model = build_model(num_classes=1, in_channels=1).to(device)
    loss_fn = build_loss("dice_bce", bce_weight=0.5)  # 0.5*BCE + 0.5*Dice
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "unet_best.pth"

    # ---- blog-pasteable Markdown log header ----
    print("\n<!-- training log (Markdown; paste into blog) -->")
    print("| epoch |      lr | train_loss | val_loss | val_dice | val_iou |")
    print("|------:|--------:|-----------:|---------:|---------:|--------:|")

    best_dice = 0.0
    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, loss_fn, optimizer, device,
                            train=True, max_batches=max_batches)
        val_m = run_epoch(model, val_loader, loss_fn, optimizer, device,
                          train=False, max_batches=max_batches)
        scheduler.step(val_m["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"| {epoch:5d} | {lr_now:.1e} | {train_m['loss']:10.4f} | "
              f"{val_m['loss']:8.4f} | {val_m['dice']:8.4f} | {val_m['iou']:7.4f} |")

        if val_m["dice"] >= best_dice:
            best_dice = val_m["dice"]
            torch.save(
                {"model": model.state_dict(), "val_dice": best_dice,
                 "val_iou": val_m["iou"], "epoch": epoch,
                 "in_channels": 1, "img_size": args.img_size},
                ckpt_path,
            )

    print(f"\nbest val dice: {best_dice:.4f} | checkpoint -> {ckpt_path}")
    if args.smoke_test:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
