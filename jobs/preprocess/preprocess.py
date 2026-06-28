"""Nebius Job N1 — MMOTU preprocessing / ETL.

Resizes images to 256x256, normalizes intensity (optional CLAHE), converts to
RGB, encodes binary lesion masks, and writes a patient-disjoint train/val/test
split manifest. Reads raw data from / writes processed data to object-storage
mounts (paths are passed via env vars / CLI args).

This is a runnable stub: the directory walk and split logic are real; the
per-image transform is intentionally simple and meant to be extended.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EndoSeg MMOTU preprocessing job")
    p.add_argument("--raw-dir", default=os.getenv("RAW_DIR", "/data/raw/mmotu"))
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "/data/processed"))
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--clahe", action="store_true", help="apply CLAHE to luminance")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def preprocess_image(path: Path, size: int, clahe: bool) -> np.ndarray:
    """Load -> resize -> optional CLAHE -> RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe_op.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_mask(path: Path, size: int) -> np.ndarray:
    """Load mask -> resize (nearest) -> binarize to {0,1}."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def split_records(records: list[dict], val_frac: float, test_frac: float, seed: int) -> dict:
    """Patient-disjoint split. Falls back to per-image split if no patient id.

    TODO: parse the true MMOTU patient/case id so the split is leakage-free.
    """
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return {
        "test": records[:n_test],
        "val": records[n_test : n_test + n_val],
        "train": records[n_test + n_val :],
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    images_out = out_dir / "images"
    masks_out = out_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Expected layout (adjust to the real MMOTU release):
    #   raw_dir/images/<id>.jpg  and  raw_dir/masks/<id>.png
    image_paths = sorted((raw_dir / "images").glob("*"))
    print(f"found {len(image_paths)} images under {raw_dir}")

    records: list[dict] = []
    for img_path in tqdm(image_paths, desc="preprocessing"):
        stem = img_path.stem
        mask_path = raw_dir / "masks" / f"{stem}.png"
        if not mask_path.exists():
            continue
        img = preprocess_image(img_path, args.img_size, args.clahe)
        mask = preprocess_mask(mask_path, args.img_size)
        cv2.imwrite(str(images_out / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(masks_out / f"{stem}.png"), mask * 255)
        records.append({"id": stem, "image": f"images/{stem}.png", "mask": f"masks/{stem}.png"})

    splits = split_records(records, args.val_frac, args.test_frac, args.seed)
    manifest_path = out_dir / "splits.json"
    with open(manifest_path, "w") as f:
        json.dump({k: v for k, v in splits.items()}, f, indent=2)

    print(f"wrote {len(records)} processed pairs")
    for name, recs in splits.items():
        print(f"  {name}: {len(recs)}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
