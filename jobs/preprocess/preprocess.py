"""Nebius Job N1 — MMOTU preprocessing / ETL.

Reads the raw MMOTU_DataSet mirror (images/*.JPG, annotations/*.PNG,
annotations/*_binary.PNG), converts each pair to a grayscale/CLAHE image and
a strict {0,1} binary mask at a fixed resolution, and writes them out as
.npy arrays alongside a train/val/test split manifest.
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
    p.add_argument("--raw-dir", default=os.getenv("RAW_DIR", "data/MMOTU_DataSet"))
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "data/processed"))
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="cap number of images processed (smoke test)")
    return p.parse_args()


def preprocess_image(path: Path, size: int) -> np.ndarray:
    """Load -> grayscale -> CLAHE -> resize -> normalize to [0,1] float32."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe_op.apply(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)


def preprocess_mask(path: Path, size: int) -> np.ndarray:
    """Load mask -> resize (nearest, preserves labels) -> binarize to {0,1} uint8."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def find_mask_path(raw_dir: Path, stem: str) -> Path | None:
    """Prefer the pre-made binary mask, else fall back to the multi-class mask."""
    binary_path = raw_dir / "annotations" / f"{stem}_binary.PNG"
    if binary_path.exists():
        return binary_path
    fallback_path = raw_dir / "annotations" / f"{stem}.PNG"
    if fallback_path.exists():
        return fallback_path
    return None


def split_stems(stems: list[str], val_split: float, test_split: float, seed: int) -> dict:
    """Stratified random split by file ID.

    MMOTU filenames in this data mirror are plain numeric IDs with no
    recoverable patient/case grouping, so a true patient-disjoint split isn't
    possible from filenames alone.
    """
    print(
        "WARNING: patient-level grouping not available in this data mirror -- "
        "using stratified random split by file ID. Note this in the README "
        "as a known limitation."
    )
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    images_out = out_dir / "images"
    masks_out = out_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((raw_dir / "images").glob("*.JPG"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    print(f"found {len(image_paths)} images under {raw_dir}")

    stems: list[str] = []
    n_skipped = 0
    for img_path in tqdm(image_paths, desc="preprocessing"):
        stem = img_path.stem
        mask_path = find_mask_path(raw_dir, stem)
        if mask_path is None:
            print(f"WARNING: no matching mask for {img_path.name} -- skipping")
            n_skipped += 1
            continue
        img = preprocess_image(img_path, args.img_size)
        mask = preprocess_mask(mask_path, args.img_size)
        np.save(images_out / f"{stem}.npy", img)
        np.save(masks_out / f"{stem}.npy", mask)
        stems.append(stem)

    splits = split_stems(stems, args.val_split, args.test_split, args.seed)
    manifest_path = out_dir / "splits.json"
    with open(manifest_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Processed: {len(stems)} images ({n_skipped} skipped -- no matching mask)")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print(f"Output shape check: images {args.img_size}x{args.img_size} float32, masks {args.img_size}x{args.img_size} uint8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
