#!/usr/bin/env python3
"""EndoSeg :: MMOTU dataset sanity / EDA script.

Scans data/MMOTU_DataSet (or --root) and prints a structured summary:

  Total images       — .JPG files in images/
  Paired masks       — exact-stem .PNG match in annotations/  (1.JPG -> 1.PNG)
  Pre-made binary    — _binary.PNG variants in annotations/   (1000_binary.PNG)
  8-class folders    — subdirectory names under 8_layers/

Then saves a 4x4 visualization grid (4 random samples × 4 rows) to
data/sample_grid.png.

Rows:
  0 — original ultrasound image
  1 — multi-class segmentation mask
  2 — binary mask (_binary.PNG if present, else mask > 0)
  3 — lesion overlay on image

Usage:
  python data/explore_mmotu.py --root data/MMOTU_DataSet
  python data/explore_mmotu.py --root data/MMOTU_DataSet --n 4 --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Explore MMOTU image/mask pairs.")
    p.add_argument(
        "--root",
        type=Path,
        default=here / "MMOTU_DataSet",
        help="Dataset root containing images/, annotations/, 8_layers/",
    )
    p.add_argument("--n", type=int, default=4, help="samples in the grid (default 4)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=here / "sample_grid.png",
        help="output path for the visualization grid",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset introspection helpers
# ---------------------------------------------------------------------------

def find_mask(annotations_dir: Path, stem: str) -> Path | None:
    """Exact-stem match only: '1' -> annotations/1.PNG  (case-insensitive ext)."""
    for ext in (".PNG", ".png"):
        cand = annotations_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def find_binary_mask(annotations_dir: Path, stem: str) -> Path | None:
    """Look for the precomputed _binary.PNG variant for this stem."""
    for ext in ("_binary.PNG", "_binary.png"):
        cand = annotations_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def count_binary_masks(annotations_dir: Path) -> int:
    """Count all *_binary.PNG files regardless of stem."""
    return sum(
        1 for p in annotations_dir.iterdir()
        if p.is_file() and p.name.lower().endswith("_binary.png")
    )


def list_class_folders(root: Path) -> list[str]:
    """Return sorted subfolder names from 8_layers/."""
    layers = root / "8_layers"
    if not layers.is_dir():
        return []
    return sorted(d.name for d in layers.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def load_mask_arr(mask_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path))
    return arr[..., 0] if arr.ndim == 3 else arr


def make_grid(samples: list[dict], out_path: Path) -> None:
    n = len(samples)
    rows, cols = 4, n
    row_labels = ["image", "mask", "binary", "overlay"]

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for c, s in enumerate(samples):
        img = s["image"]
        mask = s["mask"]
        bmask = s["binary_mask"]

        # Row 0: image
        axes[0, c].imshow(img)
        axes[0, c].set_title(s["name"], fontsize=8)

        # Row 1: multi-class mask
        if mask is not None:
            axes[1, c].imshow(mask, cmap="viridis")
            axes[1, c].set_title(f"IDs: {np.unique(mask).tolist()}", fontsize=7)
        else:
            axes[1, c].text(0.5, 0.5, "no mask", ha="center", va="center", transform=axes[1, c].transAxes)

        # Row 2: binary mask
        if bmask is not None:
            axes[2, c].imshow(bmask, cmap="gray")
            axes[2, c].set_title("_binary.PNG" if s["has_prebuilt_binary"] else "mask>0", fontsize=7)
        else:
            axes[2, c].text(0.5, 0.5, "no binary", ha="center", va="center", transform=axes[2, c].transAxes)

        # Row 3: red overlay
        axes[3, c].imshow(img)
        if bmask is not None:
            alpha = (bmask > 0).astype(float)
            overlay = np.zeros((*alpha.shape, 4))
            overlay[..., 0] = 1.0   # red channel
            overlay[..., 3] = alpha * 0.45
            axes[3, c].imshow(overlay)

    for r in range(rows):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=11)
        for c in range(cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("MMOTU — random samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        sys.exit(f"Dataset root not found: {root}\nRun data/download_mmotu.sh first or pass --root.")

    images_dir = root / "images"
    annotations_dir = root / "annotations"

    for d, label in [(images_dir, "images/"), (annotations_dir, "annotations/")]:
        if not d.is_dir():
            sys.exit(f"Expected subdirectory not found: {d}  (looked for {label} inside {root})")

    # Collect all .JPG files (case-insensitive)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jpg"
    )
    if not image_paths:
        sys.exit(f"No .JPG files found under {images_dir}")

    paired = sum(1 for p in image_paths if find_mask(annotations_dir, p.stem) is not None)
    n_binary = count_binary_masks(annotations_dir)
    class_folders = list_class_folders(root)

    print(f"Dataset root      : {root}")
    print(f"Total images      : {len(image_paths)}")
    print(f"Paired masks      : {paired}")
    print(f"Pre-made binary masks: {n_binary}")
    print(f"8-class folders   : {class_folders}")
    print()

    # Sample and inspect
    rng = random.Random(args.seed)
    n = min(args.n, len(image_paths))
    sample_paths = rng.sample(image_paths, n)

    samples: list[dict] = []
    for img_path in sample_paths:
        mask_path = find_mask(annotations_dir, img_path.stem)
        bin_path = find_binary_mask(annotations_dir, img_path.stem)

        img_arr = np.asarray(Image.open(img_path).convert("RGB"))
        mask_arr = load_mask_arr(mask_path) if mask_path else None

        if bin_path is not None:
            binary_arr = load_mask_arr(bin_path)
            has_prebuilt = True
        elif mask_arr is not None:
            binary_arr = (mask_arr > 0).astype(np.uint8) * 255
            has_prebuilt = False
        else:
            binary_arr = None
            has_prebuilt = False

        samples.append({
            "name": img_path.name,
            "image": img_arr,
            "mask": mask_arr,
            "binary_mask": binary_arr,
            "has_prebuilt_binary": has_prebuilt,
        })

    make_grid(samples, args.out.expanduser().resolve())
    print(f"Saved grid -> {args.out.expanduser().resolve()}")


if __name__ == "__main__":
    main()
