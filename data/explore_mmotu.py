#!/usr/bin/env python3
"""EndoSeg :: quick MMOTU (OTU_2d) sanity / EDA script.

Loads 5 random image+mask pairs from the MMOTU OTU_2d dataset and reports, for
each pair:
  * image size (W x H) and channel/mode
  * pixel value range (min, max)
  * the unique values present in the mask (i.e. the class / region IDs)

It then saves a 3x5 visualization grid to ``data/sample_grid.png``:
  row 0 -> original ultrasound image
  row 1 -> segmentation mask
  row 2 -> mask overlaid on the image

Only matplotlib, PIL (Pillow) and numpy are used.

Usage:
  python data/explore_mmotu.py
  python data/explore_mmotu.py --root data/raw/mmotu/OTU_2d
  python data/explore_mmotu.py --n 5 --seed 0 --out data/sample_grid.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")  # headless / job-friendly backend
import matplotlib.pyplot as plt  # noqa: E402

# MMOTU OTU_2d category id -> name (see arXiv:2207.06799 / PRD §8).
CLASS_NAMES = {
    0: "Chocolate cyst (endometrioma)",
    1: "Serous cystadenoma",
    2: "Teratoma",
    3: "Theca cell tumor",
    4: "Simple cyst",
    5: "Normal ovary",
    6: "Mucinous cystadenoma",
    7: "High grade serous cystadenoma",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MASK_EXTS = (".png", ".bmp")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Explore random MMOTU image/mask pairs.")
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="OTU_2d dataset root (containing images/ and annotations|masks/). "
        "If omitted, auto-detected under data/raw.",
    )
    p.add_argument("--n", type=int, default=5, help="number of random samples (default 5)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument(
        "--out",
        type=Path,
        default=here / "sample_grid.png",
        help="output path for the visualization grid",
    )
    return p.parse_args()


def find_dataset_root(explicit: Path | None) -> Path:
    """Return a directory that has images/ and an annotations|masks/ subdir."""
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.exists():
            sys.exit(f"--root does not exist: {root}")
        return root

    search_base = Path(__file__).resolve().parent / "raw"
    candidates: list[Path] = []
    if search_base.exists():
        for d in search_base.rglob("*"):
            if d.is_dir() and (d / "images").is_dir() and (
                (d / "annotations").is_dir() or (d / "masks").is_dir()
            ):
                candidates.append(d)
    if not candidates:
        sys.exit(
            "Could not auto-detect an OTU_2d dataset under data/raw.\n"
            "Run data/download_mmotu.sh first, or pass --root <path>."
        )
    # Prefer a folder literally named OTU_2d if present.
    candidates.sort(key=lambda p: (0 if p.name.lower() == "otu_2d" else 1, len(str(p))))
    return candidates[0]


def masks_dir_for(root: Path) -> Path:
    if (root / "annotations").is_dir():
        return root / "annotations"
    return root / "masks"


def find_mask(masks_dir: Path, stem: str) -> Path | None:
    """Find a mask file matching the image stem, tolerant of suffixes/case."""
    # Exact stem match first.
    for ext in MASK_EXTS:
        for cand in (masks_dir / f"{stem}{ext}", masks_dir / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    # Fall back to any file that starts with the stem (e.g. "12_binary.PNG").
    for cand in sorted(masks_dir.iterdir()):
        if cand.is_file() and cand.stem.lower().startswith(stem.lower()) and cand.suffix.lower() in MASK_EXTS:
            return cand
    return None


def describe_mask_values(unique_vals: np.ndarray) -> str:
    """Annotate mask unique values with class names where plausible."""
    parts = []
    for v in unique_vals.tolist():
        iv = int(v)
        if iv == 0:
            parts.append("0=background")
        elif iv in CLASS_NAMES:
            parts.append(f"{iv}={CLASS_NAMES[iv]}")
        elif iv == 255:
            parts.append("255=foreground/lesion")
        else:
            parts.append(str(iv))
    return ", ".join(parts)


def main() -> None:
    args = parse_args()
    root = find_dataset_root(args.root)
    images_dir = root / "images"
    masks_dir = masks_dir_for(root)

    print(f"==> dataset root : {root}")
    print(f"    images dir   : {images_dir}")
    print(f"    masks dir    : {masks_dir}")

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        sys.exit(f"No images found under {images_dir}")
    print(f"    total images : {len(image_paths)}")

    rng = random.Random(args.seed)
    n = min(args.n, len(image_paths))
    sample_paths = rng.sample(image_paths, n)

    samples: list[dict] = []
    print(f"\n==> inspecting {n} random samples (seed={args.seed})\n")
    for img_path in sample_paths:
        mask_path = find_mask(masks_dir, img_path.stem)

        img = Image.open(img_path).convert("RGB")
        img_arr = np.asarray(img)

        if mask_path is not None:
            mask = Image.open(mask_path)
            mask_arr = np.asarray(mask)
            # Collapse multi-channel masks to a single label channel.
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            uniq = np.unique(mask_arr)
        else:
            mask_arr = None
            uniq = np.array([])

        print(f"  {img_path.name}")
        print(f"    image size      : {img.width} x {img.height}  (mode={img.mode}, channels={img_arr.shape[-1]})")
        print(f"    pixel range     : min={int(img_arr.min())}, max={int(img_arr.max())}, dtype={img_arr.dtype}")
        if mask_arr is not None:
            print(f"    mask            : {mask_path.name}  shape={mask_arr.shape}, dtype={mask_arr.dtype}")
            print(f"    mask unique IDs : {uniq.tolist()}  ({describe_mask_values(uniq)})")
        else:
            print("    mask            : NOT FOUND")
        print()

        samples.append(
            {"name": img_path.name, "image": img_arr, "mask": mask_arr, "uniq": uniq}
        )

    # ----------------------------------------------------------------------
    # 3 x N visualization grid: image / mask / overlay.
    # ----------------------------------------------------------------------
    rows, cols = 3, n
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_2d(axes)
    if cols == 1:
        axes = axes.reshape(rows, 1)
    row_titles = ["image", "mask", "overlay"]

    for c, s in enumerate(samples):
        img_arr = s["image"]
        mask_arr = s["mask"]

        # Row 0: image
        axes[0, c].imshow(img_arr)
        axes[0, c].set_title(s["name"], fontsize=9)

        # Row 1: mask
        if mask_arr is not None:
            axes[1, c].imshow(mask_arr, cmap="viridis")
            axes[1, c].set_title(f"IDs: {s['uniq'].tolist()}", fontsize=8)
        else:
            axes[1, c].text(0.5, 0.5, "no mask", ha="center", va="center")

        # Row 2: overlay (binarized lesion in red over grayscale image)
        axes[2, c].imshow(img_arr)
        if mask_arr is not None:
            binary = (mask_arr > 0).astype(float)
            overlay = np.zeros((*binary.shape, 4))
            overlay[..., 0] = 1.0  # red
            overlay[..., 3] = binary * 0.45  # alpha where lesion present
            axes[2, c].imshow(overlay)

    for r in range(rows):
        axes[r, 0].set_ylabel(row_titles[r], fontsize=11)
        for c in range(cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("MMOTU OTU_2d — random samples (image / mask / overlay)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"==> saved visualization grid -> {out_path}")


if __name__ == "__main__":
    main()
