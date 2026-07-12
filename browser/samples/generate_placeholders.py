"""Generate synthetic ultrasound-like sample images for the browser demo gallery.

Produces sample1.png, sample2.png, sample3.png: 512x512 grayscale images with
a dark background, gaussian noise (speckle-like texture), a bright elliptical
"lesion", and a slight blur -- just enough to look plausible for a UI demo,
not a substitute for real scan data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
OUT_DIR = Path(__file__).parent

SAMPLES = [
    {"name": "sample1.png", "seed": 1, "center": (210, 230), "axes": (70, 55), "angle": 15},
    {"name": "sample2.png", "seed": 2, "center": (300, 280), "axes": (55, 80), "angle": -20},
    {"name": "sample3.png", "seed": 3, "center": (256, 320), "axes": (90, 60), "angle": 40},
]


def make_sample(seed: int, center: tuple[int, int], axes: tuple[int, int], angle: float) -> Image.Image:
    rng = np.random.default_rng(seed)

    background = rng.normal(loc=25, scale=1.0, size=(SIZE, SIZE))
    noise = rng.normal(loc=0, scale=18, size=(SIZE, SIZE))
    field = background + noise

    y, x = np.ogrid[:SIZE, :SIZE]
    cx, cy = center
    ax, ay = axes
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = (x - cx) * cos_t + (y - cy) * sin_t
    y_rot = -(x - cx) * sin_t + (y - cy) * cos_t
    ellipse_mask = (x_rot / ax) ** 2 + (y_rot / ay) ** 2 <= 1.0

    lesion_core = rng.normal(loc=190, scale=12, size=(SIZE, SIZE))
    field = np.where(ellipse_mask, lesion_core, field)

    field = np.clip(field, 0, 255).astype(np.uint8)
    img = Image.fromarray(field, mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


def main() -> None:
    for spec in SAMPLES:
        img = make_sample(spec["seed"], spec["center"], spec["axes"], spec["angle"])
        out_path = OUT_DIR / spec["name"]
        img.save(out_path)
        print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
