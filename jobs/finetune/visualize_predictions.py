"""Sanity-check the trained U-Net visually on real test-split images.

Loads checkpoints/unet_best.pth, runs inference on a random sample of the
test split (data/processed/splits.json), and saves a grid to
data/prediction_grid.png showing, per sample: original image | ground truth
mask | predicted mask | overlay. Prints the per-sample Dice score for each
row above the grid.

This exists because an aggregate Dice/IoU number can look "fine" even when
the model has collapsed to predicting all-background (a common failure mode
under class imbalance) -- eyeballing real predictions catches that.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.unet import build_model  # noqa: E402
from models.metrics import dice_score  # noqa: E402

DATA_DIR = Path("data/processed")
CHECKPOINT_PATH = Path("checkpoints/unet_best.pth")
OUT_PATH = Path("data/prediction_grid.png")
N_SAMPLES = 8
SEED = 42


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(num_classes=1, in_channels=ckpt.get("in_channels", 1))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_sample(stem: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.load(DATA_DIR / "images" / f"{stem}.npy")  # float32 HxW [0,1]
    mask = np.load(DATA_DIR / "masks" / f"{stem}.npy")    # uint8 HxW {0,1}
    return image, mask


def main() -> None:
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    test_stems = splits["test"]

    rng = random.Random(SEED)
    sample_stems = rng.sample(test_stems, min(N_SAMPLES, len(test_stems)))

    model = load_model(CHECKPOINT_PATH)

    fig, axes = plt.subplots(len(sample_stems), 4, figsize=(12, 3 * len(sample_stems)))
    col_titles = ["Image", "Ground truth", "Prediction", "Overlay"]

    for row, stem in enumerate(sample_stems):
        image, mask = load_sample(stem)
        image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = model(image_t)
            pred = (torch.sigmoid(logits) > 0.5).float()
            dice = dice_score(logits, mask_t).item()

        pred_np = pred.squeeze().numpy()

        overlay = np.stack([image, image, image], axis=-1)
        overlay[..., 0] = np.where(mask > 0, 1.0, overlay[..., 0])       # red = ground truth
        overlay[..., 1] = np.where(pred_np > 0, 1.0, overlay[..., 1])   # green = prediction
        overlay = np.clip(overlay, 0, 1)

        row_axes = axes[row] if len(sample_stems) > 1 else axes
        panels = [image, mask, pred_np, overlay]
        cmaps = ["gray", "gray", "gray", None]
        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = row_axes[col]
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)

        row_axes[0].set_ylabel(f"stem={stem}", fontsize=9)
        row_axes[0].text(
            0.0, 1.12, f"Dice = {dice:.4f}",
            transform=row_axes[0].transAxes, fontsize=10, fontweight="bold",
        )
        print(f"stem={stem}: Dice = {dice:.4f}")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"\nsaved grid -> {OUT_PATH}")


if __name__ == "__main__":
    main()
