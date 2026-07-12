"""Compute real test-split Dice/IoU for the trained U-Net (not just val).

Loads checkpoints/unet_best.pth, runs inference over every image in the
test split (data/processed/splits.json), and prints the mean Dice/IoU
across the full test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from models import build_model, dice_score, iou_score  # noqa: E402

DATA_DIR = Path("data/processed")
CHECKPOINT_PATH = Path("checkpoints/unet_best.pth")
BATCH_SIZE = 8


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(num_classes=1, in_channels=ckpt.get("in_channels", 1))
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def load_batch(stems: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    images = [np.load(DATA_DIR / "images" / f"{s}.npy") for s in stems]
    masks = [np.load(DATA_DIR / "masks" / f"{s}.npy") for s in stems]
    image_t = torch.from_numpy(np.stack(images)).unsqueeze(1).float()  # (B,1,H,W)
    mask_t = torch.from_numpy(np.stack(masks)).unsqueeze(1).float()    # (B,1,H,W)
    return image_t, mask_t


def main() -> None:
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    test_stems = splits["test"]

    model = load_model(CHECKPOINT_PATH)

    dice_total, iou_total, n_seen = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(test_stems), BATCH_SIZE):
            batch_stems = test_stems[i : i + BATCH_SIZE]
            images, masks = load_batch(batch_stems)
            logits = model(images)
            bs = images.size(0)
            dice_total += dice_score(logits, masks).item() * bs
            iou_total += iou_score(logits, masks).item() * bs
            n_seen += bs

    test_dice = dice_total / n_seen
    test_iou = iou_total / n_seen
    print(f"test images: {n_seen}")
    print(f"test_dice: {test_dice:.4f}")
    print(f"test_iou: {test_iou:.4f}")


if __name__ == "__main__":
    main()
