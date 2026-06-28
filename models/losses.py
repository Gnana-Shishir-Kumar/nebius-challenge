"""Segmentation losses for EndoSeg.

Defaults target the binary lesion-vs-background MVP task. A combined
Dice + BCE loss is a robust starting point for class-imbalanced medical
segmentation; swap in MONAI losses for the 8-class stretch goal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary masks (logits in, scalar out)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Weighted sum of Dice and BCE-with-logits.

    Args:
        bce_weight: relative weight of the BCE term.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


def build_loss(name: str = "dice_bce", **kwargs) -> nn.Module:
    """Loss factory referenced by the training config."""
    name = name.lower()
    if name == "dice":
        return DiceLoss(**kwargs)
    if name == "dice_bce":
        return DiceBCELoss(**kwargs)
    raise ValueError(f"unknown loss: {name!r}")
