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


class FocalLoss(nn.Module):
    """Binary focal loss — down-weights easy negatives to focus on hard pixels.

    Args:
        alpha: class-balance weight for positive class.
        gamma: focusing exponent; 0 reduces to standard BCE.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """Dice + Focal combined loss for imbalanced binary segmentation.

    Args:
        focal_weight: weight of the Focal term (Dice gets ``1 - focal_weight``).
    """

    def __init__(self, focal_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_weight * self.focal(logits, targets)
            + (1.0 - self.focal_weight) * self.dice(logits, targets)
        )


def build_loss(name: str = "dice_bce", **kwargs) -> nn.Module:
    """Loss factory referenced by the training config."""
    name = name.lower()
    if name == "dice":
        return DiceLoss(**kwargs)
    if name == "dice_bce":
        return DiceBCELoss(**kwargs)
    if name == "focal":
        return FocalLoss(**kwargs)
    if name == "combined":
        return CombinedLoss(**kwargs)
    raise ValueError(f"unknown loss: {name!r}")
