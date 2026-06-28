"""Evaluation metrics (Dice, IoU) reported on the held-out test split.

Numbers from these helpers feed the README results table and the blog. Report
them honestly on a patient-disjoint split (PRD §4.2) — do not fabricate.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean Dice over the batch for binary masks."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


@torch.no_grad()
def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean Intersection-over-Union over the batch for binary masks."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


@torch.no_grad()
def confusion_matrix_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """TP / FP / FN / TN pixel counts summed over the whole batch."""
    preds = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.float().view(-1)
    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


class MetricTracker:
    """Accumulates batch metrics into running averages for an epoch."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        bs = logits.size(0)
        self._count += bs
        self._sums["dice"] = self._sums.get("dice", 0.0) + dice_score(logits, targets).item() * bs
        self._sums["iou"] = self._sums.get("iou", 0.0) + iou_score(logits, targets).item() * bs

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {"dice": 0.0, "iou": 0.0}
        return {k: v / self._count for k, v in self._sums.items()}
