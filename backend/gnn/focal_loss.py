from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WeightedFocalLoss(nn.Module):
    """Binary focal loss with a positive-class weight and stable logit arithmetic."""

    def __init__(
        self, alpha: float = 0.75, gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        if not 0 <= alpha <= 1 or not 0 <= gamma < float("inf") or reduction not in {"none", "mean", "sum"}:
            raise ValueError("Use 0 ≤ alpha ≤ 1, finite gamma ≥ 0, and a valid reduction.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape != targets.shape or not logits.numel():
            raise ValueError("Logits and binary targets must have the same non-empty shape.")
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        if not ((targets == 0) | (targets == 1)).all():
            raise ValueError("Targets must be binary.")
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        # Clamp the base at machine precision so fractional gamma has a finite gradient at p_t = 1.
        focus = (-torch.expm1(-bce)).clamp_min(torch.finfo(bce.dtype).tiny).pow(self.gamma)
        loss = weights * focus * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
