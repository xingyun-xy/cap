# Copyright (c) Changan Auto. All rights reserved.

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

__all__ = [
    "HMFocalLoss",
    "HML1Loss",
]


@OBJECT_REGISTRY.register
class HMFocalLoss(nn.Module):
    """Calculate heatmap focal loss.

    Args:
        alpha: A weighting factor for pos-sample,default 2.0
        beta: Beta used in heatmap focal loss to compress
            the contribution of easy examples, default 4.0
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super(HMFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    @autocast(enabled=False)
    def forward(
        self,
        pred: torch.tensor,
        target: torch.tensor,
        ignore_mask: torch.tensor,
    ):
        # convert to float32 while using amp
        pred = pred.float()

        pos_mask = (target == 1).float() * (1.0 - ignore_mask)
        neg_mask = (target < 1).float() * (1.0 - ignore_mask)

        neg_weights = torch.pow(1 - target, self.beta)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_mask
        neg_loss = neg_loss * neg_weights

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        loss = -(pos_loss + neg_loss) / num_pos

        return loss


@OBJECT_REGISTRY.register
class HML1Loss(nn.Module):
    """Calculate heatmap L1 loss.

    Args:
        heatmap_type: Type of heatmap.
    """

    def __init__(self, heatmap_type: str):
        super(HML1Loss, self).__init__()
        self.heatmap_type = heatmap_type

    @autocast(enabled=False)
    def forward(
        self,
        pred: torch.tensor,
        target: torch.tensor,
        weight_mask: torch.tensor,
        ignore_mask: torch.tensor = None,
    ):
        if self.heatmap_type == "dense":
            weight_mask = weight_mask.gt(0).float()
        elif self.heatmap_type == "point":
            weight_mask = weight_mask.eq(1).float()

        if ignore_mask is not None:
            weight_mask = weight_mask * (1.0 - ignore_mask).expand(
                target.shape
            )

        eps = 1e-6
        loss = (torch.abs(pred - target) * weight_mask).sum()
        loss = loss / (weight_mask.sum() + eps)

        return loss
