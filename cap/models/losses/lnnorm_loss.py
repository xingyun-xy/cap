# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Union

import torch
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY
from .utils import weight_reduce_loss

__all__ = ["LnNormLoss"]


@OBJECT_REGISTRY.register
class LnNormLoss(nn.Module):
    """LnNorm loss.

    Different from torch.nn.L1Loss, the loss function uses Ln norm
    to calculate the distance of two feature maps.

    Args:
        norm_order: The order of norm.
        epsilon: A small constant for finetune.
        power: A power num of norm + epsilon of loss.
        reduction: Reduction mode.
        loss_weight: If present, it will be used to weight the output.
    """

    def __init__(
        self,
        norm_order: int = 2,
        epsilon: float = 0.0,
        power: float = 1.0,
        reduction: Optional[str] = None,
        loss_weight: Optional[float] = None,
    ):
        super().__init__()
        self.norm_order = norm_order
        self.epsilon = epsilon
        self.power = power
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward method.

        Args:
            pred (Tensor): Optical flow pred, with shape(N, 2, H, W).
            target (Tensor): Optical flow target, with shape(N, 2, H, W),
            which obtained by ground truth sampling.
            weight (Tensor): The weight of loss for each prediction. Default
                is None.
            avg_factor (float): Normalized factor.
        """

        diff = pred - target.detach()

        loss = torch.norm(diff, p=self.norm_order, dim=1)
        loss = loss.sum((1, 2))
        loss = loss.mean()
        loss = loss + self.epsilon
        loss = torch.pow(loss, self.power)

        loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)

        if self.loss_weight is not None:
            loss *= self.loss_weight

        return loss
