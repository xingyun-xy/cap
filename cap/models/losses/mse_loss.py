# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY
from .utils import weight_reduce_loss

__all__ = ["MSELoss"]


@OBJECT_REGISTRY.register
class MSELoss(nn.Module):
    """MSE (mean squared error) loss with clip value.

    Args:
        clip_val: Clip value. If present, it is used to constrain
            the unweighted loss value between (-clip_val, clip_val).
            For the clipped entries, the gradient is calculated as
            if label value equals to predication +- clip_val.
        reduction: Reduction mode.
        loss_weight: If present, it will be used to weight the output.

    """

    def __init__(
        self,
        clip_val: Optional[float] = None,
        reduction: Optional[str] = None,
        loss_weight: Optional[float] = None,
    ):
        super().__init__()
        if clip_val is not None:
            assert clip_val > 0
        self.clip_val = clip_val
        self.reduction = reduction
        self.loss_weight = loss_weight

    @autocast(enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:

        # convert to float32 while using amp
        pred = pred.float()
        diff = pred - target

        new_t = target.clone()

        # manipulate label value to limit the loss and grad to clip_val
        if self.clip_val is not None:
            new_t[diff > self.clip_val] = (
                pred[diff > self.clip_val] - self.clip_val
            )
            new_t[diff < (-self.clip_val)] = (
                pred[diff < (-self.clip_val)] + self.clip_val
            )
        loss = (pred - new_t.detach()) ** 2 * 0.5

        loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)

        if self.loss_weight is not None:
            loss *= self.loss_weight

        return loss
