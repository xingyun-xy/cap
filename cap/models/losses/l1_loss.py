from typing import Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.models.losses.utils import weight_reduce_loss
from cap.registry import OBJECT_REGISTRY

__all__ = ["L1Loss"]


@OBJECT_REGISTRY.register
class L1Loss(nn.Module):
    """Smooth L1 Loss.

    Args:
        beta: The threshold in the piecewise function. Defaults to 1.0.
        reduction: The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight: Loss weight.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
        loss_weight: Optional[float] = None,
    ):
        super(L1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    @autocast(enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[float, torch.Tensor]] = None,
    ):
        """Forward function.

        Args:
            pred: The prediction.
            target: The learning target of the prediction.
            weight: The weight of loss for each
                prediction. Defaults to None.
            avg_factor: Normalized factor.
        """
        # convert to float32 while using amp
        pred = pred.float()

        loss = torch.abs(pred - target)

        loss = weight_reduce_loss(
            loss=loss,
            weight=weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
        )

        if self.loss_weight is not None:
            loss *= self.loss_weight

        return loss
