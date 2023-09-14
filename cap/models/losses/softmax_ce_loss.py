from typing import Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.models.losses.utils import weight_reduce_loss
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class SoftmaxCELoss(nn.Module):
    """Softmax Cross Entropy Loss.

    Args:
        dim: A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).
        loss_weight: If present, it will be used to weight the output.
        reduction: The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
    """

    def __init__(self, dim=1, reduction="mean", loss_weight=1.0):
        super(SoftmaxCELoss, self).__init__()
        assert dim >= 0
        self.dim = dim
        self.loss_weight = loss_weight
        self.reduction = reduction

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
            weight: The weight of loss for each prediction. Defaults to None.
            avg_factor: Normalized factor. Defaults to None.
        """
        pred = pred.log_softmax(dim=self.dim)

        if pred.shape == target.shape:
            loss = -pred * target
        else:
            new_dim_order = (
                list(range(self.dim))
                + list(range(self.dim + 1, pred.ndim))
                + [self.dim]
            )
            num_classes = pred.shape[self.dim]

            pred = pred.permute(new_dim_order).view(-1, num_classes)
            target = target.flatten()
            target[target < 0] = 0
            weight = weight.flatten()

            loss = -target[
                torch.arange(len(target), device=target.device), target
            ]

        loss = (
            weight_reduce_loss(
                loss=loss,
                weight=weight,
                reduction=self.reduction,
                avg_factor=avg_factor,
            )
            * self.loss_weight
        )

        return loss
