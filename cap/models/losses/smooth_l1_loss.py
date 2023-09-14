from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.models.base_modules import LossHardNegativeMining
from cap.models.losses.utils import weight_reduce_loss
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss.

    Args:
        beta: The threshold in the piecewise function.
            Defaults to 1.0.
        reduction: The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight: Loss weight.
        hard_neg_mining_cfg: Hard negative mining cfg.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
        loss_weight: Optional[float] = None,
        hard_neg_mining_cfg: Optional[Dict] = None,
    ):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.hard_neg = (
            None
            if hard_neg_mining_cfg is None
            else LossHardNegativeMining(**hard_neg_mining_cfg)
        )

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
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff * diff / self.beta,
            diff - 0.5 * self.beta,
        )

        if self.hard_neg is not None:
            assert (
                weight is not None
            ), "`weight` can not be None when use `hard_neg`"  # noqa E501
            type_mask = torch.ones_like(weight) * self.hard_neg.NEGATIVE
            type_mask[weight == 0] = self.hard_neg.IGNORE
            type_mask[
                torch.logical_and(weight > 0, target > 0)
            ] = self.hard_neg.POSITIVE

            mining_mask = self.hard_neg(loss, type_mask)
            if weight is None:
                weight = mining_mask
            else:
                weight = weight * mining_mask

            if avg_factor is None:
                avg_factor = (weight.sum(dim=(1, 2, 3)) > 0).sum() + 1e-6

        loss = weight_reduce_loss(
            loss=loss,
            weight=weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
        )

        if self.loss_weight:
            loss *= self.loss_weight
        return loss
