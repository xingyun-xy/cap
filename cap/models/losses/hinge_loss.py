# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.models.base_modules.loss_hard_neg_mining import LossHardNegativeMining
from cap.registry import OBJECT_REGISTRY
from .utils import weight_reduce_loss

__all__ = [
    "ElementwiseL1HingeLoss",
    "ElementwiseL2HingeLoss",
    "WeightedSquaredHingeLoss",
]


class _ElementwiseHingeLoss(nn.Module):
    """Elementwise Hinge Loss.

    Args:
        loss_bound_l1: Upper bound of l1 loss value in each entry.
        pos_label: Value in label that represents positive entries.
        neg_label: Value in label that represents negative entries.
        norm_type: Normalization method, can be "positive_label_elt",
            in which normalization factor is the number of positive elements,
            or "positive_loss_elt", the number of positive losses.
        loss_weight: Global weight of loss. Defaults is 1.0.
        reduction: The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`]. By default and recommended to be 'mean'.
        hard_neg_mining_cfg: hard negative mining config. Please refer
            to LossHardNegativeMining.
        l2: Whether to use l2 loss. Default value is False.

    Returns:
        torch.Tensor: loss value
    """

    def __init__(
        self,
        loss_bound_l1: float = 0.0,
        pos_label: int = 1,
        neg_label: int = 0,
        norm_type: str = "positive_label_elt",
        loss_weight: float = 1.0,
        reduction: Optional[str] = None,
        hard_neg_mining_cfg: Optional[Dict] = None,
        l2: bool = False,
    ):
        super().__init__()
        assert neg_label < pos_label
        assert norm_type in {
            "positive_label_elt",
            "positive_loss_elt",
            "batch",
        }
        self.loss_bound_l1 = loss_bound_l1
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.norm_type = norm_type
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.hard_neg = (
            LossHardNegativeMining(**hard_neg_mining_cfg)
            if hard_neg_mining_cfg is not None
            else None
        )
        self.l2 = l2

    def _get_norm(self, loss, target):
        if self.norm_type == "positive_label_elt":
            return torch.clamp_min_((target >= self.pos_label).sum(), 1)
        elif self.norm_type == "positive_loss_elt":
            return torch.clamp_min_((loss >= 0).sum(), 1)
        else:
            return loss.shape[0]

    @autocast(enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward.

        Args:
            pred (torch.Tensor): pred tensor with any shape
            target (torch.Tensor): target tensor, the same shape as pred
            weight (torch.Tensor): The weight of loss for each prediction.
                Default is None.
            avg_factor (float): Normalized factor.
        """
        # convert to float32 while using amp
        pred = pred.float()

        diff = pred - target

        new_t = target.clone()

        # manipulate label value to limit loss and grad
        if self.loss_bound_l1 > 0:
            new_t[diff > self.loss_bound_l1] = (
                pred[diff > self.loss_bound_l1] - self.loss_bound_l1
            )
            new_t[diff < -self.loss_bound_l1] = (
                pred[diff < -self.loss_bound_l1] + self.loss_bound_l1
            )

        pos_mask = target >= self.pos_label
        zero_mask = torch.logical_and(
            target < self.pos_label, target > self.neg_label
        )
        loss = pred - new_t.detach()
        loss[pos_mask] *= -1
        loss[zero_mask] *= 0
        loss[loss < 0] *= 0

        if self.l2:
            loss = loss ** 2 * 0.5

        if self.hard_neg is not None:
            type_mask = torch.ones_like(loss) * self.hard_neg.POSITIVE
            type_mask[target <= self.neg_label] = self.hard_neg.NEGATIVE
            if weight is not None:
                type_mask[weight == 0] = self.hard_neg.IGNORE
            loss_mask = self.hard_neg(loss, type_mask)
            if weight is not None:
                weight = weight * loss_mask
            else:
                weight = loss_mask

        norm_val = self._get_norm(loss, target)

        loss = weight_reduce_loss(loss, weight, self.reduction, norm_val)

        return self.loss_weight * loss


@OBJECT_REGISTRY.register
class ElementwiseL1HingeLoss(_ElementwiseHingeLoss):
    """Elementwise L1 Hinge Loss.

    Args:
        loss_bound_l1: Upper bound of l1 loss value in each entry.
        pos_label: Value in label that represents positive entries.
        neg_label: Value in label that represents negative entries.
        norm_type: Normalization method, can be "positive_label_elt",
            in which normalization factor is the number of positive elements,
            or "positive_loss_elt", the number of positive losses.
        loss_weight: Global weight of loss. Defaults is 1.0.
        reduction: The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`]. By default and recommended to be 'mean'.
        hard_neg_mining_cfg: hard negative mining config. Please refer
            to LossHardNegativeMining.

    Returns:
        torch.Tensor: loss value
    """

    def __init__(
        self,
        loss_bound_l1: float = 0.0,
        pos_label: int = 1,
        neg_label: int = 0,
        norm_type: str = "positive_label_elt",
        loss_weight: float = 1.0,
        reduction: Optional[str] = None,
        hard_neg_mining_cfg: Optional[Dict] = None,
    ):
        super().__init__(
            loss_bound_l1=loss_bound_l1,
            pos_label=pos_label,
            neg_label=neg_label,
            norm_type=norm_type,
            loss_weight=loss_weight,
            reduction=reduction,
            hard_neg_mining_cfg=hard_neg_mining_cfg,
            l2=False,
        )


@OBJECT_REGISTRY.register
class ElementwiseL2HingeLoss(_ElementwiseHingeLoss):
    """Elementwise L2 Hinge Loss.

    Args:
        loss_bound_l1: Upper bound of l1 loss value in each entry.
        pos_label: Value in label that represents positive entries.
        neg_label: Value in label that represents negative entries.
        norm_type: Normalization method, can be "positive_label_elt",
            in which normalization factor is the number of positive elements,
            or "positive_loss_elt", the number of positive losses.
        loss_weight: Global weight of loss. Defaults is 1.0.
        reduction: The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`]. By default and recommended to be 'mean'.
        hard_neg_mining_cfg: hard negative mining config. Please refer
            to LossHardNegativeMining.

    Returns:
        torch.Tensor: loss value
    """

    def __init__(
        self,
        loss_bound_l1: float = 0.0,
        pos_label: int = 1,
        neg_label: int = 0,
        norm_type: str = "positive_label_elt",
        loss_weight: float = 1.0,
        reduction: Optional[str] = None,
        hard_neg_mining_cfg: Optional[Dict] = None,
    ):
        super().__init__(
            loss_bound_l1=loss_bound_l1,
            pos_label=pos_label,
            neg_label=neg_label,
            norm_type=norm_type,
            loss_weight=loss_weight,
            reduction=reduction,
            hard_neg_mining_cfg=hard_neg_mining_cfg,
            l2=True,
        )


@OBJECT_REGISTRY.register
class WeightedSquaredHingeLoss(nn.Module):
    """Weighted Squared ElementWiseHingeLoss.

    Args:
        reduction (str): Possible values are {'mean', 'sum',
            'sum_mean', 'none'}
        loss_weight (float): by default 1.0
        weight_low_thr (float): Lower threshold for elementwise weight,
            by default 0.1
        weight_high_thr (float): Upper threshold for pixel-wise weight,
            by default 1.0
        hard_neg_mining_cfg (dict): Hard negative mining cfg
    """

    def __init__(
        self,
        reduction: str,
        loss_weight: float = 1.0,
        weight_low_thr: float = 0.1,
        weight_high_thr: float = 1.0,
        hard_neg_mining_cfg: Optional[Dict] = None,
    ):
        assert weight_low_thr > 0 and weight_high_thr <= 1
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.weight_low_thr = weight_low_thr
        self.weight_high_thr = weight_high_thr

        self.hard_neg = (
            None
            if hard_neg_mining_cfg is None
            else LossHardNegativeMining(**hard_neg_mining_cfg)
        )

    @autocast(enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):

        # convert to float32 while using amp
        pred = pred.float()

        target = target * weight

        batch_sum = target.sum((1, 2, 3), keepdim=True)
        channel_sum = target.sum((2, 3), keepdim=True)

        weight_matrix = (batch_sum - channel_sum + 1) / (batch_sum + 1)

        weight_matrix = (target * weight_matrix).sum(1, keepdims=True)

        weight_matrix.clamp_(self.weight_low_thr, self.weight_high_thr)

        loss = (pred - target) * (
            (target <= 0).float() - (target >= 1).float()
        )

        loss[loss < 0] = 0
        loss = 0.5 * loss ** 2

        if self.hard_neg is not None:
            type_mask = torch.ones_like(loss) * self.hard_neg.POSITIVE
            type_mask[target <= 0] = self.hard_neg.NEGATIVE
            if weight is not None:
                type_mask[weight == 0] = self.hard_neg.IGNORE
            loss_mask = self.hard_neg(loss, type_mask)
            if weight is not None:
                weight = weight * loss_mask
            else:
                weight = loss_mask

        loss.mul_(weight_matrix * weight)

        weight_matrix = weight_matrix.mul(loss > 0)

        scale = weight_matrix.sum((1, 2, 3))
        scale[scale <= 0] = 1

        loss = loss.sum((1, 2, 3)) / scale

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise NotImplementedError

        return loss * self.loss_weight
