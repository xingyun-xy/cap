# Copyright (c) Changan Auto. All rights reserved.

import itertools
import warnings
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY
from .utils import weight_reduce_loss

__all__ = ["FocalLoss", "FocalLossV2", "SoftmaxFocalLoss"]


@OBJECT_REGISTRY.register
class FocalLoss(nn.Module):
    """Sigmoid focal loss.

    Args:
        loss_name (str): The key of loss in return dict.
        num_classes (int): Num_classes including background, C+1, C is number
            of foreground categories.
        alpha (float): A weighting factor for pos-sample, (1-alpha) is for
            neg-sample.
        gamma (float): Gamma used in focal loss to compress the contribution
            of easy examples.
        loss_weight (float): Global weight of loss. Defaults is 1.0.
        eps (float): A small value to avoid zero denominator.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        dict: A dict containing the calculated loss, the key of loss is
        loss_name.
    """

    def __init__(
        self,
        loss_name,
        num_classes,
        alpha=0.25,
        gamma=2.0,
        loss_weight=1.0,
        eps=1e-12,
        reduction="mean",
    ):
        super(FocalLoss, self).__init__()
        warnings.warn(
            "FocalLoss will be deprecated, please use FocalLossV2 instead",
            DeprecationWarning,
        )
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_name = loss_name

    @autocast(enabled=False)
    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        points_per_strides=None,
        valid_classes_list=None,
    ):
        """
        Forward method.

        Args:
            pred (Tensor): Cls pred, with shape(N, C), C is num_classes of
                foreground.
            target (Tensor): Cls target, with shape(N,), values in [0, C-1]
                represent the foreground, C or negative value represent the
                background.
            weight (Tensor): The weight of loss for each prediction. Default
                is None.
            avg_factor (float): Normalized factor.
        """
        # cast to fp32
        pred = pred.float().sigmoid()
        target[target < 0] = self.num_classes - 1
        one_hot = F.one_hot(target, self.num_classes)  # N x C+1
        one_hot = one_hot[..., : self.num_classes - 1]  # N x C
        pt = torch.where(torch.eq(one_hot, 1), pred, 1 - pred)
        t = torch.ones_like(one_hot)
        at = torch.where(
            torch.eq(one_hot, 1), self.alpha * t, (1 - self.alpha) * t
        )
        loss = (
            -at
            * torch.pow((1 - pt), self.gamma)
            * torch.log(torch.minimum(pt + self.eps, t))
        )  # noqa

        # for two datasets use same head, apply mask on loss
        if valid_classes_list is not None and loss.shape[-1] > 1:
            valid_loss_mask = torch.zeros_like(loss)
            # fmt: off
            start_indexs = [0, ] + list(
                itertools.accumulate(
                    [xx * len(valid_classes_list) for xx in points_per_strides]
                )
            )[:-1]
            # fmt: on
            for str_id, points_per_stride in enumerate(points_per_strides):
                for ii, valid_classes in enumerate(valid_classes_list):
                    start_idx = (
                        start_indexs[str_id] + ii * points_per_stride
                    )  # noqa
                    end_idx = (
                        start_indexs[str_id] + (ii + 1) * points_per_stride
                    )  # noqa
                    valid_loss_mask[start_idx:end_idx][
                        :, valid_classes
                    ] = 1.0  # noqa
            loss = loss * valid_loss_mask

        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    # For most cases, weight is of shape
                    # (num_priors, ), which means it does not have
                    # the second axis num_class
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim

        loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)

        result_dict = {}
        result_dict[self.loss_name] = loss * self.loss_weight
        return result_dict


# TODO (kongtao.hu, 0.5): FocalLoss should be merged with FocalLossV2
@OBJECT_REGISTRY.register
class FocalLossV2(nn.Module):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        alpha: A weighting factor for pos-sample, (1-alpha) is for
            neg-sample.
        gamma: Gamma used in focal loss to compress the contribution
            of easy examples.
        eps: A small value to avoid zero denominator.
        from_logits: Whether the input prediction is logits (before sigmoid).
        reduction: The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-12,
        from_logits: bool = True,
        reduction: str = "mean",
    ):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.from_logits = from_logits
        self.reduction = reduction

    @autocast(enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[float, torch.Tensor]] = None,
    ):
        """
        Forward method.

        Args:
            pred: cls pred, with shape (B, N, C), C is num_classes of
                foreground.
            target: cls target, with shape (B, N, C), C is num_classes
                of foreground.
            weight: The weight of loss for each prediction. It is
                mainly used to filter the ignored box. Default is None.
            avg_factor: Normalized factor.
        """
        # convert to float32 while using amp
        pred = pred.float()

        if not self.from_logits:
            pred = pred.sigmoid()
        pt = torch.where(torch.eq(target, 1), pred, 1 - pred)
        t = torch.ones_like(target)
        at = torch.where(
            torch.eq(target, 1), self.alpha * t, (1 - self.alpha) * t
        )
        loss = (
            -at
            * torch.pow((1 - pt), self.gamma)
            * torch.log(torch.clip_(pt, min=self.eps, max=1.0))
        )
        if weight is not None:
            assert pred.shape == weight.shape

        loss = weight_reduce_loss(
            loss, weight, self.reduction, avg_factor=avg_factor
        )
        return loss


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("ClassificationFocalLoss")
class SoftmaxFocalLoss(torch.nn.Module):
    r"""
    Focal Loss.

    Args:
        loss_name (str): The key of loss in return dict.
        num_classes (int): Class number.
        alpha (float, optional): Alpha. Defaults to 0.25.
        gamma (float, optional): Gamma. Defaults to 2.0.
        reduction (str, optional):
            Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        weight (Union[float, Sequence], optional): Weight to be applied to
            the loss of each input. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_name: str,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: Union[float, Sequence] = 1.0,
    ):
        super(SoftmaxFocalLoss, self).__init__()
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def focal_loss(self, logits, label):
        one_hot = label

        probs = torch.softmax(logits, dim=1)
        probs = probs + 1e-9
        ce = (-torch.log(probs)) * one_hot
        weight = torch.pow(1.0 - probs, self.gamma)

        loss = ce * weight * self.alpha
        loss = loss.sum(dim=1)

        return weight_reduce_loss(loss, reduction=self.reduction)

    def forward(self, logits, labels):
        if isinstance(logits, Sequence):
            if not isinstance(self.weight, Sequence):
                weights = [self.weight] * len(logits)
            else:
                weights = self.weight

            return {
                self.loss_name: sum(
                    [
                        self.focal_loss(logit, label) * w
                        for logit, label, w in zip(logits, labels, weights)
                    ]
                )
            }
        else:
            return {
                self.loss_name: self.focal_loss(logits, labels) * self.weight
            }


@OBJECT_REGISTRY.register
class GaussianFocalLoss(nn.Module):
    """Guassian focal loss.

    Args:
        alpha: A weighting factor for positive sample.
        gamma: Used in focal loss to balance contribution
                of easy examples and hard examples.
        loss_weight: Weight factor for guassian focal loss.
    """

    def __init__(
        self, alpha: float = 2.0, gamma: float = 4.0, loss_weight: float = 1.0
    ):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def gaussian_focal_loss(self, pred, gaussian_target, grad_tensor):
        eps = 1e-10
        # should not use mean as other losses
        pos_pos = gaussian_target.eq(1)
        neg_pos = gaussian_target.lt(1)
        neg_weights = (1 - gaussian_target).pow(self.gamma)
        pos_loss = (
            -(pred + eps).log()
            * (1 - pred).pow(self.alpha)
            * pos_pos
            * grad_tensor
        )
        neg_loss = (
            -(1 - pred + eps).log()
            * pred.pow(self.alpha)
            * neg_weights
            * neg_pos
            * grad_tensor
        )
        nums_pos = pos_pos.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        loss = neg_loss if nums_pos == 0 else (pos_loss + neg_loss) / nums_pos
        return loss

    def forward(self, logits, labels, grad_tensor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
        """
        if grad_tensor is None:
            grad_tensor = torch.ones_like(logits)
        loss_reg = self.loss_weight * self.gaussian_focal_loss(
            logits, labels, grad_tensor
        )
        return loss_reg
