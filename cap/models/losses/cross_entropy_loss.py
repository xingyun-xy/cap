# -*- coding: utf-8 -*-
# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection, gluonnas

from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from capbc.utils import deprecated
from torch.cuda.amp import autocast

from cap.models.base_modules.loss_hard_neg_mining import LossHardNegativeMining
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .utils import weight_reduce_loss

__all__ = [
    "CrossEntropyLoss",
    "CrossEntropyLossV2",
    "CEWithLabelSmooth",
    "SoftTargetCrossEntropy",
    "CEWithHardMining",
]


def binary_cross_entropy(
    pred, target, weight, reduction, avg_factor, class_weight=None, **kwargs
):  # noqa: D205,D400
    """Measure binary cross entropy between target and pred
    logits, then apply element-wise weight and reduce loss.

    Args:
        pred (Tensor): Logits Tensor of arbitrary shape.
        target (Tensor): Tensor of the same shape as pred.
        weight (Tensor): Element-wise weight loss weight.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
        avg_factor (float): Average factor that is used to average the loss.
        class_weight (Tensor, optional): Weight of each class, must be a vector
            with length equal to the number of classes.
    """
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target.float(), pos_weight=class_weight, reduction="none"
    )
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


def cross_entropy(
    pred,
    target,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
):  # noqa: D205,D400
    """Calculate the cross entropy loss using F.cross_entropy,
    then apply element-wise weight and reduce loss.

    Args:
        pred (torch.Tensor): Same as F.cross_entropy, Logits Tensor.
        target (torch.Tensor): Same as F.cross_entropy.
        weight (torch.Tensor): Element-wise weight loss weight.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
        avg_factor (float): Average factor that is used to average the loss.
        class_weight : a manual rescaling weight given to each class. If given,
            has to be a Tensor of size `C`.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        target.long(),
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
        loss = loss * weight
        if weight.sum() == 0:
            return loss.sum()
        loss = loss.sum() / weight.sum()
    else:
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss


@OBJECT_REGISTRY.register
class CrossEntropyLoss(nn.Module):
    """Calculate cross entropy loss of multi stride output.

    This class will depracated in future, use CrossEntropyLossV2 instead.

    Args:
        loss_name (str): The key of loss in return dict.
        preds_name (str): The key of pred in pred dict.
        label_name (str): The key of label in target dict.
        weight_name (str): The key of weight in target dict.
        avg_factor_name (str): The key of avg_factor in target dict.
        use_sigmoid (bool): Whether logits tensor is converted to probability
            through sigmoid, Defaults to False.
            If `True`, use `F.binary_cross_entropy_with_logits`.
            If `False`, use `F.cross_entropy`.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
        class_weight (list[float]): Weight of each class. Defaults is None.
        loss_weight (float): Global weight of loss. Defaults is 1.
        ignore_index (int): Only works when using cross_entropy.

    Returns:
        dict: A dict containing the calculated loss, the key of loss is
        loss_name.

    """

    @deprecated("Please use `CrossEntropyLossV2` instead")
    def __init__(
        self,
        loss_name,
        preds_name,
        label_name,
        weight_name="weight",
        avg_factor_name="avg_factor",
        use_sigmoid=False,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.preds_name = preds_name
        self.label_name = label_name
        self.loss_name = loss_name
        self.weight_name = weight_name
        self.avg_factor_name = avg_factor_name
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    @autocast(enabled=False)
    def forward(self, pred_dict: Mapping, target_dict: Mapping):
        preds = _as_list(pred_dict[self.preds_name])
        label = target_dict[self.label_name]

        weight = target_dict.get(self.weight_name, None)
        avg_factor = target_dict.get(self.avg_factor_name, None)

        if self.class_weight is not None:
            class_weight = preds[0].new_tensor(self.class_weight)
            class_weight = class_weight.float()
        else:
            class_weight = None

        loss_cls = 0.0
        for pred in preds:
            loss_cls += self.loss_weight * self.cls_criterion(
                pred.float(),
                label,
                weight,
                class_weight=class_weight,
                reduction=self.reduction,
                avg_factor=avg_factor,
                ignore_index=self.ignore_index,
            )

        result_dict = {}
        result_dict[self.loss_name] = loss_cls
        return result_dict


@OBJECT_REGISTRY.register
class CrossEntropyLossV2(nn.Module):
    """Calculate cross entropy loss of multi stride output.

    Args:
        use_sigmoid (bool): Whether logits tensor is converted to probability
            through sigmoid, Defaults to False.
            If `True`, use `F.binary_cross_entropy_with_logits`.
            If `False`, use `F.cross_entropy`.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
        class_weight (list[float]): Weight of each class. Defaults is None.
        loss_weight (float): Global weight of loss. Defaults is 1.
        ignore_index (int): Only works when using cross_entropy.
        loss_name (str): The key of loss in return dict. If None, return loss
            directly.

    Returns:
        cross entropy loss

    """

    def __init__(
        self,
        use_sigmoid: bool = False,
        reduction: str = "mean",
        class_weight: Optional[List[float]] = None,
        loss_weight: float = 1.0,
        ignore_index: int = -1,
        loss_name: Optional[str] = None,
        auto_class_weight: Optional[bool] = False,
        weight_min: Optional[float] = None,
        weight_noobj: Optional[float] = None,
        num_class: int = 0,
    ):
        super(CrossEntropyLossV2, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.loss_name = loss_name
        self.auto_class_weight = auto_class_weight
        self.num_class = num_class

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        if self.auto_class_weight:
            self.weight_min = weight_min
            self.weight_noobj = weight_noobj

    @autocast(enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):
        loss_cls = None
        object_num = torch.reshape(target, (-1,)).shape[0]
        if pred is not None:
            preds = _as_list(pred)
            if self.auto_class_weight:
                class_weight = []
                for i in range(self.num_class):
                    num = len(
                        torch.where(torch.reshape(target, (-1,)) == i)[0]
                    )
                    if num == 0:
                        class_weight.append(self.weight_noobj)
                    else:
                        class_weight.append(
                            round(
                                np.max(
                                    [1 - num / object_num, self.weight_min]
                                ),
                                2,
                            )
                        )
                class_weight = preds[0].new_tensor(class_weight)
                class_weight = class_weight.float()
            else:
                if self.class_weight is not None:
                    class_weight = preds[0].new_tensor(self.class_weight)
                    class_weight = class_weight.float()
                else:
                    class_weight = None
            loss_cls = 0
            for each_pred in preds:
                loss_cls += self.loss_weight * self.cls_criterion(
                    each_pred.float(),
                    target,
                    weight,
                    class_weight=class_weight,
                    reduction=self.reduction,
                    avg_factor=avg_factor,
                    ignore_index=self.ignore_index,
                )
        if self.loss_name is None:
            return loss_cls
        else:
            return {self.loss_name: loss_cls}


@OBJECT_REGISTRY.register
class CEWithLabelSmooth(torch.nn.CrossEntropyLoss):
    """
    The losses of cross-entropy with label smooth.

    Args:
        smooth_alpha (float): Alpha of label smooth.
    """

    def __init__(self, smooth_alpha=0.1):
        super(CEWithLabelSmooth, self).__init__()
        self.smooth_alpha = smooth_alpha

    def forward(self, input, target):
        n = input.size()[-1]
        log_pred = F.log_softmax(input, dim=-1)
        loss = -log_pred.sum(dim=-1).mean()
        nll = F.nll_loss(log_pred, target, reduction="mean")
        sa = self.smooth_alpha
        return sa * (loss / n) + (1 - sa) * nll


@OBJECT_REGISTRY.register
class SoftTargetCrossEntropy(torch.nn.Module):
    """
    The losses of cross-entropy with soft target.

    Args:
        loss_name (str): The name of returned losses.
    """

    def __init__(self, loss_name=None):
        super(SoftTargetCrossEntropy, self).__init__()
        self.loss_name = loss_name

    def forward(self, input, target):
        if target.ndim == 1:
            loss_cls = F.cross_entropy(input, target)
        else:
            loss = torch.sum(-target * F.log_softmax(input, dim=-1), dim=-1)
            loss_cls = loss.mean()
        if self.loss_name is None:
            return loss_cls
        else:
            return {self.loss_name: loss_cls}


@OBJECT_REGISTRY.register
class CEWithWeightMap(CrossEntropyLossV2):
    """Crossentropy loss with image-specfic class weighted map within batch."""

    def __init__(self, weight_min=0.5, **kwargs):
        super().__init__(**kwargs)
        self.weight_min = weight_min

    def forward(self, pred, target, weight=None, avg_factor=None):
        if weight is None:
            weight = torch.zeros_like(target)
            weight = weight.float()
            batch_size, h, w = target.shape[:3]
            target_reshape = target.view(batch_size, -1)
            target_sum = (
                (target_reshape != self.ignore_index).sum(dim=-1).float()
            )
            min_area_ratio = torch.full_like(target_sum, self.weight_min)
            for cls_idx in range(self.num_class):
                sum_area = (target_reshape == cls_idx).sum(dim=-1)
                if sum_area.sum() > 0:
                    area_ratio = torch.max(
                        1.0 - sum_area / target_sum, min_area_ratio
                    )
                    area_ratio = area_ratio.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    area_ratio = area_ratio.repeat(1, h, w)
                    target_cls_idx = target == cls_idx
                    weight[target_cls_idx] = area_ratio[target_cls_idx]

        loss_cls = self.loss_weight * self.cls_criterion(
            pred.float(),
            target,
            weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index,
        )

        result_dict = {}
        result_dict[self.loss_name] = loss_cls
        return result_dict


@OBJECT_REGISTRY.register
class CEWithHardMining(nn.Module):
    """CE loss with online hard negative mining and auto average factor.

    Args:
        use_sigmoid: Whether logits tensor is converted to probability
            through sigmoid, Defaults to False.
            If `True`, use `F.binary_cross_entropy_with_logits`.
            If `False`, use `F.cross_entropy`.
        ignore_index: Specifies a target value that is ignored
            and does not contribute to the loss.
        norm_type: Normalization method, can be "fg_elt",
            in which normalization factor is the number of foreground elements,
            "fbg_elt" the number of foreground and background element.
            "none" no normalize on loss.
            Defaults to "none".
        reduction: The method used to reduce the loss. Options are
            ["none", "mean", "sum"].
            Default to "mean".
        loss_weight: Global weight of loss. Defaults is 1.0.
        class_weight: Weight of each class. If given must be a vector
            with length equal to the number of classes.
            Default to None.
        hard_neg_mining_cfg: hard negative mining config. Please refer
            to LossHardNegativeMining.

    """

    def __init__(
        self,
        use_sigmoid: bool = False,
        ignore_index: int = -1,
        norm_type: str = "none",
        reduction: str = "mean",
        loss_weight: float = 1.0,
        class_weight: Optional[torch.Tensor] = None,
        hard_neg_mining_cfg: Optional[Dict] = None,
    ):
        super(CEWithHardMining, self).__init__()

        assert reduction in [
            "mean",
            "sum",
            "none",
        ], f"{reduction} is not implement"

        assert norm_type in [
            "fg_elt",
            "fbg_elt",
            "none",
        ], f"{norm_type} is not implement"

        self.use_sigmoid = use_sigmoid
        self.ignore_index = ignore_index
        self.norm_type = norm_type
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.hard_neg = (
            LossHardNegativeMining(**hard_neg_mining_cfg)
            if hard_neg_mining_cfg is not None
            else None
        )

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def _get_norm(self, target, weight):
        """Get norm value. Background label in target is 0."""
        if self.norm_type == "fg_elt":
            return torch.clamp_min_(((target * weight) > 0).sum(), 1)
        elif self.norm_type == "fbg_elt":
            return torch.clamp_min_((weight > 0).sum(), 1)
        else:
            return None

    @autocast(enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
            class_weight = class_weight.float()
        else:
            class_weight = None

        loss = self.cls_criterion(
            pred.float(),
            target,
            weight=None,
            class_weight=class_weight,
            reduction="none",
            avg_factor=None,
            ignore_index=self.ignore_index,
        )

        if self.hard_neg is not None:
            type_mask = torch.ones_like(loss) * self.hard_neg.POSITIVE
            type_mask[target <= 0] = self.hard_neg.NEGATIVE
            type_mask[target == self.ignore_index] = self.hard_neg.IGNORE
            if weight is not None:
                type_mask[weight == 0] = self.hard_neg.IGNORE
            loss_mask = self.hard_neg(loss, type_mask)
            if weight is not None:
                weight = weight * loss_mask
            else:
                weight = loss_mask

        if self.norm_type != "none":
            avg_factor = self._get_norm(target, weight)
        else:
            avg_factor = "none"

        loss = self.loss_weight * weight_reduce_loss(
            loss, weight, self.reduction, avg_factor
        )

        return loss
