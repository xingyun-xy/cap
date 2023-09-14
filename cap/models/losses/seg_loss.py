# Copyright (c) Changan Auto. All rights reserved.
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

__all__ = ["SegLoss", "MultiStrideLosses"]


@OBJECT_REGISTRY.register
class SegLoss(torch.nn.Module):
    """
    Segmentation loss wrapper.

    Args:
        loss (dict): loss config.

    Note:
        This class is not universe. Make sure you know this class limit before
        using it.

    """

    def __init__(
        self,
        loss: List[torch.nn.Module],
    ):
        super(SegLoss, self).__init__()
        self.loss = loss

    def forward(self, pred: Any, target: List[Dict]) -> Dict:
        # Since pred is resized in SegTarget, use target directly.
        assert len(target) == len(self.loss)
        res = [
            single_loss(**single_target)
            for single_target, single_loss in zip(target, self.loss)
        ]

        return res


@OBJECT_REGISTRY.register
class MixSegLoss(nn.Module):
    """Calculate multi-losses with same prediction and target.

    Args:
        losses: List of losses with the same input pred and target.
        losses_weight: List of weights used for loss calculation.
            Default: None

    """

    def __init__(
        self, losses: List[nn.Module], losses_weight: List[float] = None
    ):
        super(MixSegLoss, self).__init__()
        assert losses is not None
        self.losses_name = []
        self.losses = []
        self.loss_name = "mixsegloss"
        for loss in losses:
            self.losses.append(loss)
            self.losses_name.append(loss.loss_name)

        if losses_weight is None:
            losses_weight = [1.0 for i in range(len(losses))]
        self.losses_weight = losses_weight

    def forward(self, pred, target):
        losses_res = {}
        for idx, loss in enumerate(self.losses):
            loss_name = self.losses_name[idx]
            loss_val = loss(pred, target)[loss_name] * self.losses_weight[idx]
            losses_res[loss_name] = loss_val

        return losses_res


@OBJECT_REGISTRY.register
class MixSegLossMultipreds(MixSegLoss):
    """Calculate multi-losses with multi-preds and correspondence targets.

    Args:
        losses: List of losses with different prediction and target.
        losses_weight: List of weights used for loss calculation.
            Default: None
    """

    def __init__(
        self, losses: List[nn.Module], losses_weight: List[float] = None
    ):
        super(MixSegLossMultipreds, self).__init__(losses, losses_weight)
        self.loss_name = "multipredsloss"

    def forward(self, pred, target):
        assert isinstance(pred, List)
        losses_res = {}
        for idx, loss in enumerate(self.losses):
            loss_name = self.losses_name[idx]
            loss_val = loss(pred[idx], target[idx])
            if len(loss_val.keys()) > 1:
                for key, item in loss_val.items():
                    loss_val[key] = item * self.losses_weight[idx]
                losses_res.update(loss_val)
            else:
                loss_val_item = loss_val[loss_name] * self.losses_weight[idx]
                losses_res[loss_name] = loss_val_item

        return losses_res


@OBJECT_REGISTRY.register
class MultiStrideLosses(nn.Module):
    """Multiple Stride Losses.

    Apply the same loss function with different loss weights
    to multiple outputs.

    Args:
        num_classes: Number of classes.
        out_strides: strides of output feature maps
        loss: Loss module.
        loss_weights: Loss weight.
    """

    def __init__(
        self,
        num_classes: int,
        out_strides: List[int],
        loss: nn.Module,
        loss_weights: Optional[List[float]] = None,
    ):

        super().__init__()
        self.num_classes = num_classes
        if loss_weights is not None:
            assert len(loss_weights) == len(out_strides)
        else:
            loss_weights = [1.0] * len(out_strides)
        self.out_strides = out_strides
        self.loss = loss
        self.loss_weights = loss_weights

    @autocast(enabled=False)
    def forward(
        self,
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # convert to float32 while using amp
        preds = [pred.float() for pred in preds]
        assert (
            len(preds) == len(targets) == len(self.loss_weights)
        ), "%d vs. %d vs. %d" % (
            len(preds),
            len(targets),
            len(self.loss_weights),
        )

        targets, weights = self.slice_vanilla_labels(targets)

        losses = OrderedDict()

        for pred, target, weight, stride, loss_weight in zip(
            preds, targets, weights, self.out_strides, self.loss_weights
        ):

            losses[f"stride_{stride}_loss"] = (
                self.loss(pred, target, weight=weight) * loss_weight
            )

        return losses

    def slice_vanilla_labels(self, target):

        labels, weights = [], []

        # (N, 1, H, W) --> (N, 1, H, W, C) --> (N, C, H, W)
        for target_i in target:

            assert torch.all(target_i.abs() < self.num_classes)

            label_neg_mask = target_i < 0
            all_pos_label = target_i.detach().clone()

            # set neg label to positive to work around torch one_hot
            all_pos_label[label_neg_mask] *= -1

            target_i = F.one_hot(
                all_pos_label.type(torch.long), num_classes=self.num_classes
            )
            target_i[label_neg_mask] = -1

            target_i.squeeze_(axis=1)
            target_i = target_i.permute(0, 3, 1, 2)

            label_weight = target_i != -1

            labels.append(target_i)
            weights.append(label_weight)

        return labels, weights
