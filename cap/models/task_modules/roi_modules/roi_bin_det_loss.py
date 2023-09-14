# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

__all__ = ["RCNNBinDetLoss", "RCNNMultiBinDetLoss"]


@OBJECT_REGISTRY.register
class RCNNBinDetLoss(nn.Module):
    """RCNN bin detection loss.

    Bin detection is the detection task in the bins. This
    loss module compute classification loss and regression
    loss separately, keyed by 'label_map_loss' and 'offset_loss'
    respectively. The type of each loss is specified in
    corresponding config.

    Args:
        cls_loss: classification loss module.
        reg_loss: regression loss module.
        thresh: a threshold of label_map.
    """

    def __init__(
        self,
        cls_loss: nn.Module,
        reg_loss: nn.Module,
        thresh: float = 0.3,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self._thresh = thresh

    @autocast(enabled=False)
    def forward(
        self, pred: Dict[str, torch.tensor], img_meta: Dict[str, torch.tensor]
    ):
        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()

        label_map_pred = pred["rcnn_cls_pred"]
        offset_pred = pred["rcnn_reg_pred"]
        label_map = img_meta["label_map"]
        mask = img_meta["mask"]
        offset = img_meta["offset"]

        assert label_map_pred.shape == label_map.shape
        assert offset_pred.shape == offset.shape

        label_map_loss = self.cls_loss(
            pred=label_map_pred,
            target=label_map,
            weight=mask[..., None, None].expand(label_map.shape),
        )

        offset_loss_weight = (label_map > self._thresh).sum(
            dim=1, keepdim=True
        ).expand(offset.shape) * mask.sum(dim=1, keepdim=True)[
            ..., None, None
        ].expand(
            offset.shape
        )

        offset_loss = self.reg_loss(
            pred=offset_pred,
            target=offset,
            weight=offset_loss_weight,
            avg_factor=(offset_loss_weight.sum(dim=[1, 2, 3]) > 0).sum()
            + 1e-6,
        )

        output_loss = OrderedDict(
            label_map_loss=label_map_loss, offset_loss=offset_loss
        )

        return output_loss


@OBJECT_REGISTRY.register
class RCNNMultiBinDetLoss(nn.Module):
    """RCNN multi bin detection loss.

    Bin detection is the detection task in the bins. This
    loss module compute classification loss and regression
    loss separately, keyed by 'label_map_loss' and 'offset_loss'
    respectively. The type of each loss is specified in
    corresponding config.

    Args:
        cls_loss: Classification loss module.
        reg_loss: Regression loss module.
    """

    def __init__(
        self,
        cls_loss: nn.Module,
        reg_loss: nn.Module,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss

    @autocast(enabled=False)
    def forward(
        self, pred: Dict[str, torch.tensor], target: Dict[str, torch.tensor]
    ):
        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()

        label_map_pred = pred["rcnn_cls_pred"]
        offset_pred = pred["rcnn_reg_pred"]
        label_map = target["label_map"]
        label_mask = target["label_mask"]
        offset = target["offset"]
        offset_mask = target["offset_mask"]

        assert label_map_pred.shape == label_map.shape
        assert offset_pred.shape == offset.shape

        label_map_loss = self.cls_loss(
            pred=label_map_pred,
            target=label_map,
            weight=label_mask,
        )

        offset_loss = self.reg_loss(
            pred=offset_pred,
            target=offset,
            weight=offset_mask,
            avg_factor=(offset_mask.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )

        output_loss = OrderedDict(
            label_map_loss=label_map_loss, offset_loss=offset_loss
        )

        return output_loss
