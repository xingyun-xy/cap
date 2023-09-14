# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

__all__ = ["RCNNKPSLoss"]


@OBJECT_REGISTRY.register
class RCNNKPSLoss(nn.Module):
    """RCNN KPS Loss.

    Args:
        kps_num: number of keypoints to be predicted.
        feat_height: the height of the output feature.
        feat_width: the width of the output feature.
        cls_loss: classification loss module.
        reg_loss: regression loss module.
    """

    def __init__(
        self,
        kps_num: int,
        feat_height: int,
        feat_width: int,
        cls_loss: nn.Module,
        reg_loss: nn.Module,
    ):
        super().__init__()
        self.kps_num = kps_num
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.feat_height = feat_height
        self.feat_width = feat_width

    @autocast(enabled=False)
    def forward(
        self, preds: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # convert to float32 while using amp
        for k, v in preds.items():
            preds[k] = v.float()

        cls_pred = preds["kps_rcnn_cls_pred"]
        reg_pred = preds["kps_rcnn_reg_pred"]

        cls_label = labels["kps_cls_label"].view(
            -1, self.kps_num, self.feat_height, self.feat_width
        )
        cls_weight = labels["kps_cls_label_weight"].view(
            -1, self.kps_num, self.feat_height, self.feat_width
        )

        reg_label = labels["kps_reg_label"].view(
            -1, self.kps_num * 2, self.feat_height, self.feat_width
        )
        reg_weight = labels["kps_reg_label_weight"].view(
            -1, self.kps_num * 2, self.feat_height, self.feat_width
        )

        kps_label_loss = self.cls_loss(
            pred=cls_pred,
            target=cls_label,
            weight=cls_weight,
            avg_factor=(cls_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )

        kps_pos_offset_loss = self.reg_loss(
            pred=reg_pred,
            target=reg_label,
            weight=reg_weight,
            avg_factor=(reg_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )

        output_loss = OrderedDict(
            rcnn_kps_class_loss=kps_label_loss,
            rcnn_kps_reg_loss=kps_pos_offset_loss,
        )

        return output_loss
