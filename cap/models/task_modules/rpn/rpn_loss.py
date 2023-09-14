# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.core.detection_utils import rearrange_det_dense_head_out
from cap.registry import OBJECT_REGISTRY

__all__ = ["RPNSepLoss"]


@OBJECT_REGISTRY.register
class RPNSepLoss(nn.Module):
    """RPN Loss module with separate regression and classification loss.

    Compute classification loss and regression loss separately, keyed
    by 'rpn_cls_loss' and 'rpn_reg_loss' respectively. The type of
    each loss is specified in corresponding config.

    Args:
        cls_loss: classification loss module.
        reg_loss: regression loss module.
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
        self,
        head_out: Dict[str, List[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        reg_pred, cls_pred = rearrange_det_dense_head_out(
            [v.float() for v in head_out["rpn_reg_pred"]],
            [v.float() for v in head_out["rpn_cls_pred"]],
        )

        cls_loss = self.cls_loss(
            pred=cls_pred.flatten(end_dim=-2),
            target=targets["cls_label"].flatten(end_dim=-2),
            weight=targets["cls_label_mask"].flatten(end_dim=-2),
        )

        reg_label_mask = targets["reg_label_mask"].expand_as(
            targets["reg_label"]
        )
        reg_loss = self.reg_loss(
            pred=reg_pred.flatten(end_dim=-2),
            target=targets["reg_label"].flatten(end_dim=-2),
            weight=reg_label_mask.flatten(end_dim=-2),
            avg_factor=reg_label_mask.sum() + 1e-6,
        )

        return OrderedDict(rpn_cls_loss=cls_loss, rpn_reg_loss=reg_loss)
