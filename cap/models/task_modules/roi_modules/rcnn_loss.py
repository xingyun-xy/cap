# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

__all__ = ["RCNNLoss", "RCNNCLSLoss"]


@OBJECT_REGISTRY.register
class RCNNLoss(nn.Module):
    """RCNN Loss module with separate regression and classification loss.

    Compute classification loss and regression loss separately, keyed
    by 'rcnn_cls_loss' and 'rcnn_reg_loss' respectively. The type of
    each loss is specified in corresponding config.

    Args:
        class_agnostic_reg: Determines whether to use the
            class_agnostic way for box reg.
        cls_loss: classification loss module.
        reg_loss: regression loss module.
    """

    def __init__(
        self,
        class_agnostic_reg: bool = True,
        cls_loss: Optional[nn.Module] = None,
        reg_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.class_agnostic_reg = class_agnostic_reg

        assert not (cls_loss is None and reg_loss is None)
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss

    @autocast(enabled=False)
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # convert to float32 while using amp
        for k, v in preds.items():
            preds[k] = v.float()

        out_dict = OrderedDict()

        if self.cls_loss is not None:
            cls_pred = preds["rcnn_cls_pred"]
            if cls_pred.ndim == 4:
                cls_pred = cls_pred[:, :, 0, 0]
            cls_label = labels["cls_label"].view(cls_pred.shape[0], -1)
            cls_label_mask = labels["cls_label_mask"].view_as(cls_label)
            cls_loss = self.cls_loss(
                pred=cls_pred,
                target=cls_label,
                weight=cls_label_mask,
                avg_factor=cls_label_mask.sum() + 1e-6,
            )
            out_dict["rcnn_cls_loss"] = cls_loss

        if self.reg_loss is not None:
            reg_pred = preds["rcnn_reg_pred"]
            if reg_pred.ndim == 4:
                reg_pred = reg_pred[:, :, 0, 0]
            reg_label = labels["reg_label"].view_as(reg_pred)
            reg_label_mask = labels["reg_label_mask"].view_as(reg_pred)

            if self.class_agnostic_reg:
                reg_loss = self.reg_loss(
                    pred=reg_pred,
                    target=reg_label,
                    weight=reg_label_mask,
                    avg_factor=reg_label_mask.sum() + 1e-6,
                )
            else:
                raise NotImplementedError  # TODO: add the other case

            out_dict["rcnn_reg_loss"] = reg_loss

        return out_dict


@OBJECT_REGISTRY.register
class RCNNCLSLoss(nn.Module):
    """RCNN Loss module with only classification loss.

    Only Compute classification loss, keyed by 'rcnn_cls_loss'.
    The type of cls loss is specified in corresponding config.

    Args:
        cls_loss: classification loss module.
    """

    def __init__(self, cls_loss: nn.Module):
        super().__init__()
        self.cls_loss = cls_loss

    @autocast(enabled=False)
    def forward(
        self, preds: Dict[str, torch.tensor], labels: Dict[str, torch.tensor]
    ):

        cls_pred = preds["rcnn_cls_pred"].float()
        if cls_pred.ndim == 4:
            cls_pred = cls_pred[:, :, 0, 0]
        cls_label = labels["cls_label"].reshape(cls_pred.shape[0], -1)

        cls_label_mask = (cls_label.sum(dim=1, keepdim=True) > 0).float()

        cls_loss = self.cls_loss(
            pred=cls_pred,
            target=cls_label,
            weight=cls_label_mask,
            avg_factor=cls_label_mask.sum() + 1e-6,
        )

        return OrderedDict(rcnn_cls_loss=cls_loss)
