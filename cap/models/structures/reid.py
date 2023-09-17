# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Any, Dict, Optional

import changan_plugin_pytorch as changan
import torch
from torch import nn

from cap.registry import OBJECT_REGISTRY
from cap.utils import qconfig_manager

__all__ = ["ReIDModule"]


@OBJECT_REGISTRY.register
class ReIDModule(nn.Module):
    """
    The Structure of ReID task.

    Args:
        backbone: Backbone Module.
        head: Head Module.
        neck: Neck Module.
        desc: Description module.
        losses: Loss Module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        desc: Optional[nn.Module] = None,
        losses: Optional[nn.Module] = None,
    ):
        super(ReIDModule, self).__init__()
        self.backbone = backbone
        self.head = head
        self.neck = neck
        self.losses = losses
        self.desc = desc

    def forward(self, inputs: Dict[str, Any]):
        features = self.backbone(inputs["img"])
        features = self.neck(features)
        feature = self.head(features)
        if self.losses is not None and "labels" in inputs:
            cls_loss = self.losses(feature, inputs["labels"])
            output = OrderedDict(
                cls_loss=cls_loss,
                cls_softmax_output=torch.softmax(feature, dim=1),
            )
        else:
            output = feature

        if self.desc is not None:
            output = self.desc(output)

        return output

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        self.qconfig = changan.quantization.get_default_calib_qconfig()

        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_calibration_qconfig"):
                module.set_calibration_qconfig()

        if self.losses is not None:
            self.losses.qconfig = None
