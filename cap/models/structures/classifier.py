# Copyright (c) Changan Auto. All rights reserved.

import logging

import torch.nn as nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["Classifier"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Classifier(nn.Module):
    """
    The basic structure of classifier.

    Args:
        backbone (torch.nn.Module): Backbone module.
        losses (torch.nn.Module): Losses module.
    """

    def __init__(self, backbone, losses=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.losses = losses

    def forward(self, data):
        image = data["img"]
        target = data.get("labels", None)

        preds = self.backbone(image)
        if target is None:
            return preds

        if not self.training or self.losses is None:
            return preds, target

        losses = self.losses(preds, target)
        return preds, losses

    def fuse_model(self):
        for module in [self.backbone, self.losses]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.backbone is not None:
            if hasattr(self.backbone, "set_qconfig"):
                self.backbone.set_qconfig()

        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from cap.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None
