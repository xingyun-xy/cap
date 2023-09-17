# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Dict

from torch import nn

from cap.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FCOS(nn.Module):
    """The basic structure of fcos.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target module.
        loss_cls: Classification loss module.
        loss_reg: Regiression loss module.
        loss_centerness: Centerness loss module
        postprocess: Postprocess module.

    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        post_process: nn.Module = None,
        loss_cls: nn.Module = None,
        loss_reg: nn.Module = None,
        loss_centerness: nn.Module = None,
    ):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.loss_centerness = loss_centerness

        self.targets = targets
        self.post_process = post_process

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, data: Dict):
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        preds = self.head(feats)

        if self.training:
            cls_targets, reg_targets, centerness_targets = self.targets(
                data, preds
            )

            cls_loss = self.loss_cls(**cls_targets)
            reg_loss = self.loss_reg(**reg_targets)
            centerness_loss = self.loss_centerness(**centerness_targets)
            return dict(**cls_loss, **reg_loss, **centerness_loss)
        else:
            if self.post_process is None:
                return preds
            results = self.post_process(preds, data)
            return results

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
