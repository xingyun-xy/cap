# Copyright (c) Changan Auto. All rights reserved.
from typing import Optional

import torch.nn as nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["YOLOV3"]


@OBJECT_REGISTRY.register
class YOLOV3(nn.Module):
    """
    The basic structure of yolov3.

    Args:
        backbone (torch.nn.Module): Backbone module.
        neck (torch.nn.Module): Neck module.
        head (torch.nn.Module): Head module.
        anchor_generator (torch.nn.Module): Anchor generator module.
        target_generator (torch.nn.Module): Target generator module.
        loss (torch.nn.Module): Loss module.
        postprocess (torch.nn.Module): Postprocess module.
    """

    def __init__(
        self,
        backbone: Optional[dict] = None,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        anchor_generator: Optional[dict] = None,
        target_generator: Optional[dict] = None,
        loss: Optional[dict] = None,
        postprocess: Optional[dict] = None,
    ):
        super(YOLOV3, self).__init__()

        names = [
            "backbone",
            "neck",
            "head",
            "anchor_generator",
            "target_generator",
            "loss",
            "postprocess",
        ]
        modules = [
            backbone,
            neck,
            head,
            anchor_generator,
            target_generator,
            loss,
            postprocess,
        ]
        for name, module in zip(names, modules):
            if module is not None:
                setattr(self, name, module)
            else:
                setattr(self, name, None)

    def forward(self, data):
        image = data["img"]

        x = self.backbone(image)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)

        if not self.training:
            if self.postprocess is not None:
                anchors = self.anchor_generator(x)
                x = self.postprocess(x, anchors)
                assert (
                    "pred_bboxes" not in data.keys()
                ), "pred_bboxes has been in data.keys()"
                data["pred_bboxes"] = x
                return data
            else:
                return x
        else:
            anchors = self.anchor_generator(x)
            targets = self.target_generator(anchors, data["gt_labels"])
            losses = self.loss(x, targets[1])
            return losses, x

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
