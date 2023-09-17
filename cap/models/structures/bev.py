# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, Optional

import torch.nn as nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["bev"]


@OBJECT_REGISTRY.register
class bev(nn.Module):
    """
    The basic structure of Camera3D task.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        desc: Description module.
        losses: Losses module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
    ):
        super(bev, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.desc = desc
        self.loss = loss

    def forward(self, data: Dict):

        img = data["img"]
        img_metas = data["img_metas_batch"]
        mats_dict = data["mats_dict"]
        features = self.backbone(img, mats_dict)
        # feat_maps = self.neck(features)
        preds = self.head(features)

        if self.loss is not None:
            out_dict.update(
                self.loss(
                    preds,
                    data["dimensions"],
                    data["location_offset"],
                    data["heatmap"],
                    data["heatmap_weight"],
                    data["depth"],
                    box2d_wh=data["box2d_wh"],
                    ignore_mask=data["ignore_mask"],
                    index_mask=data["index_mask"],
                    index=data["index"],
                    location=data["location"],
                    dimensions_=data["dimensions_"],
                    rotation_y=data["rotation_y"],
                )
            )

        return preds,img_metas
    def fuse_model(self):
        for module in self.children():
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        if self.loss is not None:
            self.loss.qconfig = None

        for module in self.children():
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
