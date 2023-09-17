# Copyright (c) Changan Auto. All rights reserved.

import logging
from collections import OrderedDict
from typing import Optional

import torch.nn as nn
import torch

from cap.registry import OBJECT_REGISTRY

__all__ = ["Segmentor"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Segmentor(nn.Module):
    """
    The basic structure of segmentor.

    Args:
        backbone (torch.nn.Module): Backbone module.
        neck (torch.nn.Module): Neck module.
        head (torch.nn.Module): Head module.
        losses (torch.nn.Module): Losses module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        target: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
    ):
        super(Segmentor, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.target = target
        self.loss = loss
        self.desc = desc
        self.postprocess = postprocess

    def forward(self, data: dict):

        # N,C,H,W = data["img"].shape
        # data["img"] = data["img"].view(-1,1,N,C,H,W)
        # img =  data["img"]

        # batch_size, num_sweeps, num_cams, num_channels, imH, imW = img.shape
        # if torch.onnx.is_in_onnx_export(): ###################
        #     imgs = img.flatten().view(batch_size * num_cams, num_channels, imH, imW)    
        # else:
        #     imgs = img.flatten().view(batch_size * num_sweeps * num_cams, num_channels, imH, imW)

        feat = self.backbone(data["img"])
        # feat = self.backbone(imgs)
        feat = self.neck(feat)
        pred = self.head(feat)

        if self.desc is not None:
            pred = self.desc(pred)

        output = OrderedDict(pred=pred)

        if self.loss is not None:
            target = (
                data["labels"] if self.target is None else self.target(data)
            )
            output.update(self.loss(pred, target))

        if self.postprocess is not None:
            return self.postprocess(pred)

        return output

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        if self.losses is not None:
            self.losses.qconfig = None
