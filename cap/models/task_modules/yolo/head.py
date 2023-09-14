# Copyright (c) Changan Auto. All rights reserved.

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d
from cap.models.weight_init import xavier_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["YOLOV3Head"]


@OBJECT_REGISTRY.register
class YOLOV3Head(nn.Module):
    """
    Heads module of yolov3.

    shared convs -> conv head (include all objs)

    Args:
        in_channels_list (list): List of input channels.
        feature_idx (list): Index of feature for head.
        num_classes (int): Num classes of detection object.
        anchors (list): Anchors for all feature maps.
        bn_kwargs (dict): Config dict for BN layer.
        bias (bool): Whether to use bias in module.
    """

    def __init__(
        self,
        in_channels_list: list,
        feature_idx: list,
        num_classes: int,
        anchors: list,
        bn_kwargs: dict,
        bias: bool = True,
    ):
        super(YOLOV3Head, self).__init__()
        assert len(in_channels_list) == len(feature_idx)
        assert len(feature_idx) == len(anchors)
        self.num_classes = num_classes
        self.anchors = anchors

        self.feature_idx = feature_idx
        for i, anchor, in_channels in zip(
            range(len(anchors)), anchors, in_channels_list
        ):
            num_anchor = len(anchor)
            self.add_module(
                "head%d" % (i),
                ConvModule2d(
                    in_channels,
                    num_anchor * (num_classes + 5),
                    1,
                    bias=bias,
                ),
            )
            self.add_module("dequant%d" % (i), DeQuantStub())
        self.init_weight()

    def init_weight(self):
        for i, _idx in enumerate(self.feature_idx):
            for m in getattr(self, "head%d" % (i)):
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution="uniform")

    def forward(self, x):
        output = []
        for i, idx, anchor in zip(
            range(len(self.feature_idx)), self.feature_idx, self.anchors
        ):
            out = getattr(self, "head%d" % (i))(x[idx])
            out = getattr(self, "dequant%d" % (i))(out)
            if self.training:
                bs, _, h, w = out.size()
                num_anchor = len(anchor)
                out = (
                    out.view(bs, num_anchor, 5 + self.num_classes, h, w)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                out = out.view(bs, -1, 5 + self.num_classes).contiguous()
                out[..., 0:2] = torch.sigmoid(out[..., 0:2])
                out[..., 4:] = torch.sigmoid(out[..., 4:])
            output.append(out)

        if self.training:
            output = torch.cat(output, axis=1)
        return output

    def fuse_model(self):
        for i in range(len(self.feature_idx)):
            getattr(self, "head%d" % (i)).fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_out_qconfig()
