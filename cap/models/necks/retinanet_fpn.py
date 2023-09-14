# Copyright (c) Changan Auto. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn

from cap.models.base_modules import ConvModule2d
from cap.models.necks.fpn import FPN
from cap.models.weight_init import xavier_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["RetinaNetFPN"]


@OBJECT_REGISTRY.register
class RetinaNetFPN(FPN):
    """FPN for RetinaNet.

    The difference with FPN is that RetinaNetFPN
    has two extra convs correspond to stride 64 and stride 128 except
    the lateral convs.

    Args:
        in_strides (list): strides of each input feature map
        in_channels (list): channels of each input feature map,
            the length of in_channels should be equal to in_strides
        out_strides (list): strides of each output feature map,
            should be a subset of in_strides, and continuous (any
            subsequence of 2, 4, 8, 16, 32, 64 ...). The largest
            stride in in_strides and out_strides should be equal
        out_channels (list): channels of each output feature maps
                the length of out_channels should be equal to out_strides
        fix_out_channel (:obj:`int`, optional): if set, there will be
            a 1x1 conv following each output feature map so that each
            final output has fix_out_channel channels
    """

    def __init__(
        self,
        in_strides: List[int],
        in_channels: List[int],
        out_strides: List[int],
        out_channels: List[int],
        fix_out_channel: Optional[int] = None,
    ):
        super(RetinaNetFPN, self).__init__(
            in_strides=in_strides,
            in_channels=in_channels,
            out_strides=out_strides,
            out_channels=out_channels,
            fix_out_channel=fix_out_channel,
        )
        self.extra_fpn_convs = nn.ModuleList()
        self.extra_fpn_convs.append(
            ConvModule2d(
                in_channels=in_channels[-1],
                out_channels=fix_out_channel,
                kernel_size=3,
                padding=1,
                stride=2,
                act_layer=nn.ReLU(inplace=True),
            )
        )
        self.extra_fpn_convs.append(
            ConvModule2d(
                in_channels=fix_out_channel,
                out_channels=fix_out_channel,
                kernel_size=3,
                padding=1,
                stride=2,
            )
        )
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        fpn_fuse = super(RetinaNetFPN, self).forward(features=features)
        fea_64 = self.extra_fpn_convs[0](features[-1])
        fea_128 = self.extra_fpn_convs[1](fea_64)
        fpn_fuse.append(fea_64)
        fpn_fuse.append(fea_128)
        return fpn_fuse

    def fuse_model(self):
        super(RetinaNetFPN, self).fuse_model()
        for module in self.extra_fpn_convs:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
