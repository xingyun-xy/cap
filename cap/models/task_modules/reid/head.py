# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict

import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.base_modules.conv_module import ConvModule2d
from cap.registry import OBJECT_REGISTRY
from cap.utils import qconfig_manager

__all__ = ["ReIDClsOutputBlock"]


@OBJECT_REGISTRY.register
class ReIDClsOutputBlock(nn.Module):
    """ReID classification output block.

    Args:
        num_classes: Number of categories excluding the
            background category.
        include_top: Whether to include top.
        bn_kwargs: Parameters for BN layer.
        in_channels: Number of channels in the input feature map.
        feat_channels: Number of hidden channels.
        pool_kernel_size: Kernel size of pooling.
        int8_output: If True, output int8, otherwise output int32.
    """

    def __init__(
        self,
        num_classes: int,
        include_top: bool,
        bn_kwargs: Dict,
        in_channels: int,
        feat_channels: int = 128,
        pool_kernel_size: int = 4,
        int8_output: bool = True,
    ):

        super(ReIDClsOutputBlock, self).__init__()

        self.int8_output = int8_output

        self.dequant = DeQuantStub()

        self.conv1 = ConvModule2d(
            in_channels=in_channels,
            out_channels=feat_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d(feat_channels, **bn_kwargs),
            act_layer=None,
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=pool_kernel_size,
            stride=1,
            padding=0,
            ceil_mode=False,
        )
        self.include_top = include_top
        if include_top:
            self.classifier = ConvModule2d(
                in_channels=feat_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=None,
                act_layer=None,
                bias=False,
            )

        self.num_classes = num_classes

    def forward(self, features):
        if isinstance(features, list):
            feat_t = self.conv1(features[-1])
        else:
            feat_t = self.conv1(features)
        feat_t = self.avg_pool(feat_t)
        if self.include_top:
            cls_pred = self.dequant(self.classifier(feat_t))
            cls_pred = cls_pred.reshape([-1, self.num_classes])
            return cls_pred
        else:
            return self.dequant(feat_t)

    def fuse_model(self):
        self.conv1.fuse_model()

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        # disable output quantization for last quanti layer.
        if not self.int8_output and self.include_top:
            self.classifier.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )  # noqa E501
