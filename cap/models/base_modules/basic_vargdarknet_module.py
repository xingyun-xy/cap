# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict

import torch
import torch.nn as nn
from changan_plugin_pytorch import quantization
from torch.nn.quantized import FloatFunctional

from .conv_module import ConvModule2d
from .separable_conv_module import SeparableGroupConvModule2d

__all__ = ["VargDarkNetBlock"]


class VargDarkNetBlock(nn.Module):
    """
    A basic block for vargdarknet.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: Dict,
    ):
        super(VargDarkNetBlock, self).__init__()
        assert (
            in_channels == out_channels * 2
        ), f"{in_channels} != 2 * {out_channels}"
        self.conv1 = ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.conv2 = SeparableGroupConvModule2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=5,
            stride=1,
            padding=2,
            factor=1,
            groups=8,
            dw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            dw_act_layer=None,
            pw_norm_layer=nn.BatchNorm2d(out_channels * 2, **bn_kwargs),
            pw_act_layer=None,
        )
        self.relu = nn.ReLU(inplace=True)
        self.merge_add = FloatFunctional()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.merge_add.add(self.conv2(x), residual)
        x = self.relu(x)
        return x

    def fuse_model(self):
        self.conv1.fuse_model()
        getattr(self.conv2, "0").fuse_model()
        torch.quantization.fuse_modules(
            self,
            ["conv2.1.0", "conv2.1.1", "merge_add", "relu"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )
