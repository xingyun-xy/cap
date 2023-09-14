# Copyright (c) Changan Auto. All rights reserved.

import torch
import torch.nn as nn

from .conv_module import ConvModule2d

__all__ = ["BasicResBlock", "BottleNeck"]


class BasicResBlock(nn.Module):
    """
    A basic block for resnet.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        bn_kwargs (dict): Dict for Bn layer.
        stride (int): Stride for first conv.
        bias (bool): Whether to use bias in module.
        expansion (int): Expansion of channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: dict,
        stride: int = 1,
        bias: bool = True,
        expansion: int = 1,
    ):
        super(BasicResBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                stride=stride,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                stride=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            ),
        )

        self.downsample = None
        if not (stride == 1 and in_channels == out_channels):
            self.downsample = ConvModule2d(
                in_channels,
                out_channels,
                1,
                stride=stride,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )
        self.relu = nn.ReLU(inplace=True)
        self.short_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.short_add.add(out, identity)
        out = self.relu(out)
        return out

    def fuse_model(self):
        # for qat mappings
        from changan_plugin_pytorch import quantization

        if self.downsample is not None:
            self.downsample.fuse_model()
        getattr(self.conv, "0").fuse_model()
        torch.quantization.fuse_modules(
            self,
            ["conv.1.0", "conv.1.1", "short_add", "relu"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class BottleNeck(nn.Module):
    """
    A bottle neck for resnet.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        bn_kwargs (dict): Dict for Bn layer.
        stride (int): Stride for first conv.
        bias (bool): Whether to use bias in module.
        expansion (int): Expansion of channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: dict,
        stride: int = 1,
        bias: bool = True,
        expansion: int = 4,
    ):
        super(BottleNeck, self).__init__()
        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels,
                out_channels,
                1,
                padding=0,
                stride=stride,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                stride=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                out_channels,
                out_channels * expansion,
                1,
                padding=0,
                stride=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(
                    out_channels * expansion, **bn_kwargs
                ),
            ),
        )

        self.downsample = None
        if not (stride == 1 and in_channels == out_channels * expansion):
            self.downsample = ConvModule2d(
                in_channels,
                out_channels * expansion,
                1,
                stride=stride,
                norm_layer=nn.BatchNorm2d(
                    out_channels * expansion, **bn_kwargs
                ),
                act_layer=nn.ReLU(inplace=True),
            )

        self.relu = nn.ReLU(inplace=True)
        self.short_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.short_add.add(out, identity)
        out = self.relu(out)
        return out

    def fuse_model(self):
        # for qat mappings
        from changan_plugin_pytorch import quantization

        if self.downsample is not None:
            self.downsample.fuse_model()
        getattr(self.conv, "0").fuse_model()
        getattr(self.conv, "1").fuse_model()
        torch.quantization.fuse_modules(
            self,
            ["conv.2.0", "conv.2.1", "short_add", "relu"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )
