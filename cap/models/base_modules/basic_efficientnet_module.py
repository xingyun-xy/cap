# Copyright (c) Changan Auto. All rights reserved.

import copy
import math

import torch
import torch.nn as nn
from torch.nn.quantized import FloatFunctional

from .conv_module import ConvModule2d

__all__ = ["SEBlock", "MBConvBlock"]


class SEBlock(nn.Module):
    """Basic Squeeze-and-Excitation block for EfficientNet.

    Args:
        in_channels (int): Input channels.
        num_squeezed_channels (int): Squeezed channels.
        out_channels (int): Output channels.
        act_layer (torch.nn.Module): Config dict for activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        num_squeezed_channels: int,
        out_channels: int,
        act_layer: torch.nn.Module,
    ):

        super(SEBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(1),
            ConvModule2d(
                in_channels=in_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
                bias=True,
                norm_layer=None,
                act_layer=act_layer,
            ),
            ConvModule2d(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
                norm_layer=None,
                act_layer=nn.Sigmoid(),
            ),
        )
        self.float_func = FloatFunctional()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.float_func.mul(x, inputs)

        return x

    def fuse_model(self):
        getattr(self.conv, "1").fuse_model()


class MBConvBlock(nn.Module):
    """Basic MBConvBlock for EfficientNet.

    Args:
        block_args (dict): Dict for block parameters.
        bn_kwargs (dict): Dict for Bn layer.
        act_layer (torch.nn.Module): activation layer.
        use_se_block (bool): Whether to use SEBlock in module.
    """

    def __init__(
        self,
        block_args: dict,
        bn_kwargs: dict,
        act_layer: torch.nn.Module,
        use_se_block: bool = False,
    ):
        super(MBConvBlock, self).__init__()

        self._block_args = block_args
        self.expand_ratio = self._block_args.expand_ratio
        self.in_planes = self._block_args.in_filters
        self.out_planes = self.in_planes * self.expand_ratio
        self.final_out_planes = self._block_args.out_filters

        self.kernel_size = self._block_args.kernel_size
        self.stride = self._block_args.strides
        self.id_skip = self._block_args.id_skip

        self.has_se = (
            use_se_block
            and (self._block_args.se_ratio is not None)
            and (0 < self._block_args.se_ratio <= 1)
        )

        if self.expand_ratio != 1:
            self._expand_conv = ConvModule2d(
                in_channels=self.in_planes,
                out_channels=self.out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(self.out_planes, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )

        self._depthwise_conv = ConvModule2d(
            in_channels=self.out_planes,
            out_channels=self.out_planes,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=math.ceil((self.kernel_size - 1) // 2)
            if self.stride == 1
            else math.ceil(self.kernel_size // self.stride),
            bias=False,
            groups=self.out_planes,
            norm_layer=nn.BatchNorm2d(self.out_planes, **bn_kwargs),
            act_layer=copy.deepcopy(act_layer),
        )

        if self.has_se:
            num_squeezed_num = max(
                1, int(self.in_planes * self._block_args.se_ratio)
            )
            self._se_block = SEBlock(
                in_channels=self.out_planes,
                num_squeezed_channels=num_squeezed_num,
                out_channels=self.out_planes,
                act_layer=copy.deepcopy(act_layer),
            )

        self._project_conv = ConvModule2d(
            in_channels=self.out_planes,
            out_channels=self.final_out_planes,
            kernel_size=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(self.final_out_planes, **bn_kwargs),
            act_layer=None,
        )

        self.use_shortcut = (
            self.id_skip
            and self.stride == 1
            and self.in_planes == self.final_out_planes
        )

        if self.use_shortcut:
            self.float_func = FloatFunctional()

    def forward(self, inputs):
        x = inputs

        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
        x = self._depthwise_conv(x)
        if self.has_se:
            x = self._se_block(x)
        x = self._project_conv(x)
        if self.use_shortcut:
            x = self.float_func.add(x, inputs)

        return x

    def fuse_model(self):
        # for qat mappings
        from changan_plugin_pytorch import quantization

        if hasattr(self, "_expand_conv"):
            getattr(self, "_expand_conv").fuse_model()
        if hasattr(self, "_se_block"):
            getattr(self, "_se_block").fuse_model()
        getattr(self, "_depthwise_conv").fuse_model()
        if self.use_shortcut:
            torch.quantization.fuse_modules(
                self,
                ["_project_conv.0", "_project_conv.1", "float_func"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
        else:
            getattr(self, "_project_conv").fuse_model()
