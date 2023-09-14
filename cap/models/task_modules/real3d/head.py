# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict, List, Optional, Union

import changan_plugin_pytorch as changan
import torch.nn as nn
from torch import Tensor
from torch.quantization import DeQuantStub

from cap.models.base_modules import (
    BasicVarGBlock,
    ConvModule2d,
    SeparableConvModule2d,
)
from cap.models.utils import _take_features
from cap.models.weight_init import bias_init_with_prob, normal_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["Real3DHead"]


@OBJECT_REGISTRY.register
class Real3DHead(nn.Module):
    """
    Real3DHead module.

    Args:
        in_strides (list of int): A list contains the strides of feature maps
            from backbone or neck.
        out_strides (list of int): A list contains the strides of this head
            will output.
        in_channels (dict): A list of to indicates the input channels of the
            block.
        feature_channels (dict): A dictinary which key:value is stride:channel
            pair.
        head_channels (dict): A dictionary contains output heads and
            corresponding channels.
        sep_conv (bool): Use separable convolution.
        use_varg (bool): Use VarGNet block.
        stack (int): The number of VarGNet black stacked together.
        use_bias (bool): Use bias.
        bn_kwargs (dict): Batch norm arguments.
        dw_with_relu (bool): A param for VarGNet block.
        pw_with_relu (bool): A param for VarGNet block.
        factor (int): A param for VarGNet block.
        group_base (int): A param for VarGNet block.
        interpolate_kwargs (dict): Interpolate arguments.
    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        in_channels: List[int],
        head_channels: Dict,
        feature_channels: Dict = None,
        sep_conv: bool = False,
        use_varg: bool = True,
        stack: int = 1,
        use_bias: bool = False,
        bn_kwargs: Optional[Dict] = None,
        dw_with_relu: bool = True,
        pw_with_relu: bool = False,
        factor: int = 2,
        group_base: int = 8,
        interpolate_kwargs: dict = None,
    ):
        super(Real3DHead, self).__init__()
        self.in_strides = in_strides
        self.out_strides = out_strides
        self.use_bias = use_bias
        self.group_base = group_base
        self.factor = factor
        self.dw_with_relu = dw_with_relu
        self.pw_with_relu = pw_with_relu
        self.bn_kwargs = (
            bn_kwargs
            if bn_kwargs is not None
            else {"eps": 1e-5, "momentum": 0.1}
        )
        self.sep_conv = sep_conv
        self.stack = stack
        self.use_varg = use_varg
        self.dequant = DeQuantStub()
        self.interpolate_kwargs = interpolate_kwargs
        self.feature_channels = feature_channels
        if self.feature_channels:
            feature_channel = self.feature_channels[self.out_strides[0]]
            self.do_align = not feature_channel == in_channels
        else:
            self.do_align = False
        if self.do_align:
            self.channel_align_layers = ConvModule2d(
                feature_channel,
                in_channels,
                1,
                bias=True,
                norm_layer=nn.BatchNorm2d(in_channels),
                act_layer=nn.ReLU(inplace=True),
            )
        # TODO(runzhou.ge, 0.5): replace this block with MBConv
        if not self.use_varg:
            self.head_block = SeparableConvModule2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                pw_norm_layer=nn.BatchNorm2d(in_channels),
                pw_act_layer=nn.ReLU(inplace=True),
            )
        else:
            self.head_block = BasicVarGBlock(
                in_channels=in_channels,
                mid_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                bn_kwargs=self.bn_kwargs,
                factor=self.factor,
                group_base=self.group_base,
                merge_branch=False,
                dw_with_relu=self.dw_with_relu,
                pw_with_relu=self.pw_with_relu,
            )

        self.out_block_names = []
        for name, num_channel in head_channels.items():
            if self.sep_conv:
                block = []
                for _i in range(self.stack):
                    sep_block = SeparableConvModule2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        pw_norm_layer=nn.BatchNorm2d(in_channels),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                    block.append(sep_block)
                block.append(
                    nn.Conv2d(
                        in_channels,
                        num_channel,
                        1,
                        1,
                        0,
                        groups=1,
                        bias=use_bias,
                    )
                )
                block = nn.Sequential(*block)
            else:
                block = nn.Sequential(
                    ConvModule2d(
                        in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_layer=nn.BatchNorm2d(in_channels),
                        act_layer=nn.ReLU(inplace=True),
                    ),
                    nn.Conv2d(
                        in_channels,
                        num_channel,
                        1,
                        1,
                        0,
                        groups=1,
                        bias="hm" in name,  # hard code here
                    ),
                )
            # hard code here
            if "hm" in name:
                bias = bias_init_with_prob(0.01)
                normal_init(block[-1], std=0.01, bias=bias)
            setattr(self, "out_block_{}".format(name), block)
            self.out_block_names += [name]
        # TODO(min.du, runzhou.ge, 0.5): No conv initilized. It was trained
        # well on j5_2d3d branch. After checking in all the codes. I will try
        # to add initialization.

        if self.interpolate_kwargs:
            self.interpolate_layer = changan.nn.Interpolate(
                **self.interpolate_kwargs
            )

    def forward(self, data: Union[Dict, List]) -> Dict[str, Tensor]:
        """Forward head layers."""
        input_features = data["feats"] if isinstance(data, Mapping) else data

        feat = _take_features(
            input_features, self.in_strides, self.out_strides
        )[0]
        if self.do_align:
            feat = self.channel_align_layers(feat)
        if self.interpolate_kwargs:
            feat = self.interpolate_layer(feat)
        feat = self.head_block(feat)

        out = OrderedDict()
        for name in self.out_block_names:
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            out[name] = self.dequant(block(feat))
        return out

    def fuse_model(self):
        if self.do_align:
            self.channel_align_layers.fuse_model()
        self.head_block.fuse_model()
        for name in self.out_block_names:
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            assert isinstance(block, nn.Sequential)
            for block_i in block:
                if hasattr(block_i, "fuse_model"):
                    block_i.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        for name in self.out_block_names:
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            block[1].qconfig = qconfig_manager.get_default_qat_out_qconfig()
