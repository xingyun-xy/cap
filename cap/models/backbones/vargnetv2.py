# Copyright (c) Changan Auto. All rights reserved.
from collections.abc import Sequence
from typing import Dict, List, Optional, Union

import changan_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import (
    BasicVarGBlock,
    ConvModule2d,
    ExtendVarGNetFeatures,
)
from cap.registry import OBJECT_REGISTRY

__all__ = ["VargNetV2", "get_vargnetv2_stride2channels", "TinyVargNetV2"]


@OBJECT_REGISTRY.register
class VargNetV2(nn.Module):
    """
    A module of vargnetv2.

    Args:
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        model_type (str): Choose to use `VargNetV2` or `TinyVargNetV2`.
        alpha (float): Alpha for vargnetv2.
        group_base (int): Group base for vargnetv2.
        factor (int): Factor for channel expansion in basic block.
        bias (bool): Whether to use bias in module.
        extend_features (bool): Whether to extend features.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
        input_channels (int): Input channels of first conv.
        input_sequence_length (int): Length of input sequence.
        head_factor (int): Factor for channels expansion of stage1(mod2).
        input_resize_scale (int): Narrow_model need resize input 0.65 scale,
            While int_infer or visualize or eval
    """

    def __init__(
        self,
        num_classes,
        bn_kwargs: dict,
        model_type: str = "VargNetV2",
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        bias: bool = True,
        extend_features: bool = False,
        disable_quanti_input: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
        input_channels: int = 3,
        input_sequence_length: int = 1,
        head_factor: int = 1,
        input_resize_scale: int = None,
    ):
        super(VargNetV2, self).__init__()
        self.model_type = model_type.lower()
        assert self.model_type in ["vargnetv2", "tinyvargnetv2"], (
            f"`model_type` should be one of ['vargnetv2', 'tinyvargnetv2'],"
            f" but get {model_type}."
        )
        self.group_base = group_base
        self.factor = factor
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.extend_features = extend_features
        self.include_top = include_top
        self.disable_quanti_input = disable_quanti_input
        self.flat_output = flat_output
        self.input_sequence_length = input_sequence_length
        self.head_factor = head_factor
        assert self.head_factor in [1, 2], "head_factor should be 1 or 2"
        self.input_resize_scale = input_resize_scale
        if self.input_resize_scale is not None:
            assert self.input_resize_scale > 0
        channel_list = [32, 32, 64, 128, 256]
        if self.model_type == "tinyvargnetv2":
            units = [1, 3, 4, 2]
        else:
            units = [1, 3, 7, 4]
        channel_list = [int(chls * alpha) for chls in channel_list]

        self.quant = QuantStub()
        if input_sequence_length > 1:
            self.cat_op = nn.quantized.FloatFunctional()
            for i in range(1, input_sequence_length):
                setattr(self, f"extra_quant_{i}", QuantStub())
        self.dequant = DeQuantStub()

        if self.input_resize_scale is not None:
            self.resize = hnn.Interpolate(
                scale_factor=input_resize_scale,
                align_corners=None,
                recompute_scale_factor=True,
            )

        self.in_channels = channel_list[0]
        self.mod1 = ConvModule2d(
            input_channels * input_sequence_length,
            channel_list[0],
            3,
            stride=2,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(channel_list[0], **bn_kwargs),
            act_layer=nn.ReLU(inplace=True)
            if self.model_type == "tinyvargnetv2"
            else None,
        )

        head_factor = 2 if self.head_factor == 2 else 8 // group_base
        self.mod2 = self._make_stage(
            channel_list[1], units[0], head_factor, False
        )
        self.mod3 = self._make_stage(channel_list[2], units[1], self.factor)
        self.mod4 = self._make_stage(channel_list[3], units[2], self.factor)
        self.mod5 = self._make_stage(channel_list[4], units[3], self.factor)

        if extend_features:
            self.ext = ExtendVarGNetFeatures(
                prev_channel=channel_list[-1],
                channels=channel_list[-1],
                num_units=2,
                group_base=group_base,
                bn_kwargs=bn_kwargs,
            )

        if self.include_top:
            self.output = nn.Sequential(
                ConvModule2d(
                    channel_list[-1],
                    max(int(alpha * 1024), 1024),
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(
                        max(int(alpha * 1024), 1024), **bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
                nn.AvgPool2d(7),
                ConvModule2d(
                    max(int(alpha * 1024), 1024),
                    num_classes,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(self, channels, repeats, factor, merge_branch=True):
        layers = []
        layers.append(
            BasicVarGBlock(
                self.in_channels,
                channels,
                channels,
                2,
                bias=self.bias,
                bn_kwargs=self.bn_kwargs,
                factor=factor,
                group_base=self.group_base,
                merge_branch=merge_branch,
            )
        )

        self.in_channels = channels
        for _ in range(1, repeats):
            layers.append(
                BasicVarGBlock(
                    channels,
                    channels,
                    channels,
                    1,
                    bias=self.bias,
                    bn_kwargs=self.bn_kwargs,
                    factor=factor,
                    group_base=self.group_base,
                    merge_branch=False,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.input_sequence_length > 1:
            x = self.process_sequence_input(x)
        else:
            if isinstance(x, Sequence) and len(x) == 1:
                x = x[0]
            x = x if self.disable_quanti_input else self.quant(x)
        if self.input_resize_scale is not None:
            x = self.resize(x)
        output = []
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)

        if self.extend_features:
            output = self.ext(output)

        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        self.mod1.fuse_model()
        modules = [self.mod2, self.mod3, self.mod4, self.mod5]
        if self.include_top:
            modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

        if self.extend_features:
            self.ext.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "2"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def process_sequence_input(self, x: List) -> Union[torch.Tensor, QTensor]:
        """Process sequence input with cap."""
        assert self.input_sequence_length > 1
        assert isinstance(x, Sequence)
        x_list = [x[0] if self.disable_quanti_input else self.quant(x[0])]
        for i in range(1, self.input_sequence_length):
            quant = getattr(self, f"extra_quant_{i}")
            x_list.append(x[i] if self.disable_quanti_input else quant(x[i]))
        return self.cat_op.cap(x_list, dim=1)


def get_vargnetv2_stride2channels(
    alpha: float,
    channels: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
) -> Dict:
    """
    Get vargnet v2 stride to channel dict with giving channels and strides.

    Args:
        alpha: channel multipler.
        channels: base channel of each stride.
        strides: stride list corresponding to channels.

    Returns
        strides2channels: a stride to channel dict.
    """
    if channels is None:
        channels = [8, 8, 16, 32, 64, 64, 128, 256]
    if strides is None:
        strides = [2, 4, 8, 16, 32, 64, 128, 256]

    assert alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    strides2channels = {}
    for s, c in zip(strides, channels):
        strides2channels[s] = int(alpha / 0.25) * c
    return strides2channels


@OBJECT_REGISTRY.register
class TinyVargNetV2(VargNetV2):
    """
    A module of TinyVargNetv2.

    Args:
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        alpha (float): Alpha for tinyvargnetv2.
        group_base (int): Group base for tinyvargnetv2.
        factor (int): Factor for channel expansion in basic block.
        bias (bool): Whether to use bias in module.
        extend_features (bool): Whether to extend features.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
        input_channels (int): Input channels of first conv.
        input_sequence_length (int): Length of input sequence.
        head_factor (int): Factor for channels expansion of stage1(mod2).
        input_resize_scale (int): Narrow_model need resize input 0.65 scale,
            While int_infer or visualize or eval
    """

    def __init__(
        self,
        num_classes,
        bn_kwargs: dict,
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        bias: bool = True,
        extend_features: bool = False,
        disable_quanti_input: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
        input_channels: int = 3,
        input_sequence_length: int = 1,
        head_factor: int = 1,
        input_resize_scale: int = None,
    ):

        model_type = "TinyVargNetV2"

        super(TinyVargNetV2, self).__init__(
            num_classes=num_classes,
            bn_kwargs=bn_kwargs,
            model_type=model_type,
            alpha=alpha,
            group_base=group_base,
            factor=factor,
            bias=bias,
            extend_features=extend_features,
            disable_quanti_input=disable_quanti_input,
            include_top=include_top,
            flat_output=flat_output,
            input_channels=input_channels,
            input_sequence_length=input_sequence_length,
            head_factor=head_factor,
            input_resize_scale=input_resize_scale,
        )


@OBJECT_REGISTRY.register
class CocktailVargNetV2(VargNetV2):
    """CocktailVargNetV2.

    对 VargNetV2 进行了简单魔改.
    主要是去掉对 num_classes 作为 args 的要求和支持 top_layer 自定义.

    TODO(ziyang01.wang) 重构计划, 将相应的修改吸收到 VargNetV2 中.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        model_type: str = "VargNetV2",
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        bias: bool = True,
        disable_quanti_input: bool = False,
        flat_output: bool = True,
        input_channels: int = 3,
        head_factor: int = 1,
        input_resize_scale: int = None,
        top_layer: Optional[nn.Module] = None,
    ):
        super(VargNetV2, self).__init__()
        self.model_type = model_type.lower()
        assert self.model_type in ["vargnetv2", "tinyvargnetv2"], (
            f"`model_type` should be one of ['vargnetv2', 'tinyvargnetv2'],"
            f" but get {model_type}."
        )
        self.group_base = group_base
        self.factor = factor
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.disable_quanti_input = disable_quanti_input
        self.flat_output = flat_output
        self.head_factor = head_factor
        assert self.head_factor in [1, 2], "head_factor should be 1 or 2"
        self.input_resize_scale = input_resize_scale
        if self.input_resize_scale is not None:
            assert self.input_resize_scale > 0
        channel_list = [32, 32, 64, 128, 256]
        if self.model_type == "tinyvargnetv2":
            units = [1, 3, 4, 2]
        else:
            units = [1, 3, 7, 4]
        channel_list = [int(chls * alpha) for chls in channel_list]

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if self.input_resize_scale is not None:
            self.resize = hnn.Interpolate(
                scale_factor=input_resize_scale,
                align_corners=None,
                recompute_scale_factor=True,
            )

        self.in_channels = channel_list[0]
        self.mod1 = ConvModule2d(
            input_channels,
            channel_list[0],
            3,
            stride=2,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(channel_list[0], **bn_kwargs),
            act_layer=nn.ReLU(inplace=True)
            if self.model_type == "tinyvargnetv2"
            else None,
        )

        head_factor = 2 if self.head_factor == 2 else 8 // group_base
        self.mod2 = self._make_stage(
            channel_list[1], units[0], head_factor, False
        )
        self.mod3 = self._make_stage(channel_list[2], units[1], self.factor)
        self.mod4 = self._make_stage(channel_list[3], units[2], self.factor)
        self.mod5 = self._make_stage(channel_list[4], units[3], self.factor)
        self.output = None if top_layer is None else top_layer

    def forward(self, x):
        x = x if self.disable_quanti_input else self.quant(x)
        if self.input_resize_scale is not None:
            x = self.resize(x)
        output = []
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
        if self.output is not None:
            x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(x.shape[0], -1)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]
        if self.output is not None:
            modules.append(self.output)
        for mod in modules:
            if isinstance(mod, nn.Sequential):
                for m in mod:
                    if hasattr(m, "fuse_model"):
                        m.fuse_model()
            elif hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if self.output is not None:
            if hasattr(self.output, "set_qconfig"):
                self.output[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
