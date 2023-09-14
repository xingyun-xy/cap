# Copyright (c) Changan Auto. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from cap.models.base_modules.extend_container import ExtSequential
from .conv_module import ConvModule2d

__all__ = ["BasicMixVarGEBlock", "MixVarGEBlock"]

BLOCK_CONFIG = {
    "mixvarge_k3k3_f2": {"conv2_kernel_size": 3, "padding": 1, "factor": 2},
    "mixvarge_k3k3_f2_gb16": {
        "conv2_kernel_size": 3,
        "padding": 1,
        "factor": 2,
        "conv1_group_base": 16,
        "conv2_group_base": 16,
    },
    "mixvarge_f2": {"conv2_kernel_size": 1, "padding": 0, "factor": 2},
    "mixvarge_f4": {"conv2_kernel_size": 1, "padding": 0, "factor": 4},
    "mixvarge_f2_gb16": {
        "conv2_kernel_size": 1,
        "padding": 0,
        "factor": 2,
        "conv1_group_base": 16,
    },
    "mixvarge_f4_gb16": {
        "conv2_kernel_size": 1,
        "padding": 0,
        "factor": 4,
        "conv1_group_base": 16,
    },
    "mixvarge_f2_r": {
        "conv2_kernel_size": 1,
        "padding": 0,
        "factor": 2,
        "merge_branch": True,
    },
    "mixvarge_f2_r_gb16": {
        "conv2_kernel_size": 1,
        "padding": 0,
        "factor": 2,
        "conv1_group_base": 16,
        "merge_branch": True,
    },
}


class MixVarGEBlock(nn.Module):
    """
    A block for MixVarGEBlock.

    Args:
        in_ch: The in_channels for the block.
        block_ch: The out_channels for the block.
        head_op: One key of the BLOCK_CONFIG.
        stack_ops: a list consisting the keys of the
            BLOCK_CONFIG, or be None.
        stack_factor: channel factor of middle stack ops.
            Input and output channels of stack remains the same.
        stride: Stride of basic block.
        bias: Whether to use bias in basic block.
        bn_kwargs: Dict for BN layer.
        fusion_channels: A list of fusion layer input channels,
            which should be former block's downsampled feature map channel.
            Eg, fusion_channels of stride16's head block
            could contains channels of stride4 or stride8.
        downsample_num: Downsampled feature maps of current block's
            output feature map,
            which could be next few blocks' shortcut input.
            Currently 0, 1, 2, 3 are supported.
            0 means no backbone fusion,
            1 means fusion 2x scales,
            2 means fusion 2x and 4x scales.
            3 means fusion 2x and 4x and 8x scales.
        output_downsample: To controll whether ouput downsample feature list.
    """

    def __init__(
        self,
        in_ch: int,
        block_ch: int,
        head_op: str,
        stack_ops: List[str],
        stride: int,
        bias: bool,
        bn_kwargs: dict,
        stack_factor: Optional[int] = 1,
        fusion_channels: Optional[Union[Tuple[int], List[int]]] = (),
        downsample_num: Optional[int] = 0,
        output_downsample: Optional[bool] = True,
    ):
        assert downsample_num in [0, 1, 2, 3]
        super(MixVarGEBlock, self).__init__()
        self.ouput_downsample = output_downsample
        self.head_layer = BasicMixVarGEBlock(
            in_channels=in_ch,
            out_channels=block_ch,
            stride=stride,
            bias=bias,
            bn_kwargs=bn_kwargs,
            fusion_channels=fusion_channels,
            **BLOCK_CONFIG[head_op],
        )

        modules = []

        stack_num = len(stack_ops)
        for i, stack_op in enumerate(stack_ops):
            if stack_num > 1:
                if i == 0:
                    input_channel = block_ch
                    output_channel = stack_factor * block_ch
                elif i == stack_num - 1:
                    input_channel = stack_factor * block_ch
                    output_channel = block_ch
                else:
                    input_channel = stack_factor * block_ch
                    output_channel = stack_factor * block_ch
            else:
                input_channel = block_ch
                output_channel = block_ch

            modules.append(
                BasicMixVarGEBlock(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=1,
                    bias=bias,
                    bn_kwargs=bn_kwargs,
                    **BLOCK_CONFIG[stack_op],
                )
            )

        self.stack_layers = ExtSequential(modules)

        downsample_fusion_layers = []
        for _idx in range(downsample_num):
            downsample_fusion_layers.append(
                ConvModule2d(
                    block_ch,
                    block_ch,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    norm_layer=nn.BatchNorm2d(block_ch, **bn_kwargs),
                )
            )
        self.downsample_fusion_layers = (
            ExtSequential(downsample_fusion_layers) if downsample_num else None
        )

    def forward(self, x):
        x = self.head_layer(x)
        x = self.stack_layers(x)

        if not self.ouput_downsample:
            return x

        ds_ret = []
        ds = x
        if self.downsample_fusion_layers is not None:
            for downsample in self.downsample_fusion_layers:
                ds = downsample(ds)
                ds_ret.append(ds)
        return x, ds_ret  # downsample fusion layer output from stride4

    def fuse_model(self):
        self.head_layer.fuse_model()
        for module in self.stack_layers:
            module.fuse_model()
        if self.downsample_fusion_layers is not None:
            for module in self.downsample_fusion_layers:
                module.fuse_model()


class BasicMixVarGEBlock(nn.Module):
    """
    A basic block for MixVarGEBlock.

    Args:
        in_channels (int): The in_channels for the block.
        out_channels (int): The out_channels for the block.
        stride (int): Stride of basic block.
        bias (bool): Whether to use bias in basic block.
        bn_kwargs (dict): Dict for BN layer.
        kernel_size (int): Kernel size of basic block.
        padding (int): Padding of basic block.
        factor (int): Factor for channels expansion.
        group_base (int): Group base for group conv.
        fusion_channels (list): Channels of input fusion layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        bias,
        bn_kwargs,
        conv1_kernel_size=3,
        conv2_kernel_size=1,
        padding=0,
        factor=1,
        conv1_group_base=None,
        conv2_group_base=None,
        fusion_channels=(),
        merge_branch=False,
        fuse_2x=False,
    ):
        super(BasicMixVarGEBlock, self).__init__()
        # TODO(jing.li): based on compiler optimization #
        self.fuse_2x = fuse_2x and stride == 2

        mid_channle = (
            out_channels if factor == 1 else int(in_channels * factor)
        )

        conv1_groups = (
            1
            if conv1_group_base is None
            else int(in_channels / conv1_group_base)
        )
        conv2_groups = (
            1
            if conv2_group_base is None
            else int(out_channels / conv2_group_base)
        )

        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels,
                mid_channle,
                conv1_kernel_size,
                padding=(conv1_kernel_size - 1) // 2,
                stride=stride,
                groups=conv1_groups,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channle, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                mid_channle,
                out_channels,
                conv2_kernel_size,
                padding=padding,
                stride=1,
                groups=conv2_groups,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            ),
        )

        fusion_layers = []
        for input_channel in fusion_channels:
            layer = ConvModule2d(
                input_channel,
                out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )
            fusion_layers.append(layer)
        self.fusion_layers = (
            ExtSequential(fusion_layers) if len(fusion_channels) else None
        )

        self.fusion_relu = (
            nn.ReLU(inplace=True) if len(fusion_channels) else None
        )

        self.fusion_adds = ExtSequential(
            [
                nn.quantized.FloatFunctional()
                for i in range(len(fusion_channels))
            ]
        )

        self.downsample = None
        if merge_branch or not (stride == 1 and in_channels == out_channels):
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
        if isinstance(x, list):
            fusion_inputs = x[1:]
            x = x[0]
        else:
            fusion_inputs = []

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv(x)
        out = self.short_add.add(out, identity)
        if len(fusion_inputs) == 0:
            out = self.relu(out)

        # fusion inputs
        if self.fusion_layers is not None:
            assert len(fusion_inputs) == len(self.fusion_layers), (
                len(fusion_inputs),
                self.fusion_layers,
            )
            for _idx, (fusion_input, fusion_layer, fusion_add) in enumerate(
                zip(fusion_inputs, self.fusion_layers, self.fusion_adds)
            ):
                fusion_ouput = fusion_layer(fusion_input)
                out = fusion_add.add(fusion_ouput, out)
            out = self.fusion_relu(out)
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

        if self.fusion_layers is not None:
            for idx in range(len(self.fusion_layers) - 1):
                fusion_list = [
                    f"fusion_layers.{idx}.0",
                    f"fusion_layers.{idx}.1",
                    f"fusion_adds.{idx}",
                ]
                torch.quantization.fuse_modules(
                    self,
                    fusion_list,
                    inplace=True,
                    fuser_func=quantization.fuse_known_modules,
                )
            idx = len(self.fusion_layers) - 1
            fusion_list = [
                f"fusion_layers.{idx}.0",
                f"fusion_layers.{idx}.1",
                f"fusion_adds.{idx}",
                "fusion_relu",
            ]
            torch.quantization.fuse_modules(
                self,
                fusion_list,
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
