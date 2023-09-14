# Copyright (c) Changan Auto. All rights reserved.
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, MixVarGEBlock
from cap.models.base_modules.extend_container import ExtSequential
from cap.registry import OBJECT_REGISTRY

__all__ = ["MixVarGENet", "get_mixvargenet_stride2channels"]


@dataclass
class MixVarGENetConfig:
    """
    Config of MixVarGEBlock.

    Args:
        in_channels: The in_channels for the block.
        out_channels: The out_channels for the block.
        head_op: One key of the BLOCK_CONFIG.
        stack_ops: a list consisting the keys of the
            BLOCK_CONFIG, or be None.
        stack_factor: channel factor of middle stack ops.
            Input and output channels of stack remains the same.
        stride: Stride of basic block.
        fusion_strides: A list of strides,
            whose downsampled feature map would be fused in current stage.
            Eg, fusion_strides of stride16's head block
            should contains downsample layers of stride4 or stride8.
        extra_downsample_num: Downsampled feature maps of current block's
            output feature map,
            which could be next few blocks' shortcut input.
            Currently 0, 1, 2, 3 are supported.
            0 means no backbone fusion,
            1 means fusion 2x scales,
            2 means fusion 2x and 4x scales.
            3 means fusion 2x and 4x and 8x scales.
    """

    in_channels: int
    out_channels: int
    head_op: str
    stack_ops: List[str]
    stride: int
    stack_factor: Optional[int] = 1
    fusion_strides: Optional[Union[Tuple[int], List[int]]] = ()
    extra_downsample_num: Optional[int] = 0


@OBJECT_REGISTRY.register
class MixVarGENet(nn.Module):
    """
    Module of MixVarGENet.

    Args:
        net_config (List[MixVarGENetConfig]): network setting.
        num_classes (int): Num classes.
        bn_kwargs (dict): Kwargs of bn layer.
        output_list (List[int]): Output id of net_config blocks.
            The output of these block will be the output of this net.
            Set output_list as [] would export all block's output.
        disable_quanti_input (bool): whether quanti input.
        fc_filter(int): the out_channels of the last_conv.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
        bias (bool): Whehter to use bias.
        input_channels (int): Input image channels, first conv input
            channels is input_channels times input_sequence_length.
        input_resize_scale: This will resize the input image with
            the scale value.
    """

    def __init__(
        self,
        net_config: List[MixVarGENetConfig],
        num_classes: int,
        bn_kwargs: dict,
        output_list: Union[List[int], Tuple[int]] = (),
        disable_quanti_input: bool = False,
        fc_filter: int = 1024,
        include_top: bool = True,
        flat_output: bool = True,
        bias: bool = False,
        input_channels: int = 3,
        input_sequence_length: int = 1,
        input_resize_scale: int = None,
    ):
        super(MixVarGENet, self).__init__()
        self.num_classes = num_classes
        self.disable_quanti_input = disable_quanti_input
        self.fc_filter = fc_filter
        self.include_top = include_top
        self.flat_output = flat_output
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.input_sequence_length = input_sequence_length
        self.output_list = output_list
        self.input_resize_scale = input_resize_scale
        if self.input_resize_scale is not None:
            assert self.input_resize_scale > 0
        self._set_fusion_data()

        self.quant = QuantStub()
        if input_sequence_length > 1:
            self.cat_op = nn.quantized.FloatFunctional()
            for i in range(1, input_sequence_length):
                setattr(self, f"extra_quant_{i}", QuantStub())
        self.dequant = DeQuantStub()

        self.mod1 = ConvModule2d(
            in_channels=input_channels * input_sequence_length,
            out_channels=net_config[0][0].in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(
                net_config[0][0].in_channels, **bn_kwargs
            ),
        )

        self.net_config = net_config
        stage_config = net_config

        self.mod_num = len(stage_config)

        for i in range(self.mod_num):
            cur_mod = "mod" + str(i + 2)
            setattr(self, cur_mod, self._make_stage(stage_config[i]))

        if self.include_top:
            self.output = ExtSequential(
                [
                    ConvModule2d(
                        in_channels=stage_config[-1][-1].out_channels,
                        out_channels=self.fc_filter,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(self.fc_filter, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    ),
                    nn.AvgPool2d(7, stride=1),
                    ConvModule2d(
                        in_channels=self.fc_filter,
                        out_channels=self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(
                            self.num_classes, **bn_kwargs
                        ),
                    ),
                ]
            )
        else:
            self.output = None

    def _make_stage(self, stage_config):
        def _get_fusion_channels(fusion_strides):
            if len(fusion_strides) == 0:
                return []
            strides_ids = map(
                lambda stride: int(math.log2(stride) - 1), fusion_strides
            )
            fusion_channels = map(
                lambda idx: self.net_config[idx][0].out_channels, strides_ids
            )
            return list(fusion_channels)

        layers = []
        for config_i in stage_config:
            layers.append(
                MixVarGEBlock(
                    in_ch=config_i.in_channels,
                    block_ch=config_i.out_channels,
                    head_op=config_i.head_op,
                    stack_ops=config_i.stack_ops,
                    stack_factor=config_i.stack_factor,
                    stride=config_i.stride,
                    bias=self.bias,
                    fusion_channels=_get_fusion_channels(
                        config_i.fusion_strides
                    ),
                    downsample_num=config_i.extra_downsample_num,
                    bn_kwargs=self.bn_kwargs,
                )
            )
        return ExtSequential(layers)

    def _set_fusion_data(self):
        self.start_level = None
        self.candidate_fuse2x_list = []  # features downsampled from 2X stride
        self.candidate_fuse4x_list = []  # features downsampled from 4X stride
        self.candidate_fuse8x_list = []  # features downsampled from 8X stride

    def _handle_fusion_data(self, x, downsamples, idx):
        # fusion up scale info
        assert len(downsamples) > 0
        if self.start_level is None:
            self.start_level = idx

        self.candidate_fuse2x_list.append(downsamples[0])
        if len(downsamples) > 1:
            self.candidate_fuse4x_list.append(downsamples[1])
        if len(downsamples) > 2:
            self.candidate_fuse8x_list.append(downsamples[2])

        next_input = [x]
        idx -= self.start_level
        # next module input:[cur_scale, 8X, 4X, 2X]
        if 0 <= idx - 2 < len(self.candidate_fuse8x_list):
            next_input.append(self.candidate_fuse8x_list[idx - 2])
        if 0 <= idx - 1 < len(self.candidate_fuse4x_list):
            next_input.append(self.candidate_fuse4x_list[idx - 1])
        if 0 <= idx < len(self.candidate_fuse2x_list):
            next_input.append(self.candidate_fuse2x_list[idx])

        return next_input

    def forward(self, x):
        output = []
        if self.input_sequence_length > 1:
            assert len(x) == self.input_sequence_length
            x = self.process_sequence_input(x)
        else:
            if isinstance(x, Sequence) and len(x) == 1:
                x = x[0]
            x = x if self.disable_quanti_input else self.quant(x)
        if self.input_resize_scale is not None:
            x = F.interpolate(x, scale_factor=self.input_resize_scale)
        modules = []
        for i in range(self.mod_num):
            cur_mod = "mod" + str(i + 2)
            modules.append(getattr(self, cur_mod))

        x = self.mod1(x)
        for idx, module in enumerate(modules):
            x, downsamples = module(x)
            if len(self.output_list) == 0 or idx in self.output_list:
                output.append(x)

            # fusion up scale info
            if len(downsamples) > 0:
                x = self._handle_fusion_data(x, downsamples, idx)

        self._set_fusion_data()
        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        self.mod1.fuse_model()
        modules = []
        for i in range(self.mod_num):
            cur_mod = "mod" + str(i + 2)
            modules.append(getattr(self, cur_mod))
        if self.include_top:
            modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

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
        x_list = []
        for i in range(0, self.input_sequence_length):
            if i == 0:
                quant = self.quant
            else:
                quant = getattr(self, f"extra_quant_{i}")
            x_list.append(x[i] if self.disable_quanti_input else quant(x[i]))
        return self.cat_op.cap(x_list, dim=1)


def get_mixvargenet_stride2channels(
    net_config: List[List[MixVarGENetConfig]],
    strides: Optional[List[int]] = None,
) -> Dict:
    """
    Get mixvargenet stride to channel dict with giving channels and strides.

    Args:
        net_config: network setting
        strides: stride list corresponding to channels.

    Returns
        strides2channels: a stride to channel dict.
    """
    if strides is None:
        strides = [2, 4, 8, 16, 32, 64, 128, 256]

    flat_net_config = []
    for stage in net_config:
        for config in stage:
            flat_net_config.append(config)

    net_stride_list = [config_i.stride for config_i in flat_net_config]
    channels = []

    if net_stride_list[0] == 2:
        channels.append(flat_net_config[0].in_channels)
    for i in range(len(net_stride_list) - 1):
        if net_stride_list[i + 1] == 2:
            channels.append(flat_net_config[i].out_channels)
    channels.append(flat_net_config[-1].out_channels)
    channels = channels + [flat_net_config[-1].out_channels] * (
        len(strides) - len(channels)
    )

    strides2channels = {}
    for s, c in zip(strides, channels):
        strides2channels[s] = c
    return strides2channels
