# Copyright (c) Changan Auto. All rights reserved.
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, VargNASNetBlock
from cap.registry import OBJECT_REGISTRY

__all__ = ["VargNASNet", "get_vargnasnet_stride2channels"]


@dataclass
class VargNASBlockConfig:
    in_channels: int
    out_channels: int
    head_op: str
    stack_ops: List[str]
    stack_ops_num: int
    stride: int


@OBJECT_REGISTRY.register
class VargNASNet(nn.Module):
    """
    Module of VargNASNet.

    Args:
        net_config (List[VargNASBlockConfig]): network setting.
        num_classes (int): Num classes.
        bn_kwargs (dict): Kwargs of bn layer.
        disable_quanti_input (bool): whether quanti input.
        fc_filter(int): the out_channels of the last_conv.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
        bias (bool): Whehter to use bias.
        input_channels (int): Input image channels, first conv input
            channels is input_channels times input_sequence_length.
        input_sequence_length (int): Input sequence length, used in
            multiple input images case.
    """

    def __init__(
        self,
        net_config: List[VargNASBlockConfig],
        num_classes: int,
        bn_kwargs: dict,
        disable_quanti_input: bool = False,
        fc_filter: int = 1024,
        include_top: bool = True,
        flat_output: bool = True,
        bias: bool = False,
        input_channels: int = 3,
        input_sequence_length: int = 1,
    ):
        super(VargNASNet, self).__init__()
        self.num_classes = num_classes
        self.disable_quanti_input = disable_quanti_input
        self.fc_filter = fc_filter
        self.include_top = include_top
        self.flat_output = flat_output
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.input_sequence_length = input_sequence_length

        self.quant = QuantStub()
        if input_sequence_length > 1:
            self.cat_op = nn.quantized.FloatFunctional()
            for i in range(1, input_sequence_length):
                setattr(self, f"extra_quant_{i}", QuantStub())
        self.dequant = DeQuantStub()

        self.mod1 = ConvModule2d(
            in_channels=input_channels * input_sequence_length,
            out_channels=net_config[0].in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(net_config[0].in_channels, **bn_kwargs),
        )
        stage_config = self._split_stage(net_config)

        self.mod2 = self._make_stage(stage_config[0])
        self.mod3 = self._make_stage(stage_config[1])
        self.mod4 = self._make_stage(stage_config[2])
        self.mod5 = self._make_stage(stage_config[3])

        if self.include_top:
            self.output = nn.Sequential(
                ConvModule2d(
                    in_channels=net_config[-1].out_channels,
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
                    norm_layer=nn.BatchNorm2d(self.num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _split_stage(self, net_config):
        # TODO (mengao.zhao, v0.1), more flexible
        stage_config = []
        stage_config.append([net_config[0], net_config[1]])
        stage_config.append([net_config[2]])
        stage_config.append([net_config[3], net_config[4]])
        stage_config.append([net_config[5], net_config[6]])
        assert len(stage_config) == 4
        return stage_config

    def _make_stage(self, stage_config):
        layers = []
        for config_i in stage_config:
            layers.append(
                VargNASNetBlock(
                    in_ch=config_i.in_channels,
                    block_ch=config_i.out_channels,
                    head_op=config_i.head_op,
                    stack_ops=config_i.stack_ops,
                    stride=config_i.stride,
                    bias=self.bias,
                    bn_kwargs=self.bn_kwargs,
                )
            )
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        output = []
        if self.input_sequence_length > 1:
            assert len(x) == self.input_sequence_length
            x = self.process_sequence_input(x)
        else:
            if isinstance(x, Sequence) and len(x) == 1:
                x = x[0]
            x = x if self.disable_quanti_input else self.quant(x)
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
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


def get_vargnasnet_stride2channels(
    net_config: List[VargNASBlockConfig],
    strides: Optional[List[int]] = None,
) -> Dict:
    """
    Get vargnasnet stride to channel dict with giving channels and strides.

    Args:
        net_config: network setting
        strides: stride list corresponding to channels.

    Returns
        strides2channels: a stride to channel dict.
    """
    if strides is None:
        strides = [2, 4, 8, 16, 32, 64, 128, 256]

    net_stride_list = [config_i.stride for config_i in net_config]
    channels = []

    if net_stride_list[0] == 2:
        channels.append(net_config[0].in_channels)
    for i in range(len(net_stride_list) - 1):
        if net_stride_list[i + 1] == 2:
            channels.append(net_config[i].out_channels)
    channels.append(net_config[-1].out_channels)
    channels = channels + [net_config[-1].out_channels] * (
        len(strides) - len(channels)
    )

    strides2channels = {}
    for s, c in zip(strides, channels):
        strides2channels[s] = c
    return strides2channels
