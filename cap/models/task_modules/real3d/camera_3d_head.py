# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.base_modules.conv_module import ConvModule2d
from cap.models.base_modules.extend_container import ExtSequential
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class Camera3DHead(nn.Module):
    """Camera3DHead module for real3d task.

    Args:
        output_cfg(dict): To define items of outputs.
        bn_kwargs(dict): Extra keyword arguments for bn layers.
        in_strides(list[int]): The strides corresponding to the inputs of
                    seg_head, the inputs usually come from backbone or neck.
        in_channels(list[int]): Number of channels of each input stride.
        out_stride(int): output stride.
        use_bias(bool): Whether to use bias in conv module.
        int8_output(bool):If True, output int8, otherwise output int32.
    """

    def __init__(
        self,
        output_cfg,
        bn_kwargs,
        in_strides: List[int],
        in_channels: List[int],
        out_stride: int,
        use_bias: bool = True,
        int8_output: bool = False,
    ):
        super().__init__()

        self.in_strides = in_strides

        assert len(in_strides) == len(
            in_channels
        ), f"{in_strides} vs. f{in_channels}"

        self.int8_output = int8_output

        self.out_idx = in_strides.index(out_stride)
        in_channel = in_channels[self.out_idx]

        self.output_blocks = nn.ModuleDict()

        for out_name, head_cfg in output_cfg.items():
            block = []
            _in_channel = in_channel
            if (
                "out_conv_channels" in head_cfg
                and head_cfg["out_conv_channels"] > 0
            ):
                block.append(
                    ConvModule2d(
                        in_channels=_in_channel,
                        out_channels=head_cfg["out_conv_channels"],
                        kernel_size=(1, 1),
                        stride=1,
                        padding=0,
                        bias=use_bias,
                        norm_layer=nn.BatchNorm2d(
                            head_cfg["out_conv_channels"], **bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                _in_channel = head_cfg["out_conv_channels"]
            block.append(
                ConvModule2d(
                    in_channels=_in_channel,
                    out_channels=head_cfg["out_channels"],
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=use_bias,
                    norm_layer=None,
                    act_layer=None,
                )
            )
            self.output_blocks[out_name] = ExtSequential(block)

        self.dequant = DeQuantStub()

    def forward(self, x: List[torch.Tensor]) -> OrderedDict:
        outputs = OrderedDict()
        x_input = x[self.out_idx]
        for out_name, output_block in self.output_blocks.items():
            out = self.dequant(output_block(x_input))
            outputs[out_name] = out

        return outputs

    def fuse_model(self):
        for m in self.output_blocks.values():
            m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if not self.int8_output:
            for m in self.output_blocks.values():
                m[-1].qconfig = qconfig_manager.get_default_qat_out_qconfig()
