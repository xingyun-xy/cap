# Copyright (c) Changan Auto. All rights reserved.
import math
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.quantized as nnq
from changan_plugin_pytorch import nn as nnh
from changan_plugin_pytorch import quantization
from torch import nn
from torch.nn import Module

from cap.models.base_modules import SeparableConvModule2d
from cap.registry import OBJECT_REGISTRY

__all__ = ["DwUnet"]


class UpscaleAndFusion(Module):
    def __init__(
        self,
        vertical_in_channels,
        out_channels,
        horizontal_in_channels,
        bn_kwargs,
        act_type=nn.ReLU,
        use_deconv=False,
        dw_with_act=False,
    ):
        super(UpscaleAndFusion, self).__init__()
        self.horizontal_conv = SeparableConvModule2d(
            in_channels=horizontal_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dw_norm_layer=nn.BatchNorm2d(horizontal_in_channels, **bn_kwargs),
            dw_act_layer=act_type() if dw_with_act else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            pw_act_layer=act_type(),
        )

        self.vertical_conv = SeparableConvModule2d(
            vertical_in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dw_norm_layer=nn.BatchNorm2d(vertical_in_channels, **bn_kwargs),
            dw_act_layer=act_type() if dw_with_act else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            pw_act_layer=act_type(),
        )
        # TODO: support use_deconv param
        if use_deconv:
            raise ValueError("DwUnet do not support use_deconv=True currently")
        else:
            self.upscale = nnh.Interpolate(
                scale_factor=2, recompute_scale_factor=True
            )

        self.fuse_conv = SeparableConvModule2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            dw_act_layer=act_type() if dw_with_act else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            pw_act_layer=None,
        )

        self.add = nnq.FloatFunctional()
        self.relu = act_type()

    def forward(self, vertical_input, horizontal_input):
        horizontal_input = self.horizontal_conv(horizontal_input)

        vertical_input = self.vertical_conv(vertical_input)
        vertical_input = self.upscale(vertical_input)

        horizontal_input = self.fuse_conv(horizontal_input)

        return self.relu(self.add.add(horizontal_input, vertical_input))

    def fuse_model(self):
        self.horizontal_conv.fuse_model()
        self.vertical_conv.fuse_model()
        getattr(self.fuse_conv, "0").fuse_model()

        fuse_list = [
            "fuse_conv.1.0",
            "fuse_conv.1.1",
            "add",
            "relu",
        ]

        torch.quantization.fuse_modules(
            self,
            [fuse_list],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


@OBJECT_REGISTRY.register
class DwUnet(Module):
    """
    Unet segmentation neck structure.

    Built with separable convolution layers.

    Args:
        base_channels (int):
            Output channel number of the output layer of scale 1.
        bn_kwargs (Dict, optional): Keyword arguments for BN layer.
            Defaults to {}.
        use_deconv (bool, optional): Whether user deconv for upsampling layer.
            Defaults to False.
        dw_with_act (bool, optional):
            Whether user relu after the depthwise conv in SeparableConv.
            Defaults to False.
        output_scales (Sequence, optional): The scale of each output layer.
            Defaults to (4, 8, 16, 32, 64).
    """

    def __init__(
        self,
        base_channels: int,
        bn_kwargs: Dict = None,
        act_type: nn.Module = nn.ReLU,
        use_deconv: bool = False,
        dw_with_act: bool = False,
        output_scales: Sequence = (4, 8, 16, 32, 64),
    ):
        super(DwUnet, self).__init__()
        self.base_channels = base_channels
        self.bn_kwargs = bn_kwargs or {}
        self.act_type = act_type
        self.use_deconv = use_deconv
        self.dw_with_act = dw_with_act
        assert set(output_scales).issubset(
            {1, 2, 4, 8, 16, 32, 64}
        ), "output_scales must be in (1, 2, 4, 8, 16, 32, 64)"
        self.output_scales = output_scales

        self.conv64_1 = SeparableConvModule2d(
            base_channels * 32,
            base_channels * 64,
            kernel_size=3,
            stride=2,
            padding=1,
            dw_norm_layer=nn.BatchNorm2d(base_channels * 32, **self.bn_kwargs),
            dw_act_layer=act_type() if self.dw_with_act else None,
            pw_norm_layer=nn.BatchNorm2d(base_channels * 64, **self.bn_kwargs),
            pw_act_layer=act_type(),
        )
        self.conv64_2 = SeparableConvModule2d(
            base_channels * 64,
            base_channels * 64,
            kernel_size=3,
            padding=1,
            dw_norm_layer=nn.BatchNorm2d(base_channels * 64, **self.bn_kwargs),
            dw_act_layer=act_type() if self.dw_with_act else None,
            pw_norm_layer=nn.BatchNorm2d(base_channels * 64, **self.bn_kwargs),
            pw_act_layer=act_type(),
        )

        stage_num = int(math.log2(np.max(64 / np.array(output_scales))))

        self.stages = nn.ModuleList()
        for stage in range(stage_num):
            vertical_in_channels = int(base_channels * (2 ** (6 - stage)))
            out_channels = int(vertical_in_channels / 2)
            horizontal_in_channels = out_channels
            self.stages.append(
                self._make_upscale_and_fusion(
                    vertical_in_channels, out_channels, horizontal_in_channels
                )
            )

        self.output_layers = nn.ModuleDict()
        for output_scale in output_scales:
            channels = int(base_channels * output_scale)
            self.output_layers[
                self._out_layer_scale_to_name(output_scale)
            ] = self._make_output_layers(channels, channels)

    def _out_layer_scale_to_name(self, output_scale):
        return "out_layer_scale_%d" % output_scale

    def _make_output_layers(self, in_channels, out_channels):
        return SeparableConvModule2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dw_norm_layer=nn.BatchNorm2d(in_channels, **self.bn_kwargs),
            dw_act_layer=self.act_type() if self.dw_with_act else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs),
            pw_act_layer=self.act_type(),
        )

    def _make_upscale_and_fusion(
        self,
        vertical_in_channels,
        out_channels,
        horizontal_in_channels,
    ):
        return UpscaleAndFusion(
            vertical_in_channels,
            out_channels,
            horizontal_in_channels,
            self.bn_kwargs,
            self.act_type,
            self.use_deconv,
            self.dw_with_act,
        )

    def forward(self, inputs):
        ret_outputs = []
        current_scale = 64

        inputs = list(reversed(inputs))

        x = self.conv64_1(inputs[0])
        x = self.conv64_2(x)
        if current_scale in self.output_scales:
            ret_outputs.append(
                self.output_layers[
                    self._out_layer_scale_to_name(current_scale)
                ](x)
            )

        for stage, mod in enumerate(self.stages):
            x = mod(x, inputs[stage])
            current_scale /= 2

            if current_scale in self.output_scales:
                ret_outputs.append(
                    self.output_layers[
                        self._out_layer_scale_to_name(current_scale)
                    ](x)
                )

        return list(reversed(ret_outputs))

    def fuse_model(self):
        self.conv64_1.fuse_model()
        self.conv64_2.fuse_model()
        for mod in self.stages:
            mod.fuse_model()
        for mod in self.output_layers.values():
            mod.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
