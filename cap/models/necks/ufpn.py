# Copyright (c) Changan Auto. All rights reserved.

from itertools import chain
from typing import Dict, List

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn import Interpolate

from cap.models.base_modules.basic_vargnet_module import (
    OnePathResUnit,
    SeparableGroupConvModule2d,
    TwoPathResUnit,
)
from cap.models.base_modules.conv_module import ConvModule2d
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class UFPN(nn.Module):
    """Unet FPN neck.

    Args:
        group_base: Group base of group conv.
        in_strides: strides of each input feature map.
        in_channels: channels of each input feature map, the
            length of in_channels should be equal to in_strides.
        out_channels: channels of each output feature maps, the
            length of out_channels should be equal to in_channels.
        bn_kwargs: Dict for Bn layer. No Bn layer if bn_kwargs=None.
        factor: Factor of group conv.
        out_strides: contains the strides of feature maps the neck output.
    """

    def __init__(
        self,
        group_base: int,
        in_strides: List[int],
        in_channels: List[int],
        out_channels: List[int],
        bn_kwargs: Dict,
        factor: float = 1.0,
        output_strides: List[int] = None,
    ):
        super().__init__()

        self.in_strides = in_strides
        self.output_strides = output_strides

        assert (
            len(in_strides) == len(in_channels) == len(out_channels)
        ), f"{in_strides} vs. f{in_channels} vs. f{out_channels}"

        stride2channels = {s: c for s, c in zip(in_strides, out_channels)}

        self.down_sample = nn.ModuleDict()
        self.conv1x1 = nn.ModuleDict()
        self.qat_adds = nn.ModuleDict()

        for i, (s, in_channel) in enumerate(zip(in_strides, in_channels)):
            if i != 0:
                self.conv1x1[f"stride_{s}"] = ConvModule2d(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(in_channel, **bn_kwargs),
                    act_layer=None,
                )
                self.qat_adds[f"stride_{s}"] = nn.quantized.FloatFunctional()

            if i != len(in_channels) - 1:
                self.down_sample[f"stride_{s}"] = TwoPathResUnit(
                    dw_num_filter=in_channel,
                    group_base=group_base,
                    pw_num_filter=in_channels[i + 1],
                    pw_num_filter2=in_channels[i + 1],
                    bn_kwargs=bn_kwargs,
                    stride=2,
                    is_dim_match=False,
                    use_bias=True,
                    pw_with_act=False,
                    factor=factor,
                )

        self.fusion_blocks = nn.ModuleDict()
        for s in in_strides[-2::-1]:
            top_stride = s * 2
            bottom_stride = s
            block = BottomUpFusion(
                up_c=stride2channels[top_stride],
                bottom_c=stride2channels[bottom_stride],
                out_c=stride2channels[bottom_stride],
                dw_group_base=group_base,
                bn_kwargs=bn_kwargs,
                linear_out=True,
                factor=1.0,
                use_bias=True,
            )
            self.fusion_blocks[f"stride_{s}"] = block

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:

        down_outputs = [x[0]]

        for i, s in enumerate(self.in_strides[:-1]):
            cs = self.in_strides[i + 1]
            down_outputs.append(
                self.qat_adds[f"stride_{cs}"].add(
                    self.conv1x1[f"stride_{cs}"](x[i + 1]),
                    self.down_sample[f"stride_{s}"](down_outputs[-1]),
                )
            )

        up_outputs = [down_outputs[-1]]

        for i, s in enumerate(self.in_strides[-2::-1]):
            up_outputs.append(
                self.fusion_blocks[f"stride_{s}"](
                    up_outputs[-1], down_outputs[-i - 2]
                )
            )

        if not self.output_strides:
            up_outputs = up_outputs[::-1]
        else:
            up_outputs = [
                up_outputs[::-1][self.in_strides.index(i)]
                for i in self.output_strides
            ]
        return up_outputs

    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        for m in chain(
            self.down_sample.values(),
            self.fusion_blocks.values(),
        ):
            m.fuse_model()

        for k in self.conv1x1:
            torch.quantization.fuse_modules(
                self,
                [f"conv1x1.{k}.0", f"conv1x1.{k}.1", f"qat_adds.{k}"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )


class BottomUpFusion(nn.Module):
    def __init__(
        self,
        up_c,
        bottom_c,
        out_c,
        dw_group_base,
        bn_kwargs,
        linear_out=True,
        use_bias=True,
        factor=2.0,
    ):
        super().__init__()

        assert bottom_c % dw_group_base == 0

        self.upscale = Upscale(
            in_c=up_c,
            out_c=bottom_c,
            gc_group_base=dw_group_base,
            bn_kwargs=bn_kwargs,
            linear_out=linear_out,
            use_bias=use_bias,
            factor=factor,
        )
        self.bottom_proj = OnePathResUnit(
            dw_num_filter=bottom_c,
            group_base=dw_group_base,
            pw_num_filter=bottom_c,
            pw_num_filter2=out_c,
            stride=(1, 1),
            is_dim_match=True,
            use_bias=use_bias,
            bn_kwargs=bn_kwargs,
            pw_with_act=not linear_out,
            factor=factor,
        )
        self.fusion = SeparableGroupConvModule2d(
            in_channels=int(bottom_c * factor),
            out_channels=out_c,
            groups=int(bottom_c / dw_group_base),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=use_bias,
            dw_act_layer=nn.ReLU(inplace=True),
            pw_act_layer=None,
            pw_norm_layer=nn.BatchNorm2d(int(bottom_c * factor), **bn_kwargs),
            dw_norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
        )
        self.qat_add = nn.quantized.FloatFunctional()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up, bottom):
        upscale = self.upscale(up)
        out = self.qat_add.add(self.fusion(upscale), self.bottom_proj(bottom))

        out = self.relu(out)

        return out

    def fuse_model(self):

        from changan_plugin_pytorch import quantization

        self.upscale.fuse_model()
        self.bottom_proj.fuse_model()

        getattr(self.fusion, "0").fuse_model()

        torch.quantization.fuse_modules(
            self,
            ["fusion.1.0", "fusion.1.1", "qat_add", "relu"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class Upscale(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        gc_group_base,
        bn_kwargs,
        linear_out=True,
        use_bias=True,
        factor=2.0,
    ):
        super(Upscale, self).__init__()

        self.proj_in = OnePathResUnit(
            dw_num_filter=in_c,
            group_base=gc_group_base,
            pw_num_filter=in_c,
            pw_num_filter2=out_c,
            stride=(1, 1),
            is_dim_match=False,
            use_bias=use_bias,
            bn_kwargs=bn_kwargs,
            pw_with_act=not linear_out,
            factor=factor,
        )

        self.upsample = Interpolate(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )

    def forward(self, x):
        return self.upsample(self.proj_in(x))

    def fuse_model(self):
        self.proj_in.fuse_model()
