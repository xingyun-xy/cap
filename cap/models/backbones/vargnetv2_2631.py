# Copyright (c) Changan Auto. All rights reserved.
from functools import partial

import torch.nn as nn
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import (
    BasicVarGBlock,
    ConvModule2d,
    ExtendVarGNetFeatures,
    ExtSequential,
)
from cap.registry import OBJECT_REGISTRY

__all__ = ["VargNetV2Stage2631"]


unit_dict = {
    "v2unitA": partial(BasicVarGBlock, merge_branch=False),
    "v2unitB": partial(BasicVarGBlock, merge_branch=True),
}


def get_varg_multitask_module(num_in, config, bn_kwargs, prefix):
    _, num_out, stride, group_base, search_unit_str = config

    index = search_unit_str.find("_k")
    kernel_size = int(search_unit_str[index + 2])
    expand_ratio = int(search_unit_str[index + 4 :])
    search_unit_name = search_unit_str[:index]
    search_unit = unit_dict[search_unit_name]

    padding = kernel_size // 2

    name = f"{prefix}_{search_unit_name}"
    module = search_unit(
        bn_kwargs=bn_kwargs,
        in_channels=num_in,
        mid_channels=num_out,
        out_channels=num_out,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        factor=expand_ratio,
        group_base=group_base,
        bias=False,
        pw_with_relu=False,
    )

    return name, module


@OBJECT_REGISTRY.register
class VargNetV2Stage2631(nn.Module):
    """VargNetV2 with 2631 stage setting."""

    def __init__(
        self,
        bn_kwargs,
        multiplier=0.5,
        group_base=8,
        last_channels=1024,
        num_classes=1000,
        stages=(1, 2, 3, 4, 5, 6),
        use_bias=False,
        input_channels: int = 3,
        include_top=False,
        extend_features=False,
        flat_output=True,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.flat_output = flat_output

        for i in range(1, 6):
            if i in stages:
                self.add_module(f"stage{i}", ExtSequential([]))
            else:
                setattr(self, f"stage{i}", None)

        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(
                min_depth, int(x + depth_div / 2) // depth_div * depth_div
            )
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        depth_div, min_depth = 8, 8
        width_coeff = multiplier

        gb = group_base

        configs = [
            # in_c, out_c, stride, gb, unit
            [],
            [
                [renew_ch(32), renew_ch(32), 2, gb, "v2unitA_k3f2"],
                [renew_ch(32), renew_ch(32), 1, 8, "v2unitA_k3f1"],
            ],
            [
                [renew_ch(32), renew_ch(64), 2, gb, "v2unitB_k3f2"],
                [renew_ch(64), renew_ch(64), 1, gb, "v2unitA_k3f2"],
                [renew_ch(64), renew_ch(64), 1, gb, "v2unitA_k3f2"],
                [renew_ch(64), renew_ch(64), 1, gb, "v2unitA_k3f2"],
                [renew_ch(64), renew_ch(64), 1, gb, "v2unitA_k3f2"],
                [renew_ch(64), renew_ch(64), 1, gb, "v2unitA_k3f2"],
            ],
            [
                [renew_ch(64), renew_ch(128), 2, gb, "v2unitB_k3f2"],
                [renew_ch(128), renew_ch(128), 1, gb, "v2unitA_k3f2"],
                [renew_ch(128), renew_ch(128), 1, gb, "v2unitA_k3f2"],
            ],
            [
                [renew_ch(128), renew_ch(256), 2, gb, "v2unitB_k5f2"],
            ],
        ]

        num_in = int(configs[1][0][0])

        if self.stage1 is not None:
            self.stage1.add_module(
                "conv1_conv3x3",
                ConvModule2d(
                    in_channels=input_channels,
                    out_channels=num_in,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    norm_layer=nn.BatchNorm2d(num_in, **bn_kwargs),
                ),
            )
            for i in range(len(configs[0])):
                config = configs[0][i]
                if config is None:
                    continue
                self.stage1.add_module(
                    *get_varg_multitask_module(
                        num_in=num_in,
                        config=config,
                        bn_kwargs=bn_kwargs,
                        prefix=f"unit{i + 1}",
                    )
                )
                num_in = config[1]

        for stage_i in range(2, 6):
            stage_module = getattr(self, f"stage{stage_i}")
            if stage_module is not None:
                for i in range(len(configs[stage_i - 1])):
                    config = configs[stage_i - 1][i]
                    stage_module.add_module(
                        *get_varg_multitask_module(
                            num_in=num_in,
                            config=config,
                            bn_kwargs=bn_kwargs,
                            prefix=f"unit{i + 1}",
                        )
                    )
                    num_in = config[1]

        if extend_features:
            self.ext = ExtendVarGNetFeatures(
                prev_channel=128,
                channels=128,
                num_units=2,
                group_base=group_base,
                bn_kwargs=bn_kwargs,
            )

        if include_top:
            self.output = ExtSequential(
                [
                    ConvModule2d(
                        in_channels=num_in,
                        out_channels=last_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=use_bias,
                        act_layer=nn.ReLU(inplace=True),
                        norm_layer=nn.BatchNorm2d(last_channels, **bn_kwargs),
                    ),
                    ConvModule2d(
                        in_channels=last_channels,
                        out_channels=last_channels,
                        kernel_size=7,
                        stride=1,
                        padding=0,
                        bias=use_bias,
                        act_layer=nn.ReLU(inplace=True),
                        norm_layer=nn.BatchNorm2d(last_channels, **bn_kwargs),
                    ),
                    ConvModule2d(
                        in_channels=last_channels,
                        out_channels=num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                ]
            )

    def forward(self, x):
        x = self.quant(x)

        output = []
        for module in [
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
        ]:
            x = module(x)
            output.append(x)

        if hasattr(self, "ext"):
            output = self.ext(output)

        if not hasattr(self, "output"):
            return output

        x = self.output(x)

        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return self.dequant(x)

    def fuse_model(self):
        for module in self.children():
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if hasattr(self, "output"):
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "2"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
