# Copyright (c) Changan Auto. All rights reserved.

from typing import List

import torch.nn as nn
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, VargDarkNetBlock
from cap.registry import OBJECT_REGISTRY

__all__ = ["VarGDarkNet53"]


class VarGDarknet(nn.Module):
    """
    Varg darknet.

    Args:
        layers: Number of layers for darknet.
        filters: Filters of input conv for each layer.
        bn_kwargs: Dict for BN layer.
        num_classes: Number classes of output layer.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
    """

    def __init__(
        self,
        layers: int,
        filters: List[int],
        bn_kwargs: dict,
        num_classes: int,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        super(VarGDarknet, self).__init__()
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = ConvModule2d(
            in_channels=3,
            out_channels=filters[0],
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d(filters[0], **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        for i, layer in zip(range(len(layers)), layers):
            stage = nn.ModuleList()
            stage.append(
                ConvModule2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    norm_layer=nn.BatchNorm2d(filters[i + 1], **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            for _ in range(layer):
                stage.append(
                    VargDarkNetBlock(
                        in_channels=filters[i + 1],
                        out_channels=int(filters[i + 1] / 2),
                        bn_kwargs=bn_kwargs,
                    )
                )
            stage = nn.Sequential(*stage)
            self.stages.append(stage)

        if self.include_top:
            self.output = nn.Sequential(
                nn.AvgPool2d(7),
                ConvModule2d(
                    filters[-1],
                    num_classes,
                    1,
                    bias=True,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def forward(self, x):
        output = []
        x = self.quant(x)
        x = self.conv1(x)
        output.append(x)
        for stage in self.stages:
            x = stage(x)
            output.append(x)
        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        self.conv1.fuse_model()
        for stages in self.stages:
            for stage in stages:
                stage.fuse_model()
        if self.include_top:
            for m in self.output:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class VarGDarkNet53(VarGDarknet):
    """
    A module of VarGDarkNet53.

    Args:
        max_channels: Max channels.
        bn_kwargs: Dict for BN layer.
        num_classes: Number classes of output layer.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
    """

    def __init__(
        self,
        max_channels: int,
        bn_kwargs: dict,
        num_classes: int,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        assert max_channels == 512 or max_channels == 1024, (
            f"max_channels must be in {512, 1024}, "
            f"but you set max_channels={max_channels}"
        )
        if max_channels == 512:
            layers = [1, 2, 8, 8, 4]
            filters = [32, 64, 128, 256, 256, 512]
        else:
            layers = [1, 2, 8, 8, 4]
            filters = [32, 64, 128, 256, 512, 1024]

        super(VarGDarkNet53, self).__init__(
            layers=layers,
            filters=filters,
            bn_kwargs=bn_kwargs,
            num_classes=num_classes,
            include_top=include_top,
            flat_output=flat_output,
        )
