# Copyright (c) Changan Auto. All rights reserved.

import torch.nn as nn
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, InvertedResidual
from cap.registry import OBJECT_REGISTRY

__all__ = ["MobileNetV2"]


@OBJECT_REGISTRY.register
class MobileNetV2(nn.Module):
    """
    A module of mobilenetv2.

    Args:
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        alpha (float): Alpha for mobilenetv1.
        bias (bool): Whether to use bias in module.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
        use_dw_as_avgpool (bool): Whether to replace AvgPool with DepthWiseConv
    """

    def __init__(
        self,
        num_classes,
        bn_kwargs: dict,
        alpha: float = 1.0,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
        use_dw_as_avgpool: bool = False,
    ):
        super(MobileNetV2, self).__init__()
        self.alpha = alpha
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.use_dw_as_avgpool = use_dw_as_avgpool

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        in_chls = [
            [32],
            [16, 24],
            [24, 32, 32],
            [32] + [64] * 4 + [96] * 2,
            [96] + [160] * 3,
        ]
        out_chls = [
            [16],
            [24, 24],
            [32, 32, 32],
            [64] * 4 + [96] * 3,
            [160] * 3 + [320],
        ]

        self.mod1 = self._make_stage(in_chls[0], out_chls[0], 1, True)
        self.mod2 = self._make_stage(in_chls[1], out_chls[1], 6)
        self.mod3 = self._make_stage(in_chls[2], out_chls[2], 6)
        self.mod4 = self._make_stage(in_chls[3], out_chls[3], 6)
        self.mod5 = self._make_stage(in_chls[4], out_chls[4], 6)

        if self.use_dw_as_avgpool:
            pool_layer = ConvModule2d(
                in_channels=max(1280, int(1280 * alpha)),
                out_channels=max(1280, int(1280 * alpha)),
                kernel_size=7,
                stride=1,
                padding=0,
                groups=max(1280, int(1280 * alpha)),
            )
        else:
            pool_layer = nn.AvgPool2d(7)

        if self.include_top:
            self.output = nn.Sequential(
                ConvModule2d(
                    int(out_chls[4][-1] * alpha),
                    max(1280, int(1280 * alpha)),
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(
                        max(1280, int(1280 * alpha)), **bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
                pool_layer,
                ConvModule2d(
                    max(1280, int(1280 * alpha)),
                    num_classes,
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(self, in_chls, out_chls, expand_t, first_layer=False):
        layers = []
        in_chls = [int(chl * self.alpha) for chl in in_chls]
        out_chls = [int(chl * self.alpha) for chl in out_chls]
        for i, in_chl, out_chl in zip(range(len(in_chls)), in_chls, out_chls):
            stride = 2 if i == 0 else 1
            if first_layer:
                layers.append(
                    ConvModule2d(
                        3,
                        in_chls[0],
                        3,
                        stride,
                        1,
                        bias=self.bias,
                        norm_layer=nn.BatchNorm2d(
                            in_chls[0], **self.bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                stride = 1
            layers.append(
                InvertedResidual(
                    in_chl,
                    out_chl,
                    stride,
                    expand_t,
                    self.bn_kwargs,
                    self.bias,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        x = self.quant(x)
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
        modules = [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]
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
