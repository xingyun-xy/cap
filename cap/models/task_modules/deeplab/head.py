# Copyright (c) Changan Auto. All rights reserved.

import logging
from typing import Dict, List, Optional

import changan_plugin_pytorch as changan
import changan_plugin_pytorch.nn as hnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, SeparableConvModule2d
from cap.models.weight_init import normal_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["Deeplabv3plusHead"]

logger = logging.getLogger(__name__)


class ASPPModule(nn.Module):
    """ASPP Module for segmentation.

    Args:
        dilations: List of dilations for aspp.
        num_repeats: List of repeat for each branch.
        in_channels: input channels.
        feat_channels: Channels for each branch.
        bn_kwargs: Extra keyword arguments for bn layers.
        bias: Whether has bias. Default: True.
    """

    def __init__(
        self,
        dilations: List[int],
        num_repeats: List[int],
        in_channels: int,
        feat_channels: int,
        bn_kwargs: Dict,
        bias: Optional[bool] = True,
    ):
        super(ASPPModule, self).__init__()
        self.aspp = nn.ModuleList()
        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule2d(
                in_channels,
                feat_channels,
                1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(
                    feat_channels,
                    **bn_kwargs,
                ),
                act_layer=nn.ReLU(inplace=True),
            ),
        )

        for d, r in zip(dilations, num_repeats):
            if d > 1:
                self.aspp.append(
                    nn.Sequential(
                        *(
                            SeparableConvModule2d(
                                in_channels if i == 0 else feat_channels,
                                feat_channels,
                                3,
                                dilation=d,
                                padding=d,
                                bias=bias,
                                dw_norm_layer=nn.BatchNorm2d(
                                    in_channels if i == 0 else feat_channels,
                                    **bn_kwargs,
                                ),
                                dw_act_layer=nn.ReLU(inplace=True),
                                pw_norm_layer=nn.BatchNorm2d(
                                    feat_channels,
                                    **bn_kwargs,
                                ),
                                pw_act_layer=nn.ReLU(inplace=True),
                            )
                            for i in range(r)
                        )
                    )
                )
            else:
                self.aspp.append(
                    ConvModule2d(
                        in_channels,
                        feat_channels,
                        1,
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(
                            feat_channels,
                            **bn_kwargs,
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
        self.floatF = FloatFunctional()

    def forward(self, x):
        outs = []
        pooling = self.img_pool(x)
        pooling = F.interpolate(
            pooling, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        outs.append(pooling)
        for mod in self.aspp:
            out = mod(x)
            outs.append(out)
        return self.floatF.cap(outs, dim=1)


class Fusion(nn.Module):
    """Fusion Module for deeplab plus.

    Args:
        in_channels: input channels.
        feat_channels: Channels for output.
        c1_in_channels: C1 input channels.
        upsample_decode_scale: upsample scale to c1.
        bn_kwargs: Extra keyword arguments for bn layers.
        bias: Whether has bias. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        c1_in_channels: int,
        upsample_decode_scale: int,
        bn_kwargs: Dict,
        bias: Optional[int] = True,
    ):
        super(Fusion, self).__init__()
        self.bottleneck = SeparableConvModule2d(
            in_channels,
            feat_channels,
            3,
            padding=1,
            bias=bias,
            dw_norm_layer=nn.BatchNorm2d(
                in_channels,
                **bn_kwargs,
            ),
            dw_act_layer=nn.ReLU(inplace=True),
            pw_norm_layer=nn.BatchNorm2d(
                feat_channels,
                **bn_kwargs,
            ),
            pw_act_layer=None,
        )

        self.c1_bottleneck = nn.Sequential(
            SeparableConvModule2d(
                c1_in_channels,
                feat_channels,
                3,
                stride=1,
                padding=1,
                bias=bias,
                dw_norm_layer=nn.BatchNorm2d(
                    c1_in_channels,
                    **bn_kwargs,
                ),
                dw_act_layer=nn.ReLU(inplace=True),
                pw_norm_layer=nn.BatchNorm2d(
                    feat_channels,
                    **bn_kwargs,
                ),
                pw_act_layer=None,
            ),
        )

        self.resize = hnn.Interpolate(
            scale_factor=upsample_decode_scale,
            align_corners=None,
            recompute_scale_factor=True,
        )
        self.out_bottleneck = nn.Sequential(
            SeparableConvModule2d(
                feat_channels,
                feat_channels,
                3,
                padding=1,
                bias=bias,
                dw_norm_layer=nn.BatchNorm2d(
                    feat_channels,
                    **bn_kwargs,
                ),
                dw_act_layer=nn.ReLU(inplace=True),
                pw_norm_layer=nn.BatchNorm2d(
                    feat_channels,
                    **bn_kwargs,
                ),
                pw_act_layer=nn.ReLU(inplace=True),
            ),
        )
        self.floatF = FloatFunctional()

    def forward(self, inputs):
        c1_input = inputs[0]
        x = inputs[-1]
        c1_output = self.c1_bottleneck(c1_input)

        x = self.bottleneck(x)
        x = self.resize(x)
        return self.out_bottleneck(self.floatF.add(c1_output, x))


@OBJECT_REGISTRY.register
class Deeplabv3plusHead(nn.Module):
    """Head Module for FCN.

    Args:
        in_channels: Input channels.
        c1_index: Index for c1 input.
        c1_in_channels: In channels of c1.
        feat_channels: Channels for the module.
        num_classes: Number of classes.
        dilations: List of dilations for aspp.
        num_repeats: List of repeat for each branch of ASPP.
        argmax_output: Whether conduct argmax on output. Default: False.
        dequant_output: Whether to dequant output. Default: True
        int8_output: If True, output int8, otherwise output int32.
            Default: False.
        bn_kwargs: Extra keyword arguments for bn layers. Default: None.
        dropout_ratio: Ratio for dropout during training. Default: 0.1.
        upsample_decode_scale: upsample scale to c1. Default is 4.
        upsample_output_scale: Output upsample scale, only used in
            qat model, default is None.
        bias: Whether has bias. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        c1_index: int,
        c1_in_channels: int,
        feat_channels: int,
        num_classes: int,
        dilations: List[int],
        num_repeats: List[int],
        argmax_output: Optional[bool] = False,
        dequant_output: Optional[bool] = True,
        int8_output: Optional[bool] = True,
        bn_kwargs: Optional[Dict] = None,
        dropout_ratio: Optional[float] = 0.1,
        upsample_output_scale: Optional[int] = None,
        upsample_decode_scale: Optional[int] = 4,
        bias=True,
    ):
        super(Deeplabv3plusHead, self).__init__()
        self.argmax_output = argmax_output
        self.int8_output = int8_output
        self.bn_kwargs = bn_kwargs or {}

        self.in_channels = in_channels
        self.feat_channels = feat_channels

        self.dilations = dilations
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.dequant_output = dequant_output
        self.dequant = DeQuantStub()
        self.num_repeats = num_repeats

        self.c1_in_channels = c1_in_channels
        self.c1_index = c1_index
        self.bias = bias

        self.upsample_decode_scale = upsample_decode_scale

        if upsample_output_scale:
            self.resize = hnn.Interpolate(
                scale_factor=upsample_output_scale,
                align_corners=None,
                recompute_scale_factor=True,
            )
        self.upsample_output_scale = upsample_output_scale

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.aspp = ASPPModule(
            self.dilations,
            self.num_repeats,
            self.in_channels,
            self.feat_channels,
            self.bn_kwargs,
            self.bias,
        )
        self._init_seg_convs()
        self._init_predictor()

    def _init_seg_convs(self):
        self.seg_convs = Fusion(
            self.feat_channels * (len(self.dilations) + 1),
            self.feat_channels,
            self.c1_in_channels,
            self.upsample_decode_scale,
            self.bn_kwargs,
        )

    def _init_predictor(self):
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None

        self.cls_seg = ConvModule2d(
            self.feat_channels,
            self.num_classes,
            1,
            norm_layer=None,
            act_layer=None,
        )

    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.seg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.cls_seg.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)

    def forward(self, inputs):
        x = inputs[-1]
        c1_input = inputs[self.c1_index]
        x = self.aspp(x)

        x = self.seg_convs([c1_input, x])
        if self.dropout is not None:
            x = self.dropout(x)
        seg_pred = self.cls_seg(x)
        if self.training is not True:
            if self.upsample_output_scale:
                seg_pred = self.resize(seg_pred)
            if self.argmax_output:
                seg_pred = seg_pred.argmax(dim=1)
        if self.dequant_output:
            seg_pred = self.dequant(seg_pred)
        return seg_pred

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvModule2d) and hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        self.qconfig = changan.quantization.get_default_qat_qconfig()
        # disable output quantization for last quanti layer.
        if not self.int8_output:
            getattr(
                self.cls_seg, "0"
            ).qconfig = changan.quantization.get_default_qat_out_qconfig()
