# Copyright (c) Changan Auto. All rights reserved.

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, SeparableConvModule2d
from cap.models.weight_init import normal_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["FCNHead", "DepthwiseSeparableFCNHead"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FCNHead(nn.Module):
    """Head Module for FCN.

    Args:
        input_index: Index of inputs.
        in_channels: Input channels.
        feat_channels: Channels for the module.
        num_classes: Number of classes.
        dropout_ratio: Ratio for dropout during training. Default: 0.1.
        int8_output: If True, output int8, otherwise output int32.
            Default: False.
        argmax_output: Whether conduct argmax on output. Default: False.
        dequant_output: Whether to dequant output. Default: True.
        upsample_output_scale: Output upsample scale. Default: None.
        num_convs: number of convs in head. Default: 2.
        bn_kwargs: Extra keyword arguments for bn layers. Default: None.
    """

    def __init__(
        self,
        input_index: int,
        in_channels: int,
        feat_channels: int,
        num_classes: int,
        dropout_ratio: Optional[float] = 0.1,
        int8_output: Optional[bool] = False,
        argmax_output: Optional[bool] = False,
        dequant_output: Optional[bool] = True,
        upsample_output_scale: Optional[int] = None,
        num_convs: Optional[int] = 2,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(FCNHead, self).__init__()
        self.input_index = input_index
        self.dropout_ratio = dropout_ratio
        self.dequant_output = dequant_output
        self.dequant = DeQuantStub()
        self.argmax_output = argmax_output
        self.int8_output = int8_output
        self.bn_kwargs = bn_kwargs or {}

        if upsample_output_scale:
            self.resize = torch.nn.Upsample(
                scale_factor=upsample_output_scale,
                align_corners=False,
                mode="bilinear",
            )
        self.upsample_output_scale = upsample_output_scale

        self.convs = nn.Sequential(
            ConvModule2d(
                in_channels,
                feat_channels,
                kernel_size=3,
                padding=1,
                norm_layer=nn.BatchNorm2d(
                    feat_channels,
                    **self.bn_kwargs,
                ),
                act_layer=nn.ReLU(inplace=True),
            ),
            *[
                ConvModule2d(
                    feat_channels,
                    feat_channels,
                    kernel_size=3,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(
                        feat_channels,
                        **self.bn_kwargs,
                    ),
                    act_layer=nn.ReLU(inplace=True),
                )
                for _ in range(num_convs)
            ],
        )

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.cls_seg = ConvModule2d(
            feat_channels,
            num_classes,
            1,
            norm_layer=None,
            act_layer=None,
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.cls_seg.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)

    def forward(self, inputs: List[torch.Tensor]):
        x = inputs[self.input_index]
        x = self.convs(x)
        if self.dropout:
            x = self.dropout(x)
        seg_pred = self.cls_seg(x)
        if self.training:
            if self.upsample_output_scale:
                seg_pred = self.resize(seg_pred)
            if self.argmax_output:
                seg_pred = seg_pred.argmax(dim=1)
        if self.dequant_output:
            seg_pred = self.dequant(seg_pred)
        return seg_pred

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvModule2d):
                m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        # disable output quantization for last quanti layer.
        if not self.int8_output:
            getattr(
                self.cls_seg, "0"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class DepthwiseSeparableFCNHead(FCNHead):
    def __init__(self, in_channels, feat_channels, num_convs=1, **kwargs):
        """Head Module for DepthwiseSeparableFCNHead.

        Args:
            in_channels: Input channels.
            feat_channels: Channels for the module.
            num_convs: number of convs in head. Default: 2.
        """

        super(DepthwiseSeparableFCNHead, self).__init__(
            in_channels=in_channels, feat_channels=feat_channels, **kwargs
        )
        self.convs = nn.Sequential(
            SeparableConvModule2d(
                in_channels,
                feat_channels,
                3,
                padding=1,
                bias=True,
                dw_norm_layer=nn.BatchNorm2d(
                    in_channels,
                    **self.bn_kwargs,
                ),
                dw_act_layer=None,
                pw_norm_layer=nn.BatchNorm2d(
                    feat_channels,
                    **self.bn_kwargs,
                ),
                pw_act_layer=nn.ReLU(inplace=True),
            ),
            *[
                SeparableConvModule2d(
                    feat_channels,
                    feat_channels,
                    3,
                    padding=1,
                    bias=True,
                    dw_norm_layer=nn.BatchNorm2d(
                        feat_channels,
                        **self.bn_kwargs,
                    ),
                    dw_act_layer=None,
                    pw_norm_layer=nn.BatchNorm2d(
                        feat_channels,
                        **self.bn_kwargs,
                    ),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                for _ in range(1, num_convs)
            ],
        )
