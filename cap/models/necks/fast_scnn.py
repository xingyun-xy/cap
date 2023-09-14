# Copyright (c) Changan Auto. All rights reserved.

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional

from cap.models.base_modules import ConvModule2d, SeparableConvModule2d
from cap.models.weight_init import normal_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["FastSCNNNeck"]

logger = logging.getLogger(__name__)


class PPM(nn.Module):
    """Pooling Pyramid Module.

    Args:
        pool_scales: Pooling scales used in Pooling Pyramid
            Module.
        in_channels: Input channels.
        feat_channels: Channels after modules.
        bn_kwargs: Dict for Bn layer
    """

    def __init__(
        self,
        pool_scales: List[int],
        in_channels: int,
        feat_channels: int,
        bn_kwargs: Dict,
    ):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.float_func = FloatFunctional()
        self.poolings = nn.ModuleList()
        for _ in pool_scales:
            self.poolings.append(
                ConvModule2d(
                    in_channels,
                    feat_channels // 4,
                    1,
                    norm_layer=nn.BatchNorm2d(feat_channels // 4, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        self.project = ConvModule2d(
            feat_channels + in_channels,
            feat_channels,
            3,
            padding=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(
                feat_channels,
                **bn_kwargs,
            ),
            act_layer=nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm, scale in zip(self.poolings, self.pool_scales):
            input_size = torch.Tensor((x.shape[2], x.shape[3])).cpu().numpy()
            strides = np.floor(input_size / scale).astype(np.int32)
            kernels = (input_size - (scale - 1) * strides).astype(np.int32)
            ppm_out = F.avg_pool2d(
                x,
                kernel_size=(kernels[0], kernels[1]),
                stride=(strides[0], strides[1]),
            )
            ppm_out = ppm(ppm_out)
            upsampled_ppm_out = F.interpolate(
                ppm_out, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            ppm_outs.append(upsampled_ppm_out)
        ppm_out = self.float_func.cap((*ppm_outs, x), dim=1)
        ppm_out = self.project(ppm_out)
        return ppm_out

    def fuse_model(self):
        for mod in self.poolings:
            mod.fuse_model()
        self.project.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


class Fusion(nn.Module):
    """Fusion Module.

    Args:
        in_channels: Input channels.
        feat_channels: Channels after modules.
        bn_kwargs: Dict for Bn layer
    """

    def __init__(self, in_channels: int, feat_channels: int, bn_kwargs: Dict):
        super(Fusion, self).__init__()
        self.project = ConvModule2d(
            in_channels,
            feat_channels,
            3,
            padding=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(
                feat_channels,
                **bn_kwargs,
            ),
            act_layer=None,
        )

        self.upsample = nn.Sequential(
            nn.Upsample(
                scale_factor=4,
                align_corners=False,
                mode="bilinear",
            ),
            SeparableConvModule2d(
                feat_channels,
                feat_channels,
                3,
                padding=1,
                bias=True,
                dw_norm_layer=nn.BatchNorm2d(
                    feat_channels,
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
        self.relu = nn.ReLU(inplace=True)
        self.fusion = SeparableConvModule2d(
            feat_channels,
            feat_channels,
            3,
            padding=1,
            bias=True,
            dw_norm_layer=nn.BatchNorm2d(
                feat_channels,
                **bn_kwargs,
            ),
            dw_act_layer=None,
            pw_norm_layer=nn.BatchNorm2d(
                feat_channels,
                **bn_kwargs,
            ),
            pw_act_layer=nn.ReLU(inplace=True),
        )
        self.float_func = FloatFunctional()

    def forward(self, x, low):
        x = self.project(x)
        low = self.upsample(low)

        return self.fusion(self.relu(self.float_func.add(x, low)))

    def fuse_model(self):
        self.project.fuse_model()
        self.upsample[1].fuse_model()
        self.fusion.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class FastSCNNNeck(nn.Module):
    """
    Upper neck module for segmentation.

    Args:
        in_channels: channels of each input feature map
        feat_channels: channels for featture maps.
        indexes: indexes of inputs.
        bn_kwargs: Dict for Bn layer.
    """

    def __init__(
        self,
        in_channels: List[int],
        feat_channels: List[int],
        indexes: List[int],
        bn_kwargs: Optional[Dict] = None,
    ):
        super(FastSCNNNeck, self).__init__()
        self.bn_kwargs = bn_kwargs or {}
        self.indexes = indexes

        in_channels = in_channels[::-1]
        feat_channels = feat_channels[::-1]
        self._init_layers(in_channels, feat_channels)
        self._init_weights()

    def _init_layers(self, in_channels, feat_channels):
        self._init_extract(in_channels[0], feat_channels[0])
        self._init_seg_convs(in_channels[1:], feat_channels[1:])

    def _init_extract(self, in_channels, feat_channels):
        self.extract = PPM(
            pool_scales=[1, 2, 4, 8],
            in_channels=in_channels,
            feat_channels=feat_channels,
            bn_kwargs=self.bn_kwargs,
        )

    def _init_seg_convs(self, in_channels, feat_channels):
        self.seg_convs = nn.ModuleList(
            Fusion(in_channels[i], feat_channels[i], self.bn_kwargs)
            for i in range(len(in_channels))
        )

    def _init_weights(self):
        """Initialize weights of the neck."""
        for m in self.seg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, inputs):
        outs = []
        inputs = [inputs[i] for i in self.indexes]
        inputs = inputs[::-1]
        x = self.extract(inputs[0])
        outs.append(inputs[-1])
        outs.append(x)
        for up, mod in zip(inputs[1:], self.seg_convs):
            x = mod(up, x)
        outs.append(x)
        return outs

    def fuse_model(self):
        self.extract.fuse_model()
        for mod in self.seg_convs:
            mod.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
