# Copyright (c) Changan Auto. All rights reserved.
from collections import OrderedDict
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d
from cap.models.utils import _take_features
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, _is_increasing_sequence

__all__ = ["ElevationHead", "GroundHead"]


@OBJECT_REGISTRY.register
class GroundHead(nn.Module):
    """Build GroundHead.

    GroundHead taske 1/2^5 downscale features as input, and output
    the norm of plane, which shape is (b, 1, 3, 1).

    Args:
        in_strides: a list contains the strides of
            feature maps from backbone or neck.
        stride2channels: a stride to channel dict.
        bn_kwargs: batch norm parameters, 'eps' and 'momentum'.
        input_shape: input shape of img, in (h, w) order.
        take_strides: a list contains the strides of
            feature maps used in ResidualFlowPoseHead.
        feature_name: Name of features from backbone(or neck) in dict.
        use_bias: if use bias in ConvModule2d.
    """

    def __init__(
        self,
        in_strides: Sequence,
        stride2channels: Mapping,
        bn_kwargs: Mapping,
        input_shape: Sequence,
        take_strides: Sequence,
        feature_name: str,
        forward_frame_idxs: Sequence,
        use_bias: bool = False,
        **kwargs
    ):
        super(GroundHead, self).__init__(**kwargs)
        assert _is_increasing_sequence(in_strides), in_strides
        assert _is_increasing_sequence(take_strides), take_strides

        self.in_strides = _as_list(in_strides)
        self.take_strides = _as_list(take_strides)
        # use the maximum stride in take_strides as the out_stride
        self.out_stride = self.take_strides[0]
        self.feature_name = feature_name
        self.forward_frame_idxs = forward_frame_idxs[0]
        self.ground_convs = nn.Sequential(
            ConvModule2d(
                stride2channels[in_strides[-1]],
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm_layer=nn.BatchNorm2d(256, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm_layer=nn.BatchNorm2d(256, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm_layer=nn.BatchNorm2d(256, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
        )

        self.ground_conv = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(
                3,
                3,
                kernel_size=(
                    input_shape[1] // self.take_strides[-1],
                    input_shape[0] // self.take_strides[-1],
                ),
            ),
        )
        self.ground_dequant = DeQuantStub()

    def forward(self, data: Mapping):
        res = OrderedDict()
        res.update(OrderedDict(pred_ground=[]))
        feature = data[self.feature_name][self.forward_frame_idxs]
        ground = self.forward_ground(feature)
        res["pred_ground"].append(ground)
        return res

    def forward_ground(self, features: Sequence):
        features = _take_features(features, self.in_strides, self.take_strides)

        ground_fea = self.ground_convs(features[-1])
        ground = self.ground_conv(ground_fea)

        ground = self.ground_dequant(ground)
        return ground

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.ground_conv[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        out_next_stride = self.take_strides[
            self.take_strides.index(self.out_stride) + 1
        ]
        fusion_name = "fusion_{}_to_{}".format(
            out_next_stride, self.out_stride
        )
        getattr(
            self, fusion_name
        ).flow_add.qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def fuse_model(self):
        for m in self.ground_convs:
            if hasattr(m, "fuse_model"):
                m.fuse_model()


@OBJECT_REGISTRY.register
class ElevationHead(nn.Module):
    """Build ElevationHead.

    The ElevationHead includec two sub-head: gamma_head and ground_head,
    the gamma_head output gamma map which can be decomposed as depth
    map and height map, the ground_head output the norm of ground plane
    which can be used in the process of gamma to depth/height. The
    ground_head is optional, when ground_head is None, using calibrated
    normal.

    Args:
        feature_name: Name of features from backbone(or neck)
            in input dict.
        gamma_head: gamma head.
        ground_head: ground head.
    """

    def __init__(
        self,
        feature_name: str,
        gamma_head: torch.nn.Module,
        ground_head: Optional[torch.nn.Module] = None,
        **kwargs
    ):
        super(ElevationHead, self).__init__(**kwargs)
        self.feature_name = feature_name
        self.gamma_head = gamma_head
        self.ground_head = ground_head

    def forward(self, data: Mapping):
        res = self.gamma_head(data)
        if self.ground_head:
            res.update(self.ground_head(data))
        return res

    def fuse_model(self):
        self.gamma_head.fuse_model()
        self.ground_head.fuse_model()

    def set_qconfig(self):
        self.gamma_head.set_qconfig()
        self.ground_head.fuse_model()
