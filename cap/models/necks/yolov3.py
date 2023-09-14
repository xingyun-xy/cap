# Copyright (c) Changan Auto. All rights reserved.

import changan_plugin_pytorch as changan
import torch.nn as nn

from cap.models.base_modules import ConvModule2d
from cap.models.weight_init import xavier_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["YOLOV3Neck"]


class InputModule(nn.Sequential):
    """
    Input module of yolov3 neck.

    Args:
        input_channels (int): Input channels of this module.
        output_channels (int): Output channels of this module.
        bn_kwargs (dict): Dict for BN layers.
        bias (bool): Whether to use bias in this module.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bn_kwargs: dict,
        bias: bool = True,
    ):
        conv_list = []
        for i in range(2):
            conv_list.append(
                ConvModule2d(
                    input_channels if i == 0 else output_channels * 2,
                    output_channels,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(output_channels, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            conv_list.append(
                ConvModule2d(
                    output_channels,
                    output_channels * 2,
                    3,
                    padding=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(
                        output_channels * 2, **bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        conv_list.append(
            ConvModule2d(
                output_channels * 2,
                output_channels,
                1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(output_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )
        )
        super(InputModule, self).__init__(*conv_list)
        self.init_weight()

    def init_weight(self):
        for i in range(5):
            for m in getattr(self, "%d" % (i)):
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution="uniform")

    def fuse_model(self):
        for i in range(5):
            getattr(self, "%d" % (i)).fuse_model()


@OBJECT_REGISTRY.register
class YOLOV3Neck(nn.Module):
    """
    Necks module of yolov3.

    Args:
        backbone_idx (list): Index of backbone output for necks.
        in_channels_list (list): List of input channels.
        out_channels_list (list): List of output channels.
        bn_kwargs (dict): Config dict for BN layer.
        bias (bool): Whether to use bias in module.
    """

    def __init__(
        self,
        backbone_idx: list,
        in_channels_list: list,
        out_channels_list: list,
        bn_kwargs: dict,
        bias: bool = True,
    ):
        super(YOLOV3Neck, self).__init__()
        assert len(backbone_idx) == len(in_channels_list)
        assert len(in_channels_list) == len(out_channels_list)
        self.backbone_idx = backbone_idx

        for i, channels in enumerate(out_channels_list):
            if i > 0:
                in_channels_list[i] += out_channels_list[i - 1]

            self.add_module(
                "module%d" % (i),
                InputModule(
                    in_channels_list[i],
                    channels,
                    bn_kwargs=bn_kwargs,
                    bias=bias,
                ),
            )
            self.add_module(
                "conv%d" % (i),
                ConvModule2d(
                    channels,
                    channels * 2,
                    3,
                    padding=1,
                    stride=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(channels * 2, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                ),
            )

            if i < len(self.backbone_idx) - 1:
                self.add_module(
                    "concat%d" % (i), nn.quantized.FloatFunctional()
                )
                self.add_module(
                    "resize%d" % (i),
                    changan.nn.Interpolate(
                        scale_factor=2, recompute_scale_factor=True
                    ),
                )
                self.add_module(
                    "trans_conv%d" % (i),
                    ConvModule2d(
                        channels,
                        channels,
                        1,
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(channels, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    ),
                )
        self.init_weight()

    def init_weight(self):
        for i, _idx in enumerate(self.backbone_idx):
            getattr(self, "module%d" % (i)).init_weight()
            for m in getattr(self, "conv%d" % (i)):
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution="uniform")
            if i < len(self.backbone_idx) - 1:
                for m in getattr(self, "trans_conv%d" % (i)):
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution="uniform")

    def forward(self, x):
        outputs = []
        input = x[self.backbone_idx[0]]
        for i, _idx in enumerate(self.backbone_idx):
            out = getattr(self, "module%d" % (i))(input)
            outputs.append(getattr(self, "conv%d" % (i))(out))
            if i < len(self.backbone_idx) - 1:
                out = getattr(self, "trans_conv%d" % (i))(out)
                out = getattr(self, "resize%d" % (i))(out)
                input = getattr(self, "concat%d" % (i)).cap(
                    (out, x[self.backbone_idx[i + 1]]), dim=1
                )
        return outputs

    def fuse_model(self):
        for i in range(len(self.backbone_idx)):
            getattr(self, "module%d" % (i)).fuse_model()
            getattr(self, "conv%d" % (i)).fuse_model()
            if i < len(self.backbone_idx) - 1:
                getattr(self, "trans_conv%d" % (i)).fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
