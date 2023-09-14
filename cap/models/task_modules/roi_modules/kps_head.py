from collections import OrderedDict

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn import Interpolate
from torch.quantization import DeQuantStub

from cap.models.base_modules.conv_module import ConvModule2d
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class RCNNKPSSplitHead(nn.Module):
    """RCNN keypoints split head is not shared within a task group.

    Args:
        points_num: number of keypoints to be predicted.
        in_channel: number of channels of input feature maps.
        upscale: determines whether upscale the input.
        int8_output: determines whether the dtype of output is int8.
    """

    def __init__(
        self,
        points_num: int,
        in_channel: int,
        upscale: bool = False,
        int8_output: bool = False,
    ):

        super().__init__()

        self.int8_output = int8_output

        self.dequant = DeQuantStub()

        if upscale:
            self.upscale = Interpolate(
                scale_factor=2,
                align_corners=False,
                recompute_scale_factor=True,
            )

        self.label_out_block = ConvModule2d(
            in_channels=in_channel,
            out_channels=points_num,
            kernel_size=1,
            stride=1,
            padding=0,
            act_layer=None,
        )
        self.pos_offset_out_block = ConvModule2d(
            in_channels=in_channel,
            out_channels=points_num * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            act_layer=None,
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:

        if hasattr(self, "upscale"):
            x = self.upscale(x)

        rcnn_cls_pred = self.dequant(self.label_out_block(x))
        rcnn_reg_pred = self.dequant(self.pos_offset_out_block(x))

        output = OrderedDict(
            kps_rcnn_cls_pred=rcnn_cls_pred, kps_rcnn_reg_pred=rcnn_reg_pred
        )

        return output

    def fuse_model(self):
        self.label_out_block.fuse_model()
        self.pos_offset_out_block.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self.label_out_block.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.pos_offset_out_block.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
