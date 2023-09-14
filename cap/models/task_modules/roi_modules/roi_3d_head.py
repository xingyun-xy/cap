# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.backbones.mixvargenet import MixVarGENetConfig
from cap.models.base_modules import MixVarGEBlock
from cap.models.base_modules.basic_vargnet_module import OnePathResUnit
from cap.models.base_modules.conv_module import ConvModule2d
from cap.models.base_modules.extend_container import ExtSequential
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class RCNNHM3DVarGNetHead(nn.Module):
    """RCNN 3d pred head.

    Args:
        bn_kwargs: Kwargs of BN layer.
        dw_num_filter: Num filters of dw conv.
        pw_num_filter: Num filters of pw conv.
        pw_num_filter2: Num filters of second pw conv.
        roi_out_channel: Num of output channels of roi_feat_extractor,
            same as Num filters of input.
        group_base: Group base of group conv.
        factor: Factor of group conv.
        rot_channel: Num of output channels of rotation pred.
        is_dim_match: Whether to use dim match.
        stop_gradient Whether to stop gradient.
        with_iou_pred: Determines whether the output includes
            the iou pred.
        undistort_depth_uv: Whether to undistort depth branch into depth_u/v.
        int8_output: Determines whether the dtype of output is int8.
    """

    def __init__(
        self,
        bn_kwargs: Dict,
        dw_num_filter: int,
        pw_num_filter: int,
        pw_num_filter2: int,
        roi_out_channel: int,
        group_base: int,
        factor: float,
        rot_channel: int,
        is_dim_match: bool,
        stop_gradient: bool,
        with_iou_pred: bool,
        undistort_depth_uv: bool = False,
        int8_output: bool = False,
    ):
        super(RCNNHM3DVarGNetHead, self).__init__()

        self.stop_gradient = stop_gradient
        self.with_iou_pred = with_iou_pred
        self.undistort_depth_uv = undistort_depth_uv
        self.int8_output = int8_output

        self.dequant = DeQuantStub()

        self.rcnn_conv = OnePathResUnit(
            in_filter=roi_out_channel,
            dw_num_filter=dw_num_filter,
            group_base=group_base,
            pw_num_filter=pw_num_filter,
            pw_num_filter2=pw_num_filter2,
            bn_kwargs=bn_kwargs,
            stride=(1, 1),
            is_dim_match=is_dim_match,
            use_bias=True,
            pw_with_act=True,
            factor=factor,
        )
        self.offset3d_conv = ExtSequential([])
        self.dim_conv = ExtSequential([])
        if self.undistort_depth_uv:
            self.depth_u_conv = ExtSequential([])
            self.depth_v_conv = ExtSequential([])
        else:
            self.depth_conv = ExtSequential([])
        self.rot_conv = ExtSequential([])

        self.label_out_block = ConvModule2d(
            in_channels=pw_num_filter2,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            act_layer=None,
        )

        if self.with_iou_pred:
            self.iou_conv = ExtSequential([])
            self.iou_conv.add_module(
                "conv1",
                ConvModule2d(
                    in_channels=pw_num_filter2,
                    out_channels=16,
                    kernel_size=(4, 4),
                    stride=2,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(16, **bn_kwargs),
                    act_layer=None,
                ),
            )
            self.iou_conv.add_module(
                "conv2",
                ConvModule2d(
                    in_channels=16,
                    out_channels=1,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=0,
                    act_layer=None,
                ),
            )

        self.offset_2d_out_block = ConvModule2d(
            in_channels=pw_num_filter2,
            out_channels=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            act_layer=None,
        )

        self.offset3d_conv.add_module(
            "conv1",
            ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=pw_num_filter2,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
                act_layer=None,
            ),
        )

        self.offset3d_conv.add_module(
            "conv2",
            ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                act_layer=None,
            ),
        )

        if self.undistort_depth_uv:
            self.depth_u_conv.add_module(
                "conv1",
                ConvModule2d(
                    in_channels=pw_num_filter2,
                    out_channels=pw_num_filter2,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
                    act_layer=None,
                ),
            )
            self.depth_u_conv.add_module(
                "conv2",
                ConvModule2d(
                    in_channels=pw_num_filter2,
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    act_layer=None,
                ),
            )

            self.depth_v_conv.add_module(
                "conv1",
                ConvModule2d(
                    in_channels=pw_num_filter2,
                    out_channels=pw_num_filter2,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
                    act_layer=None,
                ),
            )
            self.depth_v_conv.add_module(
                "conv2",
                ConvModule2d(
                    in_channels=pw_num_filter2,
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    act_layer=None,
                ),
            )
        else:
            self.depth_conv.add_module(
                "conv1",
                ConvModule2d(
                    in_channels=pw_num_filter2,
                    out_channels=pw_num_filter2,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
                    act_layer=None,
                ),
            )
            self.depth_conv.add_module(
                "conv2",
                ConvModule2d(
                    in_channels=pw_num_filter2,
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    act_layer=None,
                ),
            )

        self.dim_conv.add_module(
            "conv1",
            ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=16,
                kernel_size=(4, 4),
                stride=2,
                padding=0,
                norm_layer=nn.BatchNorm2d(16, **bn_kwargs),
                act_layer=None,
            ),
        )

        self.dim_conv.add_module(
            "conv2",
            ConvModule2d(
                in_channels=16,
                out_channels=3,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                act_layer=None,
            ),
        )

        self.rot_conv.add_module(
            "conv1",
            ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=16,
                kernel_size=(4, 4),
                stride=2,
                padding=0,
                norm_layer=nn.BatchNorm2d(16, **bn_kwargs),
                act_layer=None,
            ),
        )

        self.rot_conv.add_module(
            "conv2",
            ConvModule2d(
                in_channels=16,
                out_channels=rot_channel,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                act_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor):
        # TODO: ensure the shape of inputs are 8 x 8.
        assert x.shape[2:] == (8, 8)
        if self.stop_gradient:
            x = x.detach()
        x = self.rcnn_conv(x)

        cls_pred = self.dequant(self.label_out_block(x))
        offset_2d_pred = self.dequant(self.offset_2d_out_block(x))
        offset_3d_pred = self.dequant(self.offset3d_conv(x))

        rot_pred = self.dequant(self.rot_conv(x))
        dims_pred = self.dequant(self.dim_conv(x))
        if self.undistort_depth_uv:
            depth_u_pred = self.dequant(self.depth_u_conv(x))
            depth_v_pred = self.dequant(self.depth_v_conv(x))
            output = OrderedDict(
                cls_pred=cls_pred,
                offset_2d_pred=offset_2d_pred,
                offset_3d_pred=offset_3d_pred,
                depth_u_pred=depth_u_pred,
                depth_v_pred=depth_v_pred,
                dims_pred=dims_pred,
                rot_pred=rot_pred,
            )
        else:
            depth_pred = self.dequant(self.depth_conv(x))
            output = OrderedDict(
                cls_pred=cls_pred,
                offset_2d_pred=offset_2d_pred,
                offset_3d_pred=offset_3d_pred,
                depth_pred=depth_pred,
                dims_pred=dims_pred,
                rot_pred=rot_pred,
            )

        if self.with_iou_pred:
            iou_pred = self.dequant(self.iou_conv(x))
            output["iou_pred"] = iou_pred

        return output

    def fuse_model(self):
        for m in self.children():
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self.label_out_block.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.offset_2d_out_block.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.offset3d_conv.conv2.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.dim_conv.conv2.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            if self.undistort_depth_uv:
                self.depth_u_conv.conv2.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
                self.depth_v_conv.conv2.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
            else:
                self.depth_conv.conv2.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
            self.rot_conv.conv2.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            if self.with_iou_pred:
                self.iou_conv.conv2.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )


@OBJECT_REGISTRY.register
class RCNNHM3DMixVarGEHead(RCNNHM3DVarGNetHead):
    """RCNN 3d pred head with MixVarGEBlock.

    The difference between RCNNHM3DVarGNetHead is that
    RCNNHM3DMixVarGEHead adopts the stacked MixVarGEBlock
    units as shared convs.

    Args:
        bn_kwargs: Kwargs of BN layer.
        mid_num_filter: Num filters of the last shared conv.
        rot_channel: Num of output channels of rotation pred.
        stop_gradient Whether to stop gradient.
        with_iou_pred: Determines whether the output includes
            the iou pred.
        head_config: Setting list of the stacked MixVarGEBlocks.
        undistort_depth_uv: Whether to undistort depth branch into depth_u/v.
        int8_output: Determines whether the dtype of output is int8.
    """

    def __init__(
        self,
        bn_kwargs: Dict,
        mid_num_filter: int,
        rot_channel: int,
        stop_gradient: bool,
        with_iou_pred: bool,
        head_config: List[MixVarGENetConfig],
        undistort_depth_uv: bool = False,
        int8_output: bool = False,
    ):
        super(RCNNHM3DMixVarGEHead, self).__init__(
            bn_kwargs=bn_kwargs,
            dw_num_filter=mid_num_filter,
            pw_num_filter=mid_num_filter,
            pw_num_filter2=mid_num_filter,
            roi_out_channel=16,
            group_base=mid_num_filter,
            factor=1,
            rot_channel=rot_channel,
            is_dim_match=True,
            stop_gradient=stop_gradient,
            with_iou_pred=with_iou_pred,
            undistort_depth_uv=undistort_depth_uv,
            int8_output=int8_output,
        )

        assert (
            len(head_config) > 0
            and mid_num_filter == head_config[-1].out_channels
        )

        layers = []
        for config_i in head_config:
            if len(config_i.fusion_strides) != 0:
                raise NotImplementedError("unsupport downsample fusion")
            layers.append(
                MixVarGEBlock(
                    in_ch=config_i.in_channels,
                    block_ch=config_i.out_channels,
                    head_op=config_i.head_op,
                    stack_ops=config_i.stack_ops,
                    stack_factor=config_i.stack_factor,
                    stride=config_i.stride,
                    bias=True,
                    fusion_channels=(),
                    downsample_num=config_i.extra_downsample_num,
                    output_downsample=False,
                    bn_kwargs=bn_kwargs,
                )
            )
        self.rcnn_conv = ExtSequential(layers)
