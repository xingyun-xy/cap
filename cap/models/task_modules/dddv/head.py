# Copyright (c) Changan Auto. All rights reserved.
from collections import OrderedDict
from typing import Mapping, Sequence, Union

import changan_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from changan_plugin_pytorch.qtensor import QTensor
from torch.quantization import DeQuantStub

from cap.models.base_modules import (
    BasicVarGBlock,
    ConvModule2d,
    SeparableConvModule2d,
)
from cap.models.utils import _take_features
from cap.registry import OBJECT_REGISTRY
from cap.utils import qconfig_manager
from cap.utils.apply_func import _as_list, _is_increasing_sequence

__all__ = ["PixelHead", "DepthPoseResflowHead"]


class FusionResidualFlowBlock(nn.Module):
    """
    Build fusion block used in ResidualFlowPoseHead.

    Args:
        channels  (int): Feature map channle.
        upsampling (bool): Wheather upsample resflow.
            Only setting False in stride32.
        dequant_out (bool): Wheather dequanti output.
            Only setting True in stride4.
    """

    def __init__(
        self,
        channels: int,
        bn_kwargs: Mapping,
        upsampling: bool = True,
        dequant_out: bool = False,
        use_bias: bool = False,
        factor: int = 2,
        group_base: int = 8,
    ):
        super(FusionResidualFlowBlock, self).__init__()
        assert channels % group_base == 0
        self.upsampling = upsampling
        self.dequant_out = dequant_out
        self.fusion_block = BasicVarGBlock(
            in_channels=channels,
            mid_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            factor=factor,
            group_base=group_base,
            merge_branch=False,
        )

        self.upsampling = upsampling
        if upsampling:
            self.upsampling_op = hnn.Interpolate(
                scale_factor=2, recompute_scale_factor=True
            )

        self.pw_conv = ConvModule2d(
            in_channels=3,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            norm_layer=nn.BatchNorm2d(channels, **bn_kwargs),
        )
        self.pw_add = nn.quantized.FloatFunctional()

        self.flow_conv = ConvModule2d(
            in_channels=channels,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm_layer=nn.BatchNorm2d(3, **bn_kwargs),
        )
        self.flow_add = nn.quantized.FloatFunctional()
        self.dequant = DeQuantStub()

    def forward(
        self,
        residual_flow: Union[torch.Tensor, QTensor],
        feature: Union[torch.Tensor, QTensor],
    ):
        upsampling_residual_flow = (
            self.upsampling_op(residual_flow)
            if self.upsampling
            else residual_flow
        )
        fusion_feature = self.pw_conv(upsampling_residual_flow)
        fusion_feature = self.pw_add.add(fusion_feature, feature)
        fusion_feature = self.fusion_block(fusion_feature)

        fusion_feature = self.flow_conv(fusion_feature)
        result_residual_flow = self.flow_add.add(
            fusion_feature, upsampling_residual_flow
        )

        if self.dequant_out:
            result_residual_flow = self.dequant(result_residual_flow)

        return result_residual_flow

    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        self.fusion_block.fuse_model()
        torch.quantization.fuse_modules(
            self,
            ["pw_conv.0", "pw_conv.1", "pw_add"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )
        torch.quantization.fuse_modules(
            self,
            ["flow_conv.0", "flow_conv.1", "flow_add"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


@OBJECT_REGISTRY.register
class ResidualFlowPoseHead(nn.Module):
    """
    Build ResidualFlowPoseHead.

    Args:
        in_strides (list): a list contains the strides of
            feature maps from backbone or neck.
        take_strides (list): a list contains the strides of
            feature maps used in ResidualFlowPoseHead.
        input_shape (list/tuple): input shape of img, in (h, w) order.
        feature_name (str): Name of features from backbone(or neck) in dict.
        stride2channels (dict): a stride to channel dict.
        has_resflow (bool): output res_flow or not.

    """

    def __init__(
        self,
        in_strides: Sequence,
        stride2channels: Mapping,
        bn_kwargs: Mapping,
        input_shape: Sequence,
        take_strides: Sequence = (4, 8, 16, 32),
        feature_name: str = "feats",
        has_resflow: bool = True,
        use_bias: bool = False,
        factor: int = 2,
        group_base: int = 8,
        **kwargs,
    ):
        super(ResidualFlowPoseHead, self).__init__(**kwargs)
        assert _is_increasing_sequence(in_strides), in_strides
        assert _is_increasing_sequence(take_strides), take_strides

        self.in_strides = _as_list(in_strides)
        self.take_strides = _as_list(take_strides)
        # use the maximum stride in take_strides as the out_stride
        self.out_stride = self.take_strides[0]
        self.has_resflow = has_resflow
        self.feature_name = feature_name
        if self.has_resflow:
            for i, stride_i in enumerate(self.take_strides):
                is_last_stride = (
                    self.take_strides.index(stride_i)
                    == len(self.take_strides) - 1
                )
                if is_last_stride:
                    fusion_name = "fusion_{}_to_{}".format(stride_i, stride_i)
                else:
                    next_stride = self.take_strides[i + 1]
                    fusion_name = "fusion_{}_to_{}".format(
                        next_stride, stride_i
                    )
                setattr(
                    self,
                    fusion_name,
                    FusionResidualFlowBlock(
                        channels=stride2channels[stride_i],
                        upsampling=False if is_last_stride else True,
                        dequant_out=True
                        if stride_i == self.out_stride
                        else False,
                        factor=factor,
                        group_base=group_base,
                        bn_kwargs=bn_kwargs,
                    ),
                )
        self.pose_convs = nn.Sequential(
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

        self.rot_conv = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(
                3,
                3,
                kernel_size=(
                    input_shape[0] // self.take_strides[-1],
                    input_shape[1] // self.take_strides[-1],
                ),
            ),
        )
        self.rot_dequant = DeQuantStub()

        self.trans_conv1 = nn.Conv2d(
            256,
            3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

        self.trans_conv2 = nn.Sequential(
            nn.Conv2d(
                3,
                3,
                kernel_size=(
                    input_shape[0] // self.take_strides[-1],
                    input_shape[1] // self.take_strides[-1],
                ),
            )
        )
        self.trans_dequant = DeQuantStub()

    def forward(self, data: Mapping):
        res = OrderedDict()
        if self.training:
            assert len(data[self.feature_name]) == 2, (
                "pose_resflow head require two input," "check transform"
            )
            res.update(
                OrderedDict(axisangle=[], translation=[], residual_flow=[])
            )
            for features in data[self.feature_name]:
                (
                    rotation,
                    translation2,
                    residual_flow_out,
                ) = self.forward_pose_resflow(features)
                res["axisangle"].append(rotation)
                res["translation"].append(translation2)
                res["residual_flow"].append(residual_flow_out)
            return res
        else:
            (
                rotation,
                translation2,
                residual_flow_out,
            ) = self.forward_pose_resflow(data[self.feature_name][0])
            res.update(
                OrderedDict(
                    axisangle=rotation,
                    translation=translation2,
                )
            )
            if self.has_resflow:
                res["residual_flow"] = residual_flow_out
            return res

    def forward_pose_resflow(self, features: Sequence):
        features = _take_features(features, self.in_strides, self.take_strides)

        pose_fea = self.pose_convs(features[-1])

        rotation = self.rot_conv(pose_fea)
        rotation = self.rot_dequant(rotation)

        translation1 = self.trans_conv1(pose_fea)
        translation2 = self.trans_conv2(translation1)
        translation2 = self.trans_dequant(translation2)

        residual_flow_out = None
        if self.has_resflow:
            residual_flow_out = translation1
            for i, stride_i in enumerate(self.take_strides[::-1]):
                if i == 0:
                    next_stride = stride_i
                fusion_name = "fusion_{}_to_{}".format(next_stride, stride_i)
                residual_flow_out = getattr(self, fusion_name)(
                    residual_flow_out,
                    features[self.take_strides.index(stride_i)],
                )
                next_stride = stride_i
        return rotation, translation2, residual_flow_out

    def set_qconfig(self):

        # disable output quantization for last quanti layer.
        self.rot_conv[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        self.trans_conv2[
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
        for m in self.pose_convs:
            if hasattr(m, "fuse_model"):
                m.fuse_model()
        if self.has_resflow:
            for i, stride_i in enumerate(self.take_strides[::-1]):
                if i == 0:
                    next_stride = stride_i
                fusion_name = "fusion_{}_to_{}".format(next_stride, stride_i)
                next_stride = stride_i
                m = getattr(self, fusion_name)
                if hasattr(m, "fuse_model"):
                    m.fuse_model()


class OutputBlock(nn.Module):
    """
    Build OutputBlock for any pixle level ouput.

    Args:
        in_channels (int): in_channels for feature map.
        out_channels (int): out_channels for feature map.
        group_base (int): group_base of output_block, if > 1, use vargnet
            block.
        out_nums (int): output dim of result.
            For depth, always setting to 1 or 2.
            for optical flow, always setting to 2.
            for segmentation, always setting to num of classes.
        quanti_last_conv (bool): whether output quantize result in last conv.
        dequant_out (bool): whether dequant the output.
        dropout_ratio (float): dropout  ratio, should be in 0 ~ 1.
        bn_kwargs (dict): kwargs of bn layer.
        use_bias (bool): whether to use bias.
        factor (int): factor of Group Separable Conv.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: Mapping,
        quanti_last_conv: bool = False,
        dequant_out: bool = True,
        dropout_ratio: float = 0.0,
        use_bias: bool = False,
        out_nums: int = 1,
        factor: int = 2,
        group_base: int = 1,
    ):
        super(OutputBlock, self).__init__()
        self.quanti_last_conv = quanti_last_conv
        self.dequant_out = dequant_out
        if group_base > 1:
            assert group_base in [4, 8], "group_base should be 4 or 8"
            self.output_block = BasicVarGBlock(
                in_channels=in_channels,
                mid_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                bn_kwargs=bn_kwargs,
                factor=factor,
                group_base=group_base,
                merge_branch=False,
                dw_with_relu=True,
                pw_with_relu=False,
            )

        else:
            self.output_block = SeparableConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                pw_act_layer=nn.ReLU(inplace=True),
            )

        self.result_block = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_nums,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.dequant = DeQuantStub()

    def forward(self, data: Mapping):
        feat = self.output_block(data)
        if self.dropout is not None:
            feat = self.dropout(feat)
        result = self.result_block(feat)
        return self.dequant(result) if self.dequant_out else result

    def set_qconfig(self):

        if not self.quanti_last_conv:
            self.result_block.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )

    def fuse_model(self):
        self.output_block.fuse_model()


@OBJECT_REGISTRY.register
class PixelHead(nn.Module):
    """
    Build any pixle level output head.

    e.g. depth,otpical flow and so on.

    Args:
        in_strides (list): a list contains the strides of feature maps
            from backbone or neck.
        out_strides (list): a list contains the strides of output.
        stride2channels (dict): a stride to channel dict.
        quanti_last_conv (bool): whether output quantize result in last conv.
        dequant_out (bool): whether dequanti output.
        forward_frame_idxs (list of int): when input multi frame features,
            select which frame idxs to forward.
            e.g.
            for depth_pose training, backbone will output features of
            frame t and t+1, and depth head only output depth for t,
            so we should set forward_frame_idxs to [0].
            for bev training, backbone will output features of frame t and t+1,
            and bev_seg head will output result of frame t and t+1,
            so we should set forward_frame_idxs to [0,1].
        out_nums (int): output dim of result.
            For depth,always setting to 1 or 2.
            for optical flow, always setting to 2.
            for segmentation, always setting to num of classes.
        output_name (str): Name of result in result dict.
        feature_name (str): Name of features from backbone(or neck)
            in input dict.
        bn_kwargs (dict): kwargs of bn layer.
        use_bias (bool): whether to use bias.
        factor (int): factor of Group Separable Conv.
        group_base (int): group_base of output_block, if > 1, use vargnet
            block.
    """

    def __init__(
        self,
        in_strides: int,
        out_strides: int,
        stride2channels: Mapping,
        bn_kwargs: Mapping,
        forward_frame_idxs: Union[Sequence, int],
        quanti_last_conv: bool = False,
        dequant_out: bool = True,
        dropout_ratio: float = 0.0,
        out_nums: Union[Sequence[int], int] = 1,
        output_name: Union[Sequence[str], str] = "pred_depths",
        feature_name: str = "feats",
        factor: int = 2,
        use_bias: bool = False,
        group_base: int = 1,
        **kwargs,
    ):
        super(PixelHead, self).__init__(**kwargs)

        assert _is_increasing_sequence(in_strides), in_strides
        self.in_strides = _as_list(in_strides)
        self.forward_frame_idxs = _as_list(forward_frame_idxs)

        assert _is_increasing_sequence(out_strides), out_strides
        self.out_strides = _as_list(out_strides)
        assert out_strides[0] >= self.in_strides[0], "%d vs. %d" % (
            out_strides[0],
            self.in_strides[0],
        )
        assert out_strides[-1] <= self.in_strides[-1], "%d vs. %d" % (
            out_strides[-1],
            self.in_strides[-1],
        )

        out_nums = _as_list(out_nums)
        output_name = _as_list(output_name)

        self.output_blocks = nn.ModuleList()

        self.output_name = output_name
        self.feature_name = feature_name

        for stride in self.out_strides:
            # TODO(yueyu.wang): Compatible with old checkpoint, will refacotor
            # in next version.
            if len(self.output_name) == 1:
                self.output_blocks.append(
                    OutputBlock(
                        in_channels=stride2channels[stride],
                        out_channels=stride2channels[stride],
                        dropout_ratio=dropout_ratio,
                        bn_kwargs=bn_kwargs,
                        use_bias=use_bias,
                        out_nums=out_nums[0],
                        factor=factor,
                        group_base=group_base,
                        quanti_last_conv=quanti_last_conv,
                        dequant_out=dequant_out,
                    )
                )
            else:
                sub_output_blocks = nn.ModuleList()
                for out_nums_i in out_nums:
                    sub_output_blocks.append(
                        OutputBlock(
                            in_channels=stride2channels[stride],
                            out_channels=stride2channels[stride],
                            dropout_ratio=dropout_ratio,
                            bn_kwargs=bn_kwargs,
                            use_bias=use_bias,
                            out_nums=out_nums_i,
                            factor=factor,
                            group_base=group_base,
                            quanti_last_conv=quanti_last_conv,
                            dequant_out=dequant_out,
                        )
                    )
                self.output_blocks.append(sub_output_blocks)

    def forward(self, data: Mapping):
        res = OrderedDict()
        for frame_idx, features in enumerate(data[self.feature_name]):
            if frame_idx not in self.forward_frame_idxs:
                continue
            features = _take_features(
                features, self.in_strides, self.out_strides
            )
            assert len(features) == len(self.output_blocks), "%d vs. %d" % (
                len(features),
                len(self.output_blocks),
            )

            if len(self.output_name) == 1:
                result = []
                for feat, block in zip(features, self.output_blocks):
                    out = block(feat)
                    result.append(out)
                    if not self.training:
                        break
                res[
                    "%s_frame%s" % (self.output_name[0], str(frame_idx))
                ] = result
            else:
                assert len(self.output_name) > 1
                result = {output_name: [] for output_name in self.output_name}
                for feat, blocks in zip(features, self.output_blocks):
                    for output_name, sub_blocks in zip(
                        self.output_name, blocks
                    ):
                        out = sub_blocks(feat)
                        result[output_name].append(out)
                    if not self.training:
                        break
                for output_name in self.output_name:
                    res["%s_frame%s" % (output_name, str(frame_idx))] = result[
                        output_name
                    ]
        return res

    def fuse_model(self):
        for m in self.output_blocks:
            if len(self.output_name) == 1:
                m.fuse_model()
            else:
                for m_i in m:
                    m_i.fuse_model()

    def set_qconfig(self):
        int8_output = True
        for m in self.output_blocks:
            if len(self.output_name) == 1:
                m.set_qconfig()
            else:
                for m_i in m:
                    if int8_output:
                        m_i.set_qconfig()
                    else:
                        m_i.qconfig = (
                            qconfig_manager.get_default_qat_out_qconfig()
                        )  # noqa


@OBJECT_REGISTRY.register
class DepthPoseResflowHead(nn.Module):
    """
    Build DepthPoseResflowHead.

    Args:
        depth_head (dict): config of depth head to build.
        pose_resflow_head (dict): config of pose_resflow head to build.
        feature_name (str): Name of features from backbone(or neck)
            in input dict.

    """

    def __init__(
        self,
        depth_head: Mapping,
        pose_resflow_head: Mapping,
        feature_name: str,
        neck_out_stride2channels: Mapping,
        head_in_stride2channels: Mapping,
        **kwargs,
    ):
        super(DepthPoseResflowHead, self).__init__(**kwargs)
        self.feature_name = feature_name
        self.channel_align_layers = nn.ModuleList()
        self.neck_out_stride2channels = neck_out_stride2channels
        self.head_in_stride2channels = head_in_stride2channels
        strides = sorted(self.neck_out_stride2channels.keys())
        self.do_align = []
        for stride in strides:
            channel = self.neck_out_stride2channels[stride]
            if not channel == self.head_in_stride2channels[stride]:
                self.channel_align_layers.append(
                    ConvModule2d(
                        channel,
                        self.head_in_stride2channels[stride],
                        1,
                        bias=True,
                        norm_layer=nn.BatchNorm2d(
                            self.head_in_stride2channels[stride]
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                self.do_align.append(True)
            else:
                self.do_align.append(False)
        self.depth_head = depth_head
        self.pose_resflow_head = pose_resflow_head

    def forward(self, data: Mapping):
        align_data = {
            self.feature_name: [
                [] for _ in range(len(data[self.feature_name]))
            ]
        }
        k = 0
        for i in range(len(data[self.feature_name][0])):
            for j, feat in enumerate(data[self.feature_name]):
                if self.do_align[i]:
                    align_feat = self.channel_align_layers[k](feat[i])
                else:
                    align_feat = feat[i]
                align_data[self.feature_name][j].append(align_feat)
            if self.do_align[i]:
                k += 1

        res = self.depth_head(align_data)

        if self.pose_resflow_head is not None:
            res.update(self.pose_resflow_head(align_data))
        return res

    def fuse_model(self):
        self.depth_head.fuse_model()
        self.pose_resflow_head.fuse_model()

    def set_qconfig(self):
        for m in self.channel_align_layers:
            m.fuse_model()
        self.depth_head.set_qconfig()
        self.pose_resflow_head.set_qconfig()
