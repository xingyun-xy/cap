from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn import Interpolate
from torch.quantization import DeQuantStub

from cap.models.backbones.mixvargenet import MixVarGENetConfig
from cap.models.base_modules import MixVarGEBlock
from cap.models.base_modules.basic_vargnet_module import OnePathResUnit
from cap.models.base_modules.conv_module import ConvModule2d
from cap.models.base_modules.extend_container import ExtSequential
from cap.registry import OBJECT_REGISTRY

__all__ = [
    "RCNNMixVarGEShareHead",
    "RCNNVarGNetShareHead",
    "RCNNVarGNetSplitHead",
    "RCNNVarGNetHead",
]


@OBJECT_REGISTRY.register
class RCNNMixVarGEShareHead(nn.Module):
    """RCNN share head composed of the stacked MixVarGEBlock units.

    RCNN share head is shared by tasks belonging to the same task group.

    Args:
        bias: Whether to use bias in MixVarGEBlock.
        bn_kwargs: Kwargs of BN layer.
        head_config: Setting list of the stacked MixVarGEBlocks.
        upscale: Whether to upsample the output feature map by the scale of 2
    """

    def __init__(
        self,
        bias: bool,
        bn_kwargs: dict,
        head_config: List[MixVarGENetConfig],
        upscale=False,
    ):
        super().__init__()

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
                    bias=bias,
                    fusion_channels=(),
                    downsample_num=config_i.extra_downsample_num,
                    output_downsample=False,
                    bn_kwargs=bn_kwargs,
                )
            )
        self.head = ExtSequential(layers)

        self.upsample = (
            Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            if upscale
            else None
        )

    def forward(self, x):
        res = self.head(x)
        if self.upsample is not None:
            res = self.upsample(res)
        return res

    def fuse_model(self):
        self.head.fuse_model()


@OBJECT_REGISTRY.register
class RCNNVarGNetShareHead(OnePathResUnit):
    """RCNN share head composed of a VarGNet unit.

    RCNN share head is shared by tasks belonging to the same task group.

    Args:
        bn_kwargs: Kwargs of BN layer.
        gc_num_filter: Num filters of dw conv.
        pw_num_filter: Num filters of pw conv.
        pw_num_filter2: Num filters of second pw conv.
        group_base: Group base of group conv.
        factor: Factor of group conv.
        stride: Stride of group conv.
        roi_out_channel: Num of output channels of roi_feat_extractor,
            same as Num filters of input.
        upscale: Whether to upsample the output feature map by the scale of 2
    """

    def __init__(
        self,
        bn_kwargs: Dict,
        gc_num_filter: int,
        pw_num_filter: int,
        pw_num_filter2: int,
        group_base: int,
        factor: float,
        stride: int,
        roi_out_channel: int,
        upscale=False,
    ):
        super().__init__(
            in_filter=roi_out_channel,
            dw_num_filter=gc_num_filter,
            group_base=group_base,
            pw_num_filter=pw_num_filter,
            pw_num_filter2=pw_num_filter2,
            bn_kwargs=bn_kwargs,
            stride=stride,
            is_dim_match=False,
            use_bias=True,
            pw_with_act=True,
            factor=factor,
        )

        self.upsample = (
            Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            if upscale
            else None
        )

    def forward(self, x):
        res = super().forward(x)
        if self.upsample is not None:
            res = self.upsample(res)
        return res


@OBJECT_REGISTRY.register
class RCNNVarGNetSplitHead(nn.Module):
    """RCNN split head is not shared within a task group.

    The output of this module consists of at least one part which
    is the cls pred. By setting with_box_reg == True, the output also
    contains the reg pred.

    Args:
        num_fg_classes: Number of classes should be predicted.
        with_box_reg: Determines whether the output includes
            the reg pred.
        in_channel: Number of channels of input feature maps.
        bn_kwargs: Kwargs of BN layer.
        with_background: Determines whether the cls pred contains
            the background.
        pw_num_filter2: Number of filters of second pw conv.
        upscale: Whether to upsample input feature maps by the scale of 2.
        class_agnostic_reg: Determines whether to use the
            class_agnostic way for box reg.
        reg_channel_base: Number of the kinds of reg pred.
            The default is 4.
        use_bin: Determines the value of stride in conv.
        with_tracking_feat: Output feature of each box as tracking feature.
        int8_output: Determines whether the dtype of output is int8.
    """

    def __init__(
        self,
        num_fg_classes: int,
        with_box_reg: bool,
        in_channel: int,
        bn_kwargs: Dict,
        with_background: bool,
        pw_num_filter2: int,
        upscale: bool = False,
        class_agnostic_reg: bool = False,
        reg_channel_base: int = 4,
        use_bin: bool = False,
        with_tracking_feat: bool = False,
        int8_output: bool = False,
    ):
        super().__init__()
        with_bg = 1 if with_background else 0
        self.with_box_reg = with_box_reg
        self.with_tracking_feat = with_tracking_feat
        self.int8_output = int8_output

        self.dequant = DeQuantStub()

        self.upsample = (
            Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            if upscale
            else None
        )

        # rcnn split fc
        self.conv1 = ConvModule2d(
            in_channels=in_channel,
            out_channels=pw_num_filter2,
            kernel_size=1 if use_bin else 4,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )

        self.box_score = ConvModule2d(
            in_channels=pw_num_filter2,
            out_channels=num_fg_classes + with_bg,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=None,
            act_layer=None,
        )

        if self.with_box_reg:
            num_reg_channels = (
                reg_channel_base
                if class_agnostic_reg
                else (num_fg_classes + with_bg) * reg_channel_base
            )

            self.box_reg = ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=num_reg_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=None,
                act_layer=None,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        if self.upsample is not None:
            x = self.upsample(x)

        x = self.conv1(x)
        rcnn_cls_pred = self.dequant(self.box_score(x))
        output = OrderedDict(rcnn_cls_pred=rcnn_cls_pred)
        if self.with_box_reg:
            rcnn_reg_pred = self.dequant(self.box_reg(x))
            output.update(rcnn_reg_pred=rcnn_reg_pred)
        if self.with_tracking_feat:
            output.update(rcnn_tracking_feat=x)
        # if torch.onnx.is_in_onnx_export():
        #     for key, value in output.items():
        #         # each sample proposal 100 objects
        #         prev_shape = output[key].shape
        #         # output[key] = torch.stack(torch.split(output[key], 100, 0), 0)
        #         print(f'++++++++++++++ key={key}, type={type(output[key])}, shape change: {prev_shape} -> {output[key].shape}')
        return output

    def fuse_model(self):
        self.conv1.fuse_model()
        # Don't need to fuse output ops since no norm and act
        # ops presented

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self.box_score.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            if self.with_box_reg:
                self.box_reg.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )


@OBJECT_REGISTRY.register
class RCNNVarGNetHead(nn.Module):
    """RCNN Head composed of a VargNet unit and Convs.

    The output of this module can be flexibly set by setting flags.

    Args:
        num_fg_classes: Number of classes should be predicted.
        bn_kwargs: Kwargs of BN layer.
        class_agnostic_reg: Determines whether to use the
            class_agnostic way for box reg.
        with_background: Determines whether the cls pred contains
            the background.
        roi_out_channel: Num of output channels of roi_feat_extractor,
            same as Num filters of input.
        dw_num_filter: Num filters of dw conv.
        pw_num_filter: Num filters of pw conv.
        pw_num_filter2: Num filters of second pw conv.
        group_base: Group base of group conv.
        factor: Factor of group conv.
        with_box_cls: Determines whether the output includes
            the cls pred.
        with_box_reg: Determines whether the output includes
            the reg pred.
        reg_channel_base: Number of the kinds of reg pred.
            The default is 4.
        stride_step1: Stride of group conv of step1.
        ksize_step2: Kernel size of conv of step2.
        with_tracking_feat: Output feature of each box as tracking feature.
        upscale: Determines whether upscale the features.
        int8_output: Determines whether the dtype of output is int8.
    """

    def __init__(
        self,
        num_fg_classes: int,
        bn_kwargs: dict,
        class_agnostic_reg: bool,
        with_background: bool,
        roi_out_channel: int,
        dw_num_filter: int,
        pw_num_filter: int,
        pw_num_filter2: int,
        group_base: int,
        factor: float,
        stop_gradient: bool = False,
        with_box_cls: bool = True,
        with_box_reg: bool = True,
        reg_channel_base: int = 4,
        stride_step1: int = 2,
        ksize_step2: int = 4,
        with_tracking_feat: bool = False,
        upscale: bool = False,
        int8_output: bool = False,
    ):
        super(RCNNVarGNetHead, self).__init__()

        self.stop_gradient = stop_gradient
        self.with_box_reg = with_box_reg
        self.with_box_cls = with_box_cls
        self.with_tracking_feat = with_tracking_feat
        self.upscale = upscale
        self.int8_output = int8_output

        self.dequant = DeQuantStub()

        self.rcnn_conv = ExtSequential([])
        self.rcnn_conv.add_module(
            "step1",
            OnePathResUnit(
                in_filter=roi_out_channel,
                dw_num_filter=dw_num_filter,
                group_base=group_base,
                pw_num_filter=pw_num_filter,
                pw_num_filter2=pw_num_filter2,
                stride=(stride_step1, stride_step1),
                is_dim_match=False,
                use_bias=True,
                bn_kwargs=bn_kwargs,
                pw_with_act=True,
                factor=factor,
            ),
        )

        self.rcnn_conv.add_module(
            "step2",
            ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=pw_num_filter2,
                kernel_size=(ksize_step2, ksize_step2),
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
        )

        # if upscale=True, upscale w*h feature map to 2w*2h
        if self.upscale:
            self.upscale_module = Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
        if self.with_box_cls:
            self.box_score = ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=num_fg_classes + with_background,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                norm_layer=None,
                act_layer=None,
            )

        if self.with_box_reg:
            num_reg_channels = (
                reg_channel_base
                if class_agnostic_reg
                else (num_fg_classes + with_background) * reg_channel_base
            )

            self.box_reg = ConvModule2d(
                in_channels=pw_num_filter2,
                out_channels=num_reg_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=None,
                act_layer=None,
            )

    def forward(self, x):
        if self.stop_gradient:
            x = x.detach()
        x = self.rcnn_conv(x)
        if self.upscale:
            x = self.upscale_module(x)
        output = OrderedDict()
        if self.with_box_cls:
            rcnn_cls_pred = self.dequant(self.box_score(x))
            output["rcnn_cls_pred"] = rcnn_cls_pred
        if self.with_box_reg:
            rcnn_reg_pred = self.dequant(self.box_reg(x))
            output["rcnn_reg_pred"] = rcnn_reg_pred
        if self.with_tracking_feat:
            output.update(rcnn_tracking_feat=self.dequant(x))
        return output

    def fuse_model(self):
        self.rcnn_conv.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            if self.with_box_cls:
                self.box_score.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
            if self.with_box_reg:
                self.box_reg.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
