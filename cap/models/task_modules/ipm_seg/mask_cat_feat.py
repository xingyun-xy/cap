import logging
from typing import Dict, List

import changan_plugin_pytorch as changan
from changan_plugin_pytorch.nn import Interpolate
from torch import nn
from torch.quantization import DeQuantStub

from cap.models.base_modules import BasicVarGBlock
from cap.models.base_modules.conv_module import ConvModule2d
from cap.models.base_modules.separable_conv_module import (
    SeparableGroupConvModule2d,
)
from cap.models.weight_init import normal_init
from cap.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


def base_conv_module(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    bn_kwargs,
    group_base,
    conv_method,
):
    if conv_method == "varg_conv":
        return BasicVarGBlock(
            in_channels=in_channels,
            mid_channels=out_channels,
            out_channels=out_channels,
            stride=stride,
            bn_kwargs=bn_kwargs,
            kernel_size=kernel_size,
            padding=padding,
            factor=1,
            group_base=group_base,
        )
    elif conv_method == "sep_conv":
        return SeparableGroupConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            bias=True,
            dw_act_layer=nn.ReLU(inplace=True),
            pw_act_layer=nn.ReLU(inplace=True),
            dw_norm_layer=nn.BatchNorm2d(in_channels, **bn_kwargs),
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
        )
    else:
        return ConvModule2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )


@OBJECT_REGISTRY.register
class MaskcatFeatHead(nn.Module):
    """Head Module for ipm segmentation task.

    Args:
        num_classes: Number of classes.
        in_strides: The strides corresponding to the inputs of
            seg_head, the inputs usually come from backbone or neck.
        out_strides: List of output strides.
        in_channels: Number of input channels .
        out_channels:
            Number of hidden channels (of each output stride).
        stacked_convs: Number of stacking convs of head.  Default: 3
        start_level: Begining index of neck features input to head.
            Default: 0
        end_level: End index of neck features input to head. Default: 2
        group_base: Group param used in conv. Default: 8
        conv_method: Choice of convolution method used in head.
            Default: varg_conv
        share_conv: Whether to do parameter sharing in head.
            Default: False
        use_auxi_loss: Whether to use auxi loss in head. Default: False
        aggregation_method: Feature aggregation method. Default: "concat"
        has_project_layer: Whether add projection layer in head.
            Default: False
        argmax_output: Whether conduct argmax on output. Default: False
        dequant_output: Whether to dequant output. Default: True
        int8_output: If True, output int8, otherwise output int32.
            Default: True
        bn_kwargs: Extra keyword arguments for bn layers.
    """

    def __init__(
        self,
        num_classes: int,
        in_strides: List[int],
        out_strides: List[int],
        in_channels: int,
        out_channels: int,
        stacked_convs: int = 3,
        start_level: int = 0,
        end_level: int = 2,
        group_base: int = 8,
        conv_method: str = "varg_conv",
        share_conv: bool = False,
        use_auxi_loss: bool = False,
        aggregation_method: str = "concat",
        has_project_layer: bool = False,
        argmax_output: bool = False,
        dequant_output: bool = True,
        int8_output: bool = True,
        bn_kwargs: Dict = None,
    ):
        super(MaskcatFeatHead, self).__init__()

        assert conv_method in ["varg_conv", "conv2d", "sep_conv"]
        assert start_level >= 0 and start_level <= end_level
        assert in_strides[start_level] >= out_strides[0]
        assert aggregation_method in ["sum", "concat"]

        self.conv_method = conv_method
        self.group_base = group_base

        self.in_strides = in_strides
        self.out_strides = out_strides
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stacked_convs = stacked_convs
        self.start_level = start_level
        self.end_level = end_level

        self.num_classes = num_classes
        self.share_conv = share_conv
        self.use_auxi_loss = use_auxi_loss

        self.bn_kwargs = bn_kwargs

        self.aggregation_method = aggregation_method

        self.has_project_layer = has_project_layer

        self.int8_output = int8_output
        self.dequant_output = dequant_output
        self.argmax_output = argmax_output
        self.dequant = DeQuantStub()

        self._init_fpn_layers()
        self._init_seg_layers()
        self._init_cls_layers()

        # need to be improved
        self.scale_factor = in_strides[start_level] // out_strides[0]
        self.upsample = Interpolate(
            scale_factor=self.scale_factor,
            align_corners=False,
            recompute_scale_factor=True,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def _init_cls_layers(self):
        self.conv_cls_layers = nn.ModuleList()
        for _ in range(self.start_level, self.end_level + 1):
            self.conv_cls_layers.append(
                nn.Conv2d(self.out_channels, self.num_classes, 1)
            )
            if not self.use_auxi_loss or self.share_conv:
                break

    def _init_seg_layers(self):
        self.conv_seg_layers = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            for j in range(self.stacked_convs):
                in_channels = self.in_channels if j == 0 else self.out_channels
                one_conv = base_conv_module(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    bn_kwargs=self.bn_kwargs,
                    group_base=self.group_base,
                    conv_method=self.conv_method,
                )
                convs_per_level.add_module("conv" + str(i) + str(j), one_conv)
            self.conv_seg_layers.append(convs_per_level)
            if not self.use_auxi_loss or self.share_conv:
                break

    def _init_fpn_layers(self):
        self.upsample_layers = nn.ModuleList()
        strides = self.in_strides[self.start_level : self.end_level + 1]
        scale_factors = [
            strides[i] // strides[0] for i in range(1, len(strides))
        ]

        if self.aggregation_method == "concat":
            self.cat_layer = nn.quantized.FloatFunctional()
            in_channels = self.in_channels * len(strides)
            self.fusion_layer = ConvModule2d(
                in_channels,
                self.in_channels,  # self.out_channels
                1,
                stride=1,
                dilation=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(self.in_channels, **self.bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )
        else:
            self.level_add = nn.ModuleList()

        for i in range(len(scale_factors)):
            one_upsample = Interpolate(
                scale_factor=scale_factors[i],
                mode="bilinear",
                align_corners=None,
                recompute_scale_factor=True,
            )
            self.upsample_layers.append(one_upsample)

            if self.aggregation_method == "sum":
                self.level_add.append(nn.quantized.FloatFunctional())

        if self.has_project_layer:
            self.project_layers = nn.ModuleList()
            for _ in range(len(strides)):
                self.project_layers.append(
                    base_conv_module(
                        self.in_channels,  # self.out_channels
                        self.in_channels,  # self.out_channels
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        bn_kwargs=self.bn_kwargs,
                        group_base=self.group_base,
                        conv_method=self.conv_method,
                    )
                )

    def forward(self, inputs):
        assert isinstance(inputs, List)
        assert len(inputs) >= (self.end_level - self.start_level + 1)

        inputs = inputs[self.start_level : self.end_level + 1]
        # feature aggregation
        project_feats = []
        if self.has_project_layer:
            for i in range(len(inputs)):
                input_p = inputs[i]
                project_feat = self.project_layers[i](input_p)
                project_feats.append(project_feat)

        if len(project_feats) == 0:
            project_feats = inputs

        feature_all_level = []
        feature_all_level.append(project_feats[0])
        for i in range(1, len(project_feats)):
            input_p = project_feats[i]
            cur_feat = self.upsample_layers[i - 1](input_p)
            feature_all_level.append(cur_feat)
        if self.aggregation_method == "concat":
            feature_add_all_level = self.cat_layer.cap(
                feature_all_level, dim=1
            )
            feature_add_all_level = self.fusion_layer(feature_add_all_level)
        else:
            feature_add_all_level = feature_all_level[0]
            for i in range(1, len(feature_all_level)):
                feature_add_all_level = self.level_add[i - 1].add(
                    feature_add_all_level, feature_all_level[i]
                )

        # semantic conv
        seg_feats = []
        seg_feature = self.conv_seg_layers[0](feature_add_all_level)
        seg_feats.append(seg_feature)
        if self.use_auxi_loss:
            if self.share_conv:
                for i in range(1, len(inputs)):
                    input_p = inputs[i]
                    seg_feats.append(self.conv_seg_layers[0](input_p))
            else:
                for i in range(1, len(inputs)):
                    input_p = inputs[i]
                    seg_feats.append(self.conv_seg_layers[i](input_p))

        # seg_cls
        preds = []
        pred = self.conv_cls_layers[0](seg_feats[0])

        pred = self.upsample(pred)

        if self.argmax_output:
            pred = changan.argmax(pred, dim=1, keepdim=True)
            # pred = pred.argmax(dim=1, keepdim=True)
        elif self.dequant_output:
            pred = self.dequant(pred)

        preds.append(pred)
        if self.use_auxi_loss:
            if self.share_conv:
                for i in range(1, len(inputs)):
                    preds.append(
                        self.dequant(self.conv_cls_layers[0](seg_feats[i]))
                    )
            else:
                for i in range(1, len(inputs)):
                    preds.append(
                        self.dequant(self.conv_cls_layers[i](seg_feats[i]))
                    )

        return tuple(preds)

    def fuse_model(self):
        modules = [
            self.project_layers if self.has_project_layer else None,
            self.conv_cls_layers,
            self.conv_seg_layers,
        ]
        if self.aggregation_method == "concat":
            self.fusion_layer.fuse_model()
        for module in modules:
            if module is not None:
                for m in module:
                    if hasattr(m, "fuse_model"):
                        m.fuse_model()
                    elif isinstance(m, nn.Sequential):
                        for op in m:
                            if hasattr(op, "fuse_model"):
                                op.fuse_model()

    def set_qconfig(self):
        self.qconfig = changan.quantization.get_default_qat_qconfig()
        if not self.int8_output:
            self.conv_cls_layers[
                0
            ].qconfig = changan.quantization.get_default_qat_out_qconfig()
