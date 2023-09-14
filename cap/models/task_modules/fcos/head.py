# Copyright (c) Changan Auto. All rights reserved.

import changan_plugin_pytorch as changan
import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.base_modules import SeparableConvModule2d
from cap.models.weight_init import bias_init_with_prob, normal_init
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, multi_apply

__all__ = ["FCOSHead"]

INF = 1e8


@OBJECT_REGISTRY.register
class FCOSHead(nn.Module):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_strides (Sequence[int]): A list contains the strides of feature
            maps from backbone or neck.
        out_strides (Sequence[int]): A list contains the strides of this head
            will output.
        stride2channels (dict): A stride to channel dict.
        feat_channels (int): Number of hidden channels.
        stacked_convs (int): Number of stacking convs of the head.
        use_sigmoid (bool): Whether the classification output is obtained
            using sigmoid.
        share_bn (bool): Whether to share bn between multiple levels, default
            is share_bn.
        upscale_bbox_pred (bool): If true, upscale bbox pred by FPN strides.
        dequant_output (bool): Whether to dequant output. Default: True
        int8_output(bool): If True, output int8, otherwise output int32.
            Default: True
        share_conv(bool): Only the number of all stride channels is the same,
            share_conv can be True, branches share conv, otherwise not.
            Default: True
    """

    def __init__(
        self,
        num_classes,
        in_strides,
        out_strides,
        stride2channels,
        upscale_bbox_pred,
        feat_channels=256,
        stacked_convs=4,
        use_sigmoid=True,
        share_bn=False,
        dequant_output=True,
        int8_output=True,
        share_conv=True,
    ):
        super(FCOSHead, self).__init__()
        if upscale_bbox_pred:
            assert dequant_output, (
                "dequant_output should be True to convert "
                "QTensor to Tensor when upscale_bbox_pred is True"
            )
        self.num_classes = num_classes
        self.in_strides = sorted(_as_list(in_strides))
        self.out_strides = sorted(_as_list(out_strides))
        assert set(self.out_strides).issubset(
            self.in_strides
        ), "out_strides must be a subset of in_strides"
        self.feat_start_index = self.in_strides.index(min(self.out_strides))
        self.feat_end_index = self.in_strides.index(max(self.out_strides)) + 1
        self.stride2channels = stride2channels
        self.in_channels = (
            [stride2channels[stride] for stride in self.in_strides]
            if not share_conv
            else stride2channels[self.in_strides[0]]
        )
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid = use_sigmoid
        self.share_bn = share_bn
        self.upscale_bbox_pred = upscale_bbox_pred
        self.dequant_output = dequant_output
        self.int8_output = int8_output
        self.share_conv = share_conv
        assert self.share_bn is False if self.share_conv is False else True
        assert (
            len(set(stride2channels.values())) != 1 and not self.share_conv
        ) or len(set(stride2channels.values())) == 1
        self.background_label = num_classes
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.dequant = DeQuantStub()
        if self.share_conv:
            if self.share_bn:
                self._init_cls_convs()
                self._init_reg_convs()
            else:
                self._init_cls_reg_convs_with_independent_bn()
            self._init_predictor()
        else:
            self._init_cls_no_shared_convs()
            self._init_reg_no_shared_convs()
            self._init_no_shared_predictor()

    def _init_cls_no_shared_convs(self):
        self.cls_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            cls_convs = nn.ModuleList()
            for j in range(self.feat_start_index, self.feat_end_index):
                in_chn = self.in_channels[j]
                cls_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.cls_convs_list.append(cls_convs)

    def _init_reg_no_shared_convs(self):
        self.reg_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            reg_convs = nn.ModuleList()
            for j in range(self.feat_start_index, self.feat_end_index):
                in_chn = self.in_channels[j]
                reg_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.reg_convs_list.append(reg_convs)

    def _init_no_shared_predictor(self):
        self.conv_cls = nn.ModuleList()
        self.conv_reg = nn.ModuleList()
        self.conv_centerness = nn.ModuleList()
        # relu6 maybe affect performance
        self.single_relu = nn.ReLU(inplace=True)
        for j in range(self.feat_start_index, self.feat_end_index):
            in_chn = self.in_channels[j]
            self.conv_cls.append(
                nn.Conv2d(in_chn, self.cls_out_channels, 3, padding=1)
            )
            self.conv_reg.append(nn.Conv2d(in_chn, 4, 3, padding=1))
            self.conv_centerness.append(nn.Conv2d(in_chn, 1, 3, padding=1))

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_cls_reg_convs_with_independent_bn(self):
        """Initialize convs of cls head and reg head.

        depth-wise and point-wise convs are shared by all stride, but BN is
        independent, i.e. not shared, experiment shows that this will improve
        performance.
        """
        num_strides = len(self.out_strides)
        self.cls_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            for j in range(num_strides):
                if j == 0:
                    # to create new conv
                    cls_shared_dw_conv = None
                    cls_shared_pw_conv = None
                    reg_shared_dw_conv = None
                    reg_shared_pw_conv = None
                else:
                    # share convs of the first out stride, not create
                    cls_shared_dw_conv = self.cls_convs[0][i][0][0]
                    cls_shared_pw_conv = self.cls_convs[0][i][1][0]
                    reg_shared_dw_conv = self.reg_convs[0][i][0][0]
                    reg_shared_pw_conv = self.reg_convs[0][i][1][0]

                # construct cls_convs
                if cls_shared_dw_conv is None:
                    self.cls_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.cls_convs[j].append(
                        nn.Sequential(
                            cls_shared_dw_conv,
                            cls_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

                # construct reg_convs
                if reg_shared_dw_conv is None:
                    self.reg_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.reg_convs[j].append(
                        nn.Sequential(
                            reg_shared_dw_conv,
                            reg_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        # relu6 maybe affect performance
        self.single_relu = nn.ReLU(inplace=True)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def _init_weights(self):
        """Initialize weights of the head."""
        if self.share_conv:
            for m in self.cls_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.reg_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.conv_centerness.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self.conv_cls, std=0.01, bias=bias_cls)
            normal_init(self.conv_reg, std=0.01)
            normal_init(self.conv_centerness, std=0.01)
        else:
            for m in self.cls_convs_list.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.reg_convs_list.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            for m in self.conv_cls.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_cls)
            for m in self.conv_reg.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.conv_centerness.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward_single(self, x, i, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            i (int): Index of feature level.
            stride (int): The corresponding stride for feature maps, only
                used to upscale bbox pred when self.upscale_bbox_pred
                is True.
        """
        cls_feat = x
        reg_feat = x
        if self.share_conv:
            if self.share_bn:
                for cls_layer in self.cls_convs:
                    cls_feat = cls_layer(cls_feat)
            else:
                for cls_layer in self.cls_convs[i]:
                    cls_feat = cls_layer(cls_feat)
            cls_score = self.conv_cls(cls_feat)

            if self.share_bn:
                for reg_layer in self.reg_convs:
                    reg_feat = reg_layer(reg_feat)
            else:
                for reg_layer in self.reg_convs[i]:
                    reg_feat = reg_layer(reg_feat)
            bbox_pred = self.conv_reg(reg_feat)
            bbox_pred = self.single_relu(bbox_pred)
            centerness = self.conv_centerness(reg_feat)
        else:
            for cls_layer in self.cls_convs_list:
                cls_feat = cls_layer[i](cls_feat)
            cls_score = self.conv_cls[i](cls_feat)

            for reg_layer in self.reg_convs_list:
                reg_feat = reg_layer[i](reg_feat)
            bbox_pred = self.conv_reg[i](reg_feat)
            bbox_pred = self.single_relu(bbox_pred)
            centerness = self.conv_centerness[i](reg_feat)

        if self.dequant_output:
            cls_score = self.dequant(cls_score)
            bbox_pred = self.dequant(bbox_pred)
            centerness = self.dequant(centerness)

        if self.upscale_bbox_pred and self.training is not True:
            # Only used in eval mode when upscale_bbox_pred = True.
            # Because the ele-mul operation is not supported currently,
            # this part will be conduct in filter after dequant
            assert not isinstance(bbox_pred[0], changan.qtensor.QTensor), (
                "QTensor not support multiply op, you can set "
                "dequant_output=True to convert QTensor to Tensor"
            )
            bbox_pred *= stride

        return cls_score, bbox_pred, centerness

    def forward(self, feats):
        feats = _as_list(
            _as_list(feats)[self.feat_start_index : self.feat_end_index]
        )  # noqa
        cls_scores, bbox_preds, centernesses = multi_apply(
            self.forward_single,
            feats,
            range(len(self.out_strides)),
            self.out_strides,
        )

        return cls_scores, bbox_preds, centernesses

    def fuse_model(self):
        def fuse_model_convs(convs):
            if self.share_conv:
                for modules in convs:
                    for m in modules:
                        if self.share_bn:
                            m.fuse_model()
                        else:
                            if isinstance(m, SeparableConvModule2d):
                                modules_to_fuse = [["1.0", "1.1", "1.2"]]
                            elif isinstance(m, nn.Sequential):
                                modules_to_fuse = [["1", "2", "3"]]
                            else:
                                raise NotImplementedError(
                                    f"Not support type{type(m)} to fuse"
                                )
                            changan.quantization.fuse_conv_shared_modules(
                                m, modules_to_fuse, inplace=True
                            )
            else:
                for modules in convs:
                    for m in modules:
                        for n in m:
                            n.fuse_model()

        if self.share_conv:
            fuse_model_convs(self.cls_convs)
            fuse_model_convs(self.reg_convs)
        else:
            fuse_model_convs(self.cls_convs_list)
            fuse_model_convs(self.reg_convs_list)

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self.conv_cls.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_reg.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_centerness.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
