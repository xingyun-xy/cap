# Copyright (c) Changan Auto. All rights reserved.
from torch import nn
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d
from cap.models.weight_init import bias_init_with_prob, normal_init
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, multi_apply

__all__ = ["RetinaNetHead"]


@OBJECT_REGISTRY.register
class RetinaNetHead(nn.Module):
    """An anchor-based head used in `RetinaNet <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks.  The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the
            background category.
        num_anchors (int): Number of anchors for each pixel.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels.
        stacked_convs (int): Number of convs before cls and reg.
        int8_output (bool): If True, output int8, otherwise output int32.
            Default: True
    """  # noqa

    def __init__(
        self,
        num_classes: int,
        num_anchors: int,
        in_channels: int,
        feat_channels: int,
        stacked_convs: int = 4,
        int8_output: bool = True,
    ):
        super(RetinaNetHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.stacked_convs = stacked_convs
        self.int8_output = int8_output

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                ConvModule2d(
                    chn,
                    feat_channels,
                    3,
                    1,
                    1,
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            self.reg_convs.append(
                ConvModule2d(
                    chn,
                    feat_channels,
                    3,
                    1,
                    1,
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        self.retina_cls = nn.Conv2d(
            feat_channels,
            num_anchors * num_classes,
            3,
            1,
            1,
        )
        self.retina_reg = nn.Conv2d(
            feat_channels,
            num_anchors * 4,
            3,
            1,
            1,
        )
        self.dequant = DeQuantStub()
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            if isinstance(m[0], nn.Conv2d):
                normal_init(m[0], std=0.01)
        for m in self.reg_convs:
            if isinstance(m[0], nn.Conv2d):
                normal_init(m[0], std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Feature of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        cls_score = self.dequant(cls_score)
        bbox_pred = self.dequant(bbox_pred)
        return cls_score, bbox_pred

    def forward(self, features):
        cls_scores = []
        bbox_preds = []
        features = _as_list(features)
        cls_scores, bbox_preds = multi_apply(self.forward_single, features)
        return cls_scores, bbox_preds

    def fuse_model(self):
        for module in self.cls_convs:
            module.fuse_model()
        for module in self.reg_convs:
            module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if not self.int8_output:
            self.retina_cls.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.retina_reg.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
