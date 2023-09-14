from itertools import product
from math import sqrt
from numbers import Real
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils import fx_helper
from changan_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn,
)

from .functional import anchor_generator

@fx_helper.wrap
class AnchorGenerator(nn.Module):
    """Generate anchors with sliding window.

    The generated anchor map has the shape
    (batch_size, n_anchor * 4, height, width),
    and each anchor is encoded as (x1, y1, x2, y2)

    Args:
        feat_strides (list): strides of input features.
        scales (:obj:'list', optional): scales of base anchors in each feature
            level, should be used together with ratios
        ratios (:obj:'list', optional): aspect ratios of base anchors in each
            feature level, should be used together with scales
        base_sizes (:obj:'list', optional): The basic sizes of anchors in
            multiple levels. If None is given, strides will be used as
            base_sizes. (If strides are non square, the shortest stride is
            taken.)
        anchor_wh_groups (:obj:'list', optional): Each tuple represents the
            weight and height of a base anchor in corresponding feature level
        legacy_bbox (:obj:'bool', optional): Whether to subtract 1 from top
            and right coordinates of each box
        round_anchor (:obj:'bool', optional): Whether to round coordinates of
            each anchor to the nearest integer values
        feat_strides_bias (:obj:'list', optional): Center coordinates of base
            anchors in each feature level. If not specified,
            (stride - legacy_bbox) * 0.5 will be used.
    """

    def __init__(
        self,
        feat_strides: List[int],
        scales: Optional[Sequence[Real]] = None,
        ratios: Optional[Sequence[Real]] = None,
        base_sizes: Optional[Sequence[Real]] = None,
        anchor_wh_groups: Optional[List[List[Tuple[Real, Real]]]] = None,
        legacy_bbox: bool = False,
        round_anchor: bool = False,
        feat_strides_bias: Optional[List] = None,
        image_hw: Tuple[int, int] = (512, 960),
    ):

        super().__init__()

        assert isinstance(
            feat_strides, (list, tuple)
        ), "param feat_strides should be a list or tuple"
        assert isinstance(
            legacy_bbox, bool
        ), "param legacy_bbox must be a bool value"
        assert isinstance(
            round_anchor, bool
        ), "param round_anchor must be a bool value"

        self.feat_strides = feat_strides
        self.anchor_wh_groups = anchor_wh_groups
        self.legacy_bbox = legacy_bbox
        self.image_hw = image_hw

        if anchor_wh_groups is None:
            # generate anchor_wh_groups according to scales, ratios
            # and feature strides

            assert (
                scales is not None
            ), "param scales must be provided when anchor_wh_groups is None"
            assert isinstance(
                scales, (list, tuple)
            ), "param scales should be a list or tuple"
            assert len(scales) > 0, "param scales is empty"
            assert (
                ratios is not None
            ), "param scales must be provided when anchor_wh_groups is None"
            assert isinstance(
                ratios, (list, tuple)
            ), "param ratios should be a list or tuple"
            assert len(ratios) > 0, "param ratios is empty"
            if base_sizes is not None:
                assert isinstance(
                    base_sizes, (list, tuple)
                ), "param base_sizes should be a list or tuple"
                assert len(base_sizes) == len(
                    feat_strides
                ), "base_sizes must have same length as feat_strides"
            else:
                base_sizes = feat_strides
            anchor_wh_groups = self._get_anchor_wh_groups(
                scales, ratios, base_sizes
            )
        else:
            assert scales is None and ratios is None, (
                "scales and ratios must be None"
                + " when anchor_wh_groups is provided"
            )
            # anchor_wh_groups is preset, just check the validity
            assert isinstance(
                anchor_wh_groups, (list, tuple)
            ), "param anchor_wh_groups should be a list or tuple"
            assert len(anchor_wh_groups) == len(
                feat_strides
            ), "anchor_wh_groups must have same length as feat_strides"
            msg = (
                "param anchor_wh_groups should be "
                + "a List[List[Tuple[Real, Real]]]"
            )
            for anchor_wh_group in anchor_wh_groups:
                assert isinstance(anchor_wh_group, (list, tuple)), msg
                for v in anchor_wh_group:
                    assert (
                        isinstance(v, (list, tuple))
                        and len(v) == 2
                        and isinstance(v[0], Real)
                        and isinstance(v[1], Real)
                    ), msg

        # check the validity of feat_strides_bias if presented,
        # or fill it with default values instead
        if feat_strides_bias is not None:
            assert isinstance(
                feat_strides_bias, (list, tuple)
            ), "param feat_strides_bias should be a list or tuple"
            assert len(feat_strides) == len(
                feat_strides_bias
            ), "feat_strides_bias must have same length as feat_strides"
        else:
            feat_strides_bias = [None] * len(feat_strides)

        # generate multi-level base anchors, each shaped as (n_anchor, 4)
        for anchor_wh_group, stride, stride_bias in zip(
            anchor_wh_groups, feat_strides, feat_strides_bias
        ):
            self.register_buffer(
                f"base_anchors_{stride}",
                self._get_lvl_base_anchors(
                    anchor_wh_group,
                    stride,
                    legacy_bbox=legacy_bbox,
                    round_anchor=round_anchor,
                    feature_stride_bias=stride_bias,
                ),
                persistent=False,
            )

    def _get_anchor_wh_groups(self, scales, ratios, feat_strides):
        anchor_wh_groups = []
        for stride in feat_strides:
            anchor_group = []
            for ratio, scale in product(ratios, scales):
                lvl_scale = stride * scale
                ratio = sqrt(ratio)
                anchor_group.append((lvl_scale / ratio, lvl_scale * ratio))
            anchor_wh_groups.append(anchor_group)
        return anchor_wh_groups

    def _get_lvl_base_anchors(
        self,
        anchor_wh_group: List[Tuple[int]],
        feature_stride: Union[int, Tuple[int]],
        legacy_bbox: bool = False,
        round_anchor: bool = False,
        feature_stride_bias: Optional[Union[float, Tuple]] = None,
    ) -> torch.Tensor:
        """Get single level base anchors

        Args:
            anchor_wh_group (list): weight and height of anchors.
            feature_stride (int or tuple): stride of corresponding feature
                level, can be different in each axis
            legacy_bbox (:obj:'bool', optional): Whether to subtract 1 from
                top and right coordinates of each box
            round_anchor (:obj:'bool', optional): Whether to round coordinates
                of each anchor to the nearest integer values
            feature_stride_bias: Center coordinates of base anchors in current
                feature level. If not specified, (stride - legacy_bbox) * 0.5
                will be used.

        """

        def _ts(obj):
            if not isinstance(obj, (list, tuple)):
                obj = [obj]
            if len(obj) == 1:
                obj = [obj[0], obj[0]]
            assert len(obj) == 2
            return obj

        border = float(legacy_bbox)

        feature_stride = _ts(feature_stride)

        if feature_stride_bias is None:
            px = (feature_stride[0] - border) * 0.5
            py = (feature_stride[1] - border) * 0.5
        else:
            feature_stride_bias = _ts(feature_stride_bias)
            px = feature_stride_bias[0]
            py = feature_stride_bias[1]

        lvl_base_anchors = []
        for anchor_wh in anchor_wh_group:
            half_w = (anchor_wh[0] - border) * 0.5
            half_h = (anchor_wh[1] - border) * 0.5
            lvl_base_anchors.append(
                [px - half_w, py - half_h, px + half_w, py + half_h]
            )

        lvl_base_anchors = torch.Tensor(lvl_base_anchors)
        if round_anchor:
            lvl_base_anchors = torch.round(lvl_base_anchors)

        return lvl_base_anchors

    @script_quantized_fn
    # def forward(self, feature_maps: List[Union[torch.Tensor, QTensor]]):
    def forward(self, feature_maps: List[torch.Tensor]):
        """
        The forward pass of AnchorGenerator.

        Args:
            feature_maps (List[Tensor/QTensor]):
                Featuremaps in (n, c, h, w) format.

        Returns:
            List[Tensor/QTensor]: Anchors in (n, anchor_num * 4, h, w) format.
        """
        assert len(feature_maps) == len(
            self.feat_strides
        ), "input feature_map num mismatched with the length of feat_strides"

        mlvl_base_anchors = [
            getattr(self, f"base_anchors_{stride}")
            for stride in self.feat_strides
        ]

        feature_maps = [f.as_subclass(torch.Tensor) for f in feature_maps]

        mlvl_anchors = anchor_generator(
            mlvl_base_anchors, self.feat_strides, feature_maps
        )

        return mlvl_anchors
