# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn.functional import ig_region_match, max_iou_match

from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class MaxIoUMatcher(nn.Module):
    """Bounding box classification label matcher by max iou.

    Args:
        pos_iou: Boxes whose IOU larger than ``pos_iou_thresh``
            is regarded as positive samples for classification.
        neg_iou: Boxes whose IOU smaller than ``neg_iou_thresh``
            is regarded as negative samples for classification.
        allow_low_quality_match: Whether to allow low quality match.
            Default is True.
        low_quality_match_iou: The iou thresh for low quality match.
            Low quality match will happens if any ground truth box
            is not matched to any boxes. Default is 0.1.
        legacy_bbox: Whether to add 1 while computing box border.
            Default is False.
        overlap_type: Overlap type for the calculation of correspondence,
            can be either "ioa" or "iou". Default is "iou".
        clip_gt_before_matching: Whether to clip ground truth boxes to image
            shape before matching. Default is False.
    """

    def __init__(
        self,
        pos_iou: float,
        neg_iou: float,
        allow_low_quality_match: bool = True,
        low_quality_match_iou: float = 0.1,
        legacy_bbox: bool = False,
        overlap_type: str = "iou",
        clip_gt_before_matching: bool = False,
    ):
        super().__init__()
        self._pos_iou = pos_iou
        self._neg_iou = neg_iou
        self._allow_low_quality_match = allow_low_quality_match
        self._low_quality_match_iou = low_quality_match_iou
        self._legacy_bbox = legacy_bbox
        assert overlap_type in (
            "ioa",
            "iou",
        ), "overlap_type can only be either ioa or iou"
        self._overlap_type = overlap_type
        self._clip_gt_before_matching = clip_gt_before_matching

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_boxes_num: torch.Tensor,
        im_hw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D205,D400
        """
        Args:
            boxes: Box tensor with shape (B, N, 4) or (N, 4) when boxes
                are identical in the whole batch.
            gt_boxes: GT box tensor with shape (B, M, 5+). In one sample,
                if the number of gt boxes is less than M, the first M
                entries should be filled with real data, and others
                padded with arbitrary values.
            gt_box_num: GT box num tensor with shape (B). The actual
                number of gt boxes for each sample. Cannot be
                greater than M.
            im_hw: Image HW tensor with shape (B, 2), the height and width
                value of each input image.

        Returns:
            flag: flag tensor with shape (B, N). Entries with value
                1 represents positive in matching, 0 for neg and -1
                for ignore.
            matched_gt_id: matched_gt_id tensor in (B, anchor_num).
                The best matched gt box id. -1 means unavailable.
        """
        if self._clip_gt_before_matching:
            assert im_hw is not None
            gt_boxes = gt_boxes.clone()
            gt_boxes.clamp_min_(0)
            gt_boxes[..., 0].clamp_max_(im_hw[:, 1].view(-1, 1))
            gt_boxes[..., 1].clamp_max_(im_hw[:, 0].view(-1, 1))
            gt_boxes[..., 2].clamp_max_(im_hw[:, 1].view(-1, 1))
            gt_boxes[..., 3].clamp_max_(im_hw[:, 0].view(-1, 1))

        flag, matched_gt_id = max_iou_match(
            boxes,
            gt_boxes,
            gt_boxes_num,
            self._pos_iou,
            self._neg_iou,
            self._allow_low_quality_match,
            self._low_quality_match_iou,
            self._legacy_bbox,
            self._overlap_type,
        )
        return flag, matched_gt_id


@OBJECT_REGISTRY.register
class IgRegionMatcher(nn.Module):  # noqa: D205,D400
    """Ignore region matcher by max overlap (intersection over area of
    ignore region).

    Args:
        num_classes: Number of classes, including background class.
        ig_region_overlap: Boxes whose IoA with an ignore region greater
            than ``ig_region_overlap`` is regarded as ignored.
        legacy_bbox: Whether to add 1 while computing box border.
        exclude_background: Whether to clip off the label corresponding
            to background class (indexed as 0) in output flag.
    """

    def __init__(
        self,
        num_classes: int,
        ig_region_overlap: float,
        legacy_bbox: Optional[bool] = False,
        exclude_background: Optional[bool] = False,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._ig_region_overlap = ig_region_overlap
        self._legacy_bbox = legacy_bbox
        self._exclude_background = exclude_background

    def forward(
        self,
        boxes: torch.Tensor,
        ig_regions: torch.Tensor,
        ig_regions_num: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D205,D400
        """
        Args:
            boxes: Box tensor with shape (B, N, 4) or (N, 4) when boxes
                are identical in the hole batch.
            ig_regions: Ignore region tensor with shape (B, M, 5+). In one
                sample, if the number of ig regions is less than M, the
                first M entries should be filled with real data, and others
                padded with arbitrary values.
            ig_regions_num: Ignore region num tensor in shape (B). The actual
                number of ig regions for each sample. Cannot be greater than M.

        Returns:
            Flag tensor with shape (B, self._num_classes - 1) when
                self._exclude_background is True, or otherwise
                (B, self._num_classes). The range of the output is {0, 1}.
                Entries with value 1 are matched with ignore regions.
        """

        return ig_region_match(
            boxes,
            ig_regions,
            ig_regions_num,
            self._num_classes,
            self._ig_region_overlap,
            self._legacy_bbox,
            self._exclude_background,
        )
