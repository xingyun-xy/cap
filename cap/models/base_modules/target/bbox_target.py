# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from changan_plugin_pytorch.qtensor import QTensor
from torch.cuda.amp import autocast
from torch.quantization.stubs import DeQuantStub

from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class BBoxTargetGenerator(nn.Module):
    """BBox Target Generator for detection task.

    BBoxTargetGenerator wraps matchers, sampler and an encoder to
    generate training target by firstly matching predictions with
    ground truths to build correspondences and generating training
    target for each prediction.

    The detail of matching and label encoding are implemented
    in matcher classes.

    Args:
        matcher: Matcher defines how the matching between predictions
            and ground truths actually works.
        label_encoder: Label encoder defines how to generate training
            target for each prediction given ground truths and
            correspondences.
        ig_region_matcher: Ignore region matcher is used to generate
            ignore flags for each pred box according to its overlap
            with input ignore regions.
        sampler: Sampler defines how to do sample on the bbox and target.
            If provide, will do sample on the boxes according to the match
            state.
            Default to None.
    """

    def __init__(
        self,
        matcher: torch.nn.Module,
        label_encoder: torch.nn.Module,
        ig_region_matcher: Optional[torch.nn.Module] = None,
        sampler: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        self.matcher = matcher
        self.ig_region_matcher = ig_region_matcher
        self.label_encoder = label_encoder
        self.sampler = sampler
        self.dequant = DeQuantStub()

    @property
    def with_ig_region_matcher(self):
        return self.ig_region_matcher is not None

    def _dequant_boxes(self, boxes):
        """Dequant input boxes on demand."""
        if isinstance(boxes, QTensor):
            boxes = self.dequant(boxes)
        elif isinstance(boxes, list) and isinstance(boxes[0], QTensor):
            boxes = [self.dequant(b) for b in boxes]
        return boxes

    @staticmethod
    def _reorganize_boxes(mlvl_anchors):
        boxes = []
        for anchors in mlvl_anchors:
            bs = anchors.shape[0]
            boxes.append(anchors.permute(0, 2, 3, 1).reshape(bs, -1, 4))

        return torch.cat(boxes, dim=1)

    @staticmethod
    def _fill_with_max_len(tensor_lst: List[torch.Tensor]):  # noqa: D205,D400
        """
        Get the maximum length of each tensor, pad the rest
        to that length, and concat all tensors together.
        """
        target_tensor = tensor_lst[0]
        len_lst = [len(t) for t in tensor_lst]
        max_len = max(len_lst)
        tensors = target_tensor.new_zeros(
            (len(tensor_lst), max_len, *tensor_lst[0].shape[1:])
        )
        for i, (t, l) in enumerate(zip(tensor_lst, len_lst)):
            tensors[i, :l] = t

        return tensors, target_tensor.new_tensor(len_lst)

    def _match_and_sample(
        self,
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        gt_boxes: torch.Tensor,
        gt_boxes_num: torch.Tensor,
        ig_regions: Optional[torch.Tensor] = None,
        ig_regions_num: Optional[torch.Tensor] = None,
        im_hw: Optional[torch.Tensor] = None,
    ):
        """Match boxes to gt boxes, and optionally do sample accordingly."""
        assert boxes.ndim == 3, "boxes not in supported formats"
        if not (gt_boxes.dtype == gt_boxes_num.dtype):
            gt_boxes_num = gt_boxes_num.to(gt_boxes.dtype)

        if ig_regions is not None:
            if not (ig_regions_num.dtype == ig_regions.dtype):
                ig_regions_num = ig_regions_num.to(ig_regions.dtype)
            assert ig_regions.dtype == ig_regions_num.dtype == gt_boxes.dtype

        # Get matching between input boxes and gt boxes
        match_pos_flag, match_gt_id = self.matcher(
            boxes, gt_boxes, gt_boxes_num, im_hw=im_hw
        )

        # optionally do ignore region matching
        ig_flag = None
        if self.with_ig_region_matcher and ig_regions is not None:
            ig_flag = self.ig_region_matcher(boxes, ig_regions, ig_regions_num)

        if self.sampler is not None:
            boxes, match_pos_flag, match_gt_id, ig_flag = self.sampler(
                boxes,
                match_pos_flag,
                match_gt_id,
                ig_flag,
                gt_boxes,
                gt_boxes_num,
            )

        return boxes, match_pos_flag, match_gt_id, ig_flag

    @autocast(enabled=False)
    def forward(
        self,
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        gt_boxes: Union[torch.Tensor, List[torch.Tensor]],
        gt_boxes_num: Optional[torch.Tensor] = None,
        ig_regions: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        ig_regions_num: Optional[torch.Tensor] = None,
        im_hw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # noqa: D205,D400
        """
        Args:
            boxes: Box tensor with shape (B, N, 4) or a list of anchor
                tensors each with shape (B, N*4, H, W), where each tensor
                corresponds to anchors of one feature stride. B stands for
                batch size, N the number of boxes for each sample, H the
                height and W the width.
            gt_boxes: GT box tensor with shape (B, M1, 5+), or a list of B
                2d tensors with 5+ as the size of the last dim. For the former
                ase, in one sample, if the number of gt boxes is less than M1,
                the first M1 entries should be filled with real data, and
                others padded with arbitrary values.
            gt_box_num: If provided, it is the gt box num tensor with shape
                (B,), the actual number of  gt boxes of each sample. Cannot
                be greater than M1.
            ig_regions: Ignore region tensor with shape (B, M2, 5+), or a
                list of B 2d tensors with 5+ as the size of the last dim.
                For the former case, in one sample, if the number of ig
                regions is less than M2, the first M2 entries should be filled
                with real data, and others padded with arbitrary values.
            ig_regions_num: If provided, it is ignore region num tensor in
                shape (B,), the actual number of ig regions of each sample.
                Cannot be greater than M2.
        """

        # Handle QTensor case in QAT training.
        _boxes = self._dequant_boxes(boxes)

        # reorganize boxes on-demand
        if isinstance(_boxes, list):
            _boxes = self._reorganize_boxes(_boxes)

        if isinstance(gt_boxes, list):
            assert gt_boxes_num is None
            gt_boxes, gt_boxes_num = self._fill_with_max_len(gt_boxes)

        gt_boxes_num = gt_boxes_num.flatten()

        if isinstance(ig_regions, list):
            assert ig_regions_num is None
            ig_regions, ig_regions_num = self._fill_with_max_len(ig_regions)

        if ig_regions_num is not None:
            ig_regions_num = ig_regions_num.flatten()

        _boxes, match_pos_flag, match_gt_id, ig_flag = self._match_and_sample(
            _boxes,
            gt_boxes,
            gt_boxes_num,
            ig_regions=ig_regions,
            ig_regions_num=ig_regions_num,
            im_hw=im_hw,
        )

        kwargs = {"ig_flag": ig_flag} if ig_flag is not None else {}
        labels: Dict[str, torch.Tensor] = self.label_encoder(
            _boxes,
            gt_boxes,
            match_pos_flag,
            match_gt_id,
            **kwargs,
        )

        return _boxes, labels


@OBJECT_REGISTRY.register
class ProposalTarget(BBoxTargetGenerator):
    """Proposal Target Generator for two-stage task.

    ProposalTarget Generator wraps matchers, sampler and an encoder to
    generate training target by firstly matching predictions with
    ground truths to build correspondences and generating training
    target for each proposal. If sampler is given, the final proposal
    bbox would be sampled.

    Args:
        matcher: same as BBoxTargetGenerator.
        label_encoder: same as BBoxTargetGenerator.
        ig_region_matcher: same as BBoxTargetGenerator.
        add_gt_bbox_to_proposal: If add gt_bboxes to the pred boxes as
            positive proposal boxes.
            Default to False.
        sampler: same as BBoxTargetGenerator.
    """

    def __init__(
        self,
        matcher: torch.nn.Module,
        label_encoder: torch.nn.Module,
        ig_region_matcher: Optional[torch.nn.Module] = None,
        add_gt_bbox_to_proposal: bool = False,
        sampler: Optional[torch.nn.Module] = None,
    ):
        super(ProposalTarget, self).__init__(
            matcher, label_encoder, ig_region_matcher, sampler
        )
        self._add_gt_bbox_to_proposal = add_gt_bbox_to_proposal

    @staticmethod
    def _reorganize_boxes(batch_boxes: List[torch.Tensor]):
        return torch.stack(batch_boxes)

    def _match_and_sample(
        self,
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        gt_boxes: torch.Tensor,
        gt_boxes_num: torch.Tensor,
        ig_regions: Optional[torch.Tensor] = None,
        ig_regions_num: Optional[torch.Tensor] = None,
        im_hw: Optional[torch.Tensor] = None,
    ):
        """Match boxes to gt boxes, and optionally do sample accordingly."""
        if self._add_gt_bbox_to_proposal:
            boxes = torch.cat([boxes, gt_boxes[:, :, 0:4]], dim=1)

        (
            boxes,
            match_pos_flag,
            match_gt_id,
            ig_flag,
        ) = super()._match_and_sample(
            boxes, gt_boxes, gt_boxes_num, ig_regions, ig_regions_num, im_hw
        )

        return boxes, match_pos_flag, match_gt_id, ig_flag


@OBJECT_REGISTRY.register
class ProposalTargetBinDet(ProposalTarget):
    """The ProposalTarget used in bin detection."""

    @autocast(enabled=False)
    def forward(
        self,
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        gt_boxes: torch.Tensor,
        parent_gt_boxes: torch.Tensor,
        gt_boxes_num: torch.Tensor = None,
        parent_ig_regions: torch.Tensor = None,
        parent_gt_boxes_num: torch.Tensor = None,
        parent_ig_regions_num: torch.Tensor = None,
        im_hw: Optional[torch.Tensor] = None,
        ig_regions: torch.Tensor = None,
        ig_regions_num: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # noqa: D205,D400

        # Handle QTensor case in QAT training.
        _boxes = self._dequant_boxes(boxes)

        # reorganize boxes on-demand
        if isinstance(_boxes, list):
            _boxes = self._reorganize_boxes(_boxes)

        if isinstance(parent_gt_boxes, list):
            assert parent_gt_boxes_num is None
            parent_gt_boxes, parent_gt_boxes_num = self._fill_with_max_len(
                parent_gt_boxes
            )
        if parent_gt_boxes_num is not None:
            parent_gt_boxes_num = parent_gt_boxes_num.flatten()

        if isinstance(gt_boxes, list):
            assert gt_boxes_num is None
            gt_boxes, gt_boxes_num = self._fill_with_max_len(gt_boxes)

        if gt_boxes_num is not None:
            gt_boxes_num = gt_boxes_num.flatten()

        if isinstance(parent_ig_regions, list):
            assert parent_ig_regions_num is None
            parent_ig_regions, parent_ig_regions_num = self._fill_with_max_len(
                parent_ig_regions
            )

        if parent_ig_regions_num is not None:
            parent_ig_regions_num = parent_ig_regions_num.flatten()

        if isinstance(ig_regions, list):
            assert ig_regions_num is None
            ig_regions, ig_regions_num = self._fill_with_max_len(ig_regions)

        if ig_regions_num is not None:
            ig_regions_num = ig_regions_num.flatten()

        _boxes, match_pos_flag, match_gt_id, _ = self._match_and_sample(
            _boxes,
            parent_gt_boxes,
            parent_gt_boxes_num,
            ig_regions=parent_ig_regions,
            ig_regions_num=parent_ig_regions_num,
            im_hw=im_hw,
        )

        if ig_regions is None:
            labels: Dict[str, torch.Tensor] = self.label_encoder(
                _boxes,
                gt_boxes,
                match_pos_flag,
                match_gt_id,
            )
        else:
            labels: Dict[str, torch.Tensor] = self.label_encoder(
                _boxes,
                gt_boxes,
                match_pos_flag,
                match_gt_id,
                ig_regions,
                ig_regions_num,
            )

        return boxes, labels


@OBJECT_REGISTRY.register
class ProposalTargetGroundLine(ProposalTarget):
    @autocast(enabled=False)
    def forward(
        self,
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        gt_boxes: torch.Tensor,
        gt_flanks: torch.Tensor,
        gt_boxes_num: torch.Tensor = None,
        gt_flanks_num: torch.Tensor = None,
        im_hw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # noqa: D205,D400
        """
        Args:
            gt_flanks: GT flanks tensor with shape (B, M1, 9), or a list of B
                2d tensors with 9 as the size of the last dim.
        """
        # Handle QTensor case in QAT training.
        _boxes = self._dequant_boxes(boxes)

        # reorganize boxes on-demand
        if isinstance(_boxes, list):
            _boxes = self._reorganize_boxes(_boxes)

        if isinstance(gt_boxes, list):
            assert gt_boxes_num is None
            gt_boxes, gt_boxes_num = self._fill_with_max_len(gt_boxes)

        if isinstance(gt_flanks, list):
            assert gt_flanks_num is None
            gt_flanks, gt_flanks_num = self._fill_with_max_len(gt_flanks)

        gt_boxes_num = gt_boxes_num.flatten()
        gt_flanks_num = gt_flanks_num.flatten()

        _boxes, match_pos_flag, match_gt_id, _ = self._match_and_sample(
            _boxes,
            gt_boxes,
            gt_boxes_num,
            im_hw=im_hw,
        )

        labels: Dict[str, torch.Tensor] = self.label_encoder(
            _boxes,
            gt_boxes,
            gt_flanks,
            match_pos_flag,
            match_gt_id,
        )

        return boxes, labels


@OBJECT_REGISTRY.register
class ProposalTarget3D(ProposalTarget):
    """The ProposalTarget used in 3d detection."""

    @autocast(enabled=False)
    def forward(
        self,
        *args,
        trans_mat,
        calib,
        distCoeffs,
        eq_fu=None,
        eq_fv=None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        rois, labels = super().forward(*args, **kwargs)
        labels["trans_mat"] = trans_mat
        labels["calib"] = calib
        labels["distCoeffs"] = distCoeffs
        labels["eq_fu"] = eq_fu
        labels["eq_fv"] = eq_fv

        return rois, labels
