# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.core.box_utils import bbox_overlaps
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class RoIRandomSampler(nn.Module):
    """Random sample positive and negative roi.

    Args:
        num: number of proposals after sampled.
        pos_fraction: fraction of positive proposals. pos_num in the sampler
            result will be ``min(num * pos_fraction, all_pos_roi_num)``.
        neg_pos_ub: Upper bound ratio of neg/pos, maximum neg_pos_up * pos_num
            limit in the neg sampler. If neg_pos_up >= 0, neg_num
            in the sampler result will be
            ``min(num - pos_num, neg_pos_up * pos_num, all_neg_roi_num)``,
            otherwise will be ``min(num - pos_num, all_neg_roi_num)``.
            Default to -1.
        num_fg_classes: Number of foreground classes,
            this value now only used to generate ignore flag.
            Default to 1.
    """

    def __init__(
        self,
        num: int,
        pos_fraction: float,
        neg_pos_ub: float = -1,
        num_fg_classes: int = 1,
    ):
        super(RoIRandomSampler, self).__init__()

        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.num_fg_classes = num_fg_classes

    def random_choice(self, gallery: torch.Tensor, num: int) -> torch.Tensor:
        """Random select some elements from the gallery.

        `gallery` is a Tensor, the returned indices also is a Tensor.

        Args:
            gallery: indices pool.
            num: expected sample num.

        Returns:
            sampled indices.
        """
        assert len(gallery) >= num
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        return rand_inds

    def _sample_pos(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(
            (match_pos_flag > 0) & (ig_flag == 0), as_tuple=False
        )
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Randomly sample some negative samples.."""
        neg_inds = torch.nonzero(
            (match_pos_flag == 0) & (ig_flag == 0), as_tuple=False
        )
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if neg_inds.numel() <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def sample(self, match_pos_flag, match_gt_id, ig_flag, overlaps):
        if overlaps.shape[1] == 0:
            max_overlaps = np.zeros(overlaps.shape[0])
        else:
            max_overlaps = overlaps.max(dim=1)[0].detach().cpu().numpy()
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos(
            match_pos_flag,
            match_gt_id,
            ig_flag,
            max_overlaps,
            num_expected_pos,
        )
        # We find that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self._sample_neg(
            match_pos_flag,
            match_gt_id,
            ig_flag,
            max_overlaps,
            num_expected_neg,
        )
        neg_inds = neg_inds.unique()
        num_sampled_neg = neg_inds.numel()

        num_expected_ig = self.num - num_sampled_pos - num_sampled_neg
        if num_expected_ig > 0:
            all_set = set(np.where(max_overlaps >= 0)[0])
            ig_inds = torch.tensor(
                list(
                    all_set
                    - set(pos_inds.cpu().numpy())
                    - set(neg_inds.cpu().numpy())
                ),
                dtype=pos_inds.dtype,
                device=pos_inds.device,
            )
            ig_inds = self.random_choice(ig_inds, num_expected_ig)
        else:
            ig_inds = None

        return torch.cat([pos_inds, neg_inds]), ig_inds

    @autocast(enabled=False)
    def forward(
        self,
        boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        ig_flag: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_boxes_num: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # noqa: D205,D400,E501
        """
        Args:
            boxes: Box tensor with shape (B, N, 4). B stands for
                batch size, N the number of boxes for each sample.
            match_pos_flag: flag tensor with shape (B, N). Entries with value
                1 represents positive in matching, 0 for neg and -1
                for ignore.
            matched_gt_id: matched_gt_id tensor in (B, N).
                The best matched gt box id. -1 means unavailable.
            ig_flag: Flag tensor with shape (B, N, self._num_classes - 1) when
                self._exclude_background is True, or otherwise
                (B, N, self._num_classes). The range of the output is {0, 1}.
                Entries with value 1 are matched with ignore regions.
            gt_boxes: GT box tensor with shape (B, M1, 5+), In one sample,
                if the number of gt boxes is less than M1, the first M1
                entries should be filled with real data, and others
                padded with arbitrary values.
            gt_box_num: If provided, it is the gt box num tensor with shape
                (B,), the actual number of  gt boxes of each sample. Cannot
                be greater than M1.

        Returns:
            (boxes, match_pos_flag, match_gt_id, ig_flag): Tuple of sub-sampled
                tensors.
        """
        boxes_ret = []
        match_pos_flag_ret = []
        match_gt_id_ret = []
        ig_flag_ret = []

        assert (
            isinstance(boxes, torch.Tensor) and len(boxes.shape) == 3
        ), "boxes must be torch tensor and shape must be (B, N, 4)"

        if ig_flag is None:
            ig_flag = torch.zeros(
                (boxes.shape[0], boxes.shape[1], self.num_fg_classes),
                dtype=torch.bool,
                device=boxes.device,
            )
            ignore_match_inds = match_pos_flag == -1
            ig_flag[ignore_match_inds, :] = 1

        # set invalid bbox to ignore
        invalid_inds = torch.logical_or(
            (boxes[:, :, 2] <= boxes[:, :, 0]),
            (boxes[:, :, 3] <= boxes[:, :, 1]),
        )
        ig_flag[invalid_inds] = 1
        match_pos_flag[invalid_inds] = -1
        match_gt_id[invalid_inds] = -1

        num_imgs = gt_boxes.shape[0]
        for i in range(num_imgs):
            boxes_i = boxes[i]
            if gt_boxes_num is not None:
                num_gt_i = int(gt_boxes_num[i])
                gt_boxes_i = gt_boxes[i, 0:num_gt_i]

            match_pos_flag_i = match_pos_flag[i]
            match_gt_id_i = match_gt_id[i]
            ig_flag_i = ig_flag[i]
            overlaps_i = bbox_overlaps(boxes_i, gt_boxes_i[:, 0:4])
            sample_inds, ig_inds = self.sample(
                match_pos_flag_i,
                match_gt_id_i,
                ig_flag_i.sum(dim=1).to(torch.bool),
                overlaps_i,
            )
            if ig_inds is not None:
                sample_inds = torch.cat([sample_inds, ig_inds])
                ig_flag_i[ig_inds] = 1
                # NOTE: don't forget set ig_inds in the match_pos_flag to -1,
                # otherwise, the ignore state in the label encoder
                # will lose efficacy.
                match_pos_flag_i[ig_inds] = -1
                match_gt_id_i[ig_inds] = -1

            boxes_ret.append(boxes_i[sample_inds])
            match_pos_flag_ret.append(match_pos_flag_i[sample_inds])
            match_gt_id_ret.append(match_gt_id_i[sample_inds])
            ig_flag_ret.append(ig_flag_i[sample_inds])

        return (
            torch.stack(boxes_ret, dim=0),
            torch.stack(match_pos_flag_ret, dim=0),
            torch.stack(match_gt_id_ret, dim=0),
            torch.stack(ig_flag_ret, dim=0),
        )

    def set_qconfig(self):
        self.qconfig = None


@OBJECT_REGISTRY.register
class RoIHardProposalSampler(RoIRandomSampler):
    """Sampler hard positive & negative rois on the proposals.

    Sampling hard positive rois on the bottom of all positive rois, and
    negative rois on the top of all negative rois, all rois are the out of rpn
    postprocess (already sorted by the NMS). `bottom_pos_fraction` of needed
    positive RoIs are sampled on the bottom. `top_neg_fraction` of needed
    negative RoIs are sampled on the top. The others are sampled randomly.

    Args:
        num: Same as RoIRandomSampler.
        pos_fraction: Same as RoIRandomSampler.
        neg_pos_up: Same as RoIRandomSampler.
        bottom_pos_fraction: Sampling positive fraction of proposals on the
            bottom of proposals.
        top_neg_fraction: Sampling negative fraction of proposals on the top of
            proposals.
        num_fg_classes: Same as RoIRandomSampler.
    """

    def __init__(
        self,
        num: int,
        pos_fraction: float,
        neg_pos_ub: float = -1,
        bottom_pos_fraction: float = 0,
        top_neg_fraction: float = 0,
        num_fg_classes: int = 1,
    ):
        super(RoIHardProposalSampler, self).__init__(
            num,
            pos_fraction,
            neg_pos_ub,
            num_fg_classes,
        )

        assert 0 <= top_neg_fraction <= 1
        self.top_neg_fraction = top_neg_fraction

        assert 0 <= bottom_pos_fraction <= 1
        self.bottom_pos_fraction = bottom_pos_fraction

    def sample_on_top(
        self, gallery: torch.Tensor, num_expected: int
    ) -> torch.Tensor:
        """Sample on top.

        Args:
            gallery: indices of boxes gallery
            num_expected: Number of expected samples

        Returns:
            Indices of samples
        """
        if gallery.numel() <= num_expected:
            return gallery
        else:
            return gallery[0:num_expected]

    def sample_on_bottom(
        self, gallery: torch.Tensor, num_expected: int
    ) -> torch.Tensor:
        """Sample on bottom.

        Args:
            gallery: indices of boxes gallery
            num_expected: Number of expected samples

        Returns:
            Indices of samples
        """
        if gallery.numel() <= num_expected:
            return gallery
        else:
            return gallery[-num_expected:]

    def _sample_neg(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Sample negative boxes."""
        if self.top_neg_fraction <= 0:
            return super()._sample_neg(
                match_pos_flag,
                match_gt_id,
                ig_flag,
                max_overlaps,
                num_expected,
            )

        neg_inds = torch.nonzero(
            (match_pos_flag == 0) & (ig_flag == 0), as_tuple=False
        )
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)

        if neg_inds.numel() <= num_expected:
            return neg_inds
        else:
            # sampling on the negative top
            num_expected_top_sampling = max(
                1, int(num_expected * self.top_neg_fraction)
            )
            top_sampled_inds = self.sample_on_top(
                neg_inds,
                num_expected_top_sampling,
            )

            floor_neg_inds = neg_inds[top_sampled_inds.numel() :]
            num_expected_floor = num_expected - top_sampled_inds.numel()

            if floor_neg_inds.numel() > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor
                )
            else:
                sampled_floor_inds = floor_neg_inds
            sampled_inds = torch.cat([top_sampled_inds, sampled_floor_inds])
            return sampled_inds

    def _sample_pos(
        self, match_pos_flag, match_gt_id, ig_flag, max_overlaps, num_expected
    ):
        """Sample positive boxes."""
        if self.bottom_pos_fraction <= 0:
            return super()._sample_pos(
                match_pos_flag,
                match_gt_id,
                ig_flag,
                max_overlaps,
                num_expected,
            )

        pos_inds = torch.nonzero(
            (match_pos_flag > 0) & (ig_flag == 0), as_tuple=False
        )
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)

        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            # sampling on the negative top
            pos_set = set(pos_inds.cpu().numpy())

            num_expected_bottom_sampling = max(
                1, int(num_expected * self.bottom_pos_fraction)
            )

            # maybe gt boxes as proposal rois, so filter these
            hard_set = set(np.where(max_overlaps < 1)[0])
            pos_hard_inds = torch.tensor(
                list(pos_set & hard_set),
                dtype=pos_inds.dtype,
                device=pos_inds.device,
            )

            bottom_sampled_inds = self.sample_on_bottom(
                pos_hard_inds,
                num_expected_bottom_sampling,
            )

            ceil_pos_inds = torch.tensor(
                list(pos_set - set(bottom_sampled_inds.cpu().numpy())),
                dtype=pos_inds.dtype,
                device=pos_inds.device,
            )

            num_expected_ceil = num_expected - bottom_sampled_inds.numel()

            if len(ceil_pos_inds) > num_expected_ceil:
                sampled_ceil_inds = self.random_choice(
                    ceil_pos_inds, num_expected_ceil
                )
            else:
                sampled_ceil_inds = ceil_pos_inds
            sampled_inds = torch.cat([bottom_sampled_inds, sampled_ceil_inds])
            return sampled_inds
