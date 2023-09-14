# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from cap.core.box_utils import box_corner_to_center, zoom_boxes
from cap.registry import OBJECT_REGISTRY
from cap.utils.tensor_func import take_row


@OBJECT_REGISTRY.register
class MatchLabelSepEncoder(nn.Module):
    """Encode gt and matching results to separate bbox and class labels.

    Args:
        bbox_encoder: BBox label encoder
        class_encoder: Class label encoder
        cls_use_pos_only: Whether to use positive labels only during encoding.
        reg_on_hard: Regression on hard label only.
        cls_on_hard: Classification on hard label only.
    """

    def __init__(
        self,
        bbox_encoder: Optional[nn.Module] = None,
        class_encoder: Optional[nn.Module] = None,
        cls_use_pos_only: Optional[bool] = False,
        cls_on_hard: Optional[bool] = False,
        reg_on_hard: Optional[bool] = False,
    ):
        super().__init__()
        assert not (bbox_encoder is None and class_encoder is None)
        self.bbox_encoder = bbox_encoder
        self.class_encoder = class_encoder
        self.cls_use_pos_only = cls_use_pos_only
        self.reg_on_hard = reg_on_hard
        self.cls_on_hard = cls_on_hard

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        ig_flag: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:  # noqa: D205,D400
        """

        Args:
            boxes (torch.Tensor): (B, N, 4), batched predicted boxes
            gt_boxes (torch.Tensor): (B, M, 5+), batched ground
                truth boxes, might be padded.
            match_pos_flag (torch.Tensor): (B, N) matched result
                of each predicted box
            match_gt_id (torch.Tensor): (B, M) matched gt box index
                of each predicted box
            ig_flag (torch.Tensor): (B, N) ignore matched result
                of each predicted box
        """

        matched_gt_boxes = take_row(gt_boxes, match_gt_id)

        cls_label = matched_gt_boxes[..., 4]

        # set cls label of negative matchings to 0
        cls_label[match_pos_flag == 0] = 0

        pos_match = match_pos_flag > 0
        # regress on any positive match even when label is negative
        # (hard inst) or only on moderate cases
        reg_label_mask = (
            pos_match if self.reg_on_hard else pos_match * (cls_label > 0)
        )
        reg_label_mask = reg_label_mask[..., None].flatten(
            start_dim=1, end_dim=-2
        )

        out_dict = OrderedDict()

        if self.class_encoder is not None:
            if self.cls_on_hard:
                cls_label.abs_()

            # set cls label of ignored instances to be -abs(label)
            ig_index = match_pos_flag < 0
            cls_label[ig_index] = -cls_label[ig_index].abs()

            if self.cls_use_pos_only:
                cls_label[match_pos_flag == 0] = -1

            cls_label = self.class_encoder(cls_label)

            # optionally set cls_label to ignore
            if ig_flag is not None:
                cls_label[(ig_flag == 1) * (cls_label == 0)] = -1

            cls_label = cls_label.flatten(start_dim=1, end_dim=-2)
            cls_label_mask = cls_label >= 0

            out_dict.update(
                cls_label=cls_label,
                cls_label_mask=cls_label_mask,
            )

        if self.bbox_encoder:
            reg_label = self.bbox_encoder(boxes, matched_gt_boxes).flatten(
                start_dim=1, end_dim=-2
            )
            out_dict.update(
                reg_label=reg_label,
                reg_label_mask=reg_label_mask.expand_as(reg_label),
            )
        return out_dict


@OBJECT_REGISTRY.register
class XYWHBBoxEncoder(nn.Module):
    """Encode bounding box in XYWH ways (proposed in RCNN).

    Args:
        legacy_bbox: Whether to represent bbox in legacy way.
        reg_mean: Mean value to be subtracted from bbox regression task in
            each coordinate.
        reg_std: Standard deviation value to be divided from bbox regression
            task in each coordinate.
    """

    def __init__(
        self,
        legacy_bbox: Optional[bool] = False,
        reg_mean: Optional[Tuple] = (0.0, 0.0, 0.0, 0.0),
        reg_std: Optional[Tuple] = (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()

        assert len(reg_mean) == 4 and len(reg_std) == 4

        self.register_buffer(
            "reg_mean", torch.tensor(reg_mean), persistent=False
        )
        self.register_buffer(
            "reg_std", torch.tensor(reg_std), persistent=False
        )

        self._legacy_bbox = legacy_bbox

    def forward(
        self, boxes: torch.Tensor, gt_boxes: torch.Tensor
    ) -> torch.Tensor:

        gt_boxes = gt_boxes[..., :4]

        box_cx, box_cy, box_w, box_h = box_corner_to_center(
            boxes, split=True, legacy_bbox=self._legacy_bbox
        )
        gt_cx, gt_cy, gt_w, gt_h = box_corner_to_center(
            gt_boxes, split=True, legacy_bbox=self._legacy_bbox
        )

        target_dx = (gt_cx - box_cx) / box_w
        target_dy = (gt_cy - box_cy) / box_h

        target_dw = torch.where(
            gt_w > 0, torch.log(gt_w / box_w), boxes.new_zeros(1)
        )
        target_dh = torch.where(
            gt_h > 0, torch.log(gt_h / box_h), boxes.new_zeros(1)
        )

        boxes_delta = torch.cat(
            [target_dx, target_dy, target_dw, target_dh], dim=-1
        )

        return (boxes_delta - self.reg_mean) / self.reg_std


@OBJECT_REGISTRY.register
class OneHotClassEncoder(nn.Module):
    """One hot class encoder.

    Args:
        num_classes: Number of classes, including background class.
        class_agnostic_neg: Whether the negative label shoud be class
            agnostic. If not, hard instances will remain the original
            values. Otherwise, all negative labels will be set to -1.
        exclude_background: Whether to exclude background class in the
            returned label (usually class 0).
    """

    def __init__(
        self,
        num_classes: int,
        class_agnostic_neg: Optional[bool] = False,
        exclude_background: Optional[bool] = False,
    ):

        super().__init__()
        self._num_classes = num_classes
        self._class_agnostic_neg = class_agnostic_neg
        self._exclude_background = exclude_background

    def forward(self, cls_label: torch.Tensor) -> torch.Tensor:

        assert torch.all(cls_label < self._num_classes) and torch.all(
            cls_label > -self._num_classes
        )

        # get idx of ignore label
        label_ignore_mask = cls_label < 0

        all_pos_label = cls_label.detach().clone()

        # set neg label to positive to work around torch one_hot
        all_pos_label[label_ignore_mask] *= -1
        encoded_label = nn.functional.one_hot(
            all_pos_label.type(torch.long), num_classes=self._num_classes
        )

        if self._class_agnostic_neg:
            # set hard instance to label -1, and the target label
            # will be -1,-1, ....-1 along the last axis.
            encoded_label[label_ignore_mask] = -1
        else:
            # set negative labels back to negative
            encoded_label[label_ignore_mask] *= -1

        if self._exclude_background:
            encoded_label = encoded_label[..., 1:]

        return encoded_label


@OBJECT_REGISTRY.register
class RCNNKPSLabelFromMatch(nn.Module):
    """RCNN keypoints detection label encoder.

    Args:
        feat_h: the height of the output feature.
        feat_w: the width of the output feature.
        kps_num: number of keypoints to be predicted.
        ignore_labels: GT labels of keypoints which need to be ignored.
        roi_expand_param: a ratio of rois which need to be expanded.
        gauss_threshold: a threshold of score_map.
    """

    def __init__(
        self,
        feat_h: int,
        feat_w: int,
        kps_num: int,
        ignore_labels: Tuple[int],
        roi_expand_param: Optional[float] = 1.0,
        gauss_threshold: Optional[float] = 0.6,
    ):
        assert isinstance(ignore_labels, tuple)
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.kps_num = kps_num

        # init bin map
        bin_x_int = torch.arange(0, self.feat_h)
        bin_y_int = torch.arange(0, self.feat_w)
        bin_y_int, bin_x_int = torch.meshgrid(bin_y_int, bin_x_int)

        # generate index
        self.register_buffer(
            "bin_x_int", bin_x_int.flatten(), persistent=False
        )
        self.register_buffer(
            "bin_y_int", bin_y_int.flatten(), persistent=False
        )

        self.expand_param = roi_expand_param
        self.gauss_threshold = gauss_threshold
        self.ignore_labels = ignore_labels

    def get_score_map(self, center, sigma_x=1.6, sigma_y=1.6, bin_offset=0.5):
        """Get score map by gauss.

        The output of this module is a score map whose shape
        like (feat_h * feat_w,).

        Args:
            center: The projected coordinates of keypoints.
            sigma_x: Gauss sigma_x.
            sigma_y: Gauss sigma_y.
            bin_offset: the offset of bins.
        """
        # default gauss map
        x0, y0 = center[:2]
        gauss_map = torch.exp(
            -(
                (self.bin_x_int + bin_offset - x0) ** 2 / (2 * sigma_x ** 2)
                + (self.bin_y_int + bin_offset - y0) ** 2 / (2 * sigma_y ** 2)
            )
        )

        return gauss_map

    def get_reg_map(self, score_map, kps_xy, radius=1.0):
        # get ksp_xy and bin's distance.
        pos_offset_x = (kps_xy[0] - self.bin_x_int) / radius
        pos_offset_y = (kps_xy[1] - self.bin_y_int) / radius
        # get pos_indices.
        if self.gauss_threshold is not None:
            assert isinstance(self.gauss_threshold, float)
            keep_pos = torch.where(score_map >= self.gauss_threshold)[0]
        else:
            dis = pos_offset_x ** 2 + pos_offset_y ** 2
            keep_pos = torch.where((dis <= 1) & (dis >= 0))[0]

        return pos_offset_x[keep_pos], pos_offset_y[keep_pos], keep_pos

    def filter_kps(self, keypoints):
        # keypoints's shape (:, 3)
        if len(self.ignore_labels) == 1:
            return keypoints[:, 2] != self.ignore_labels[0]
        else:
            valid_mask = torch.logical_and(
                keypoints[:, 2] != self.ignore_labels[0],
                keypoints[:, 2] != self.ignore_labels[1],
            )  # noqa
            for ignore_label in self.ignore_labels[2:]:
                valid_mask = torch.logical_and(
                    valid_mask, keypoints[:, 2] != ignore_label
                )
            return valid_mask

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
    ):
        """Forward.

        The idea of top-down keypoint detection approach is
        adopted here.

        Args:
            boxes: (B, N, 4), batched predicted boxes
            gt_boxes: (B, M, 5+), batched ground truth boxes,
                might be padded.
            match_pos_flag: (B, N), matched result of each predicted
                box, Entries with value 1 represents positive
                in matching, 0 for neg and -1 for ignore.
            match_gt_id: (B, N), matched gt box index
                of each predicted box
        """

        matched_gt_boxes = take_row(gt_boxes, match_gt_id)

        batch_size_per_ctx, proposal_num = boxes.shape[:2]

        pos_match_label = (match_pos_flag > 0).float()
        pos_match_label = torch.repeat_interleave(
            pos_match_label, repeats=self.kps_num, axis=1
        )

        cls_labels = boxes.new_full(
            (
                batch_size_per_ctx,
                proposal_num,
                self.kps_num,
                self.feat_h,
                self.feat_w,
            ),
            fill_value=-1,
        )
        cls_label_weight = torch.zeros_like(cls_labels)

        reg_offset = boxes.new_zeros(
            (
                batch_size_per_ctx,
                proposal_num,
                self.kps_num * 2,
                self.feat_h,
                self.feat_w,
            )
        )
        reg_offset_weight = torch.zeros_like(reg_offset)

        for i in range(batch_size_per_ctx):
            pos_indices = torch.where(pos_match_label[i] == 1)[0]
            if len(pos_indices) == 0:
                continue

            # pos_num_boxes, 4
            box = zoom_boxes(boxes[i], (self.expand_param, self.expand_param))
            pos_gt_roi_boxes = matched_gt_boxes[i]

            # pos_num_boxes, self.kps_num * 3
            keypoints = pos_gt_roi_boxes[:, 4 : 4 + self.kps_num * 3]

            num_boxes = box.shape[0]
            keypoints = keypoints.reshape((num_boxes * self.kps_num, 3))

            scales_xy = box.new_zeros((num_boxes, 2))
            scales_xy[:, 0] = self.feat_w / (box[:, 2] - box[:, 0] + 1)
            scales_xy[:, 1] = self.feat_h / (box[:, 3] - box[:, 1] + 1)
            # (num_boxes, num_kps*2)
            scales_xy = torch.tile(scales_xy, (1, self.kps_num))
            # (num_boxes * num_kps, 2)
            scales_xy = scales_xy.reshape((num_boxes * self.kps_num, 2))
            # (num_boxes, 2)
            offsets_xy = box[:, :2]
            # (num_boxes, num_kps*2)
            offsets_xy = torch.tile(offsets_xy, (1, self.kps_num))
            # (num_boxes * num_kps, 2)
            offsets_xy = offsets_xy.reshape((num_boxes * self.kps_num, 2))
            # (num_boxes * num_kps, 2)
            kps_xy = (keypoints[:, :2] - offsets_xy) * scales_xy
            # (num_boxes * num_kps, 2)
            vis = self.filter_kps(keypoints)
            keep = torch.where(vis == 1)[0]
            keep_copy = keep.cpu().detach().numpy()
            pos_indices_copy = pos_indices.cpu().detach().numpy()
            keep = list(set(keep_copy) & set(pos_indices_copy))

            keep = [box.new_tensor(i).long() for i in keep]
            kps_label = box.new_full(
                (num_boxes * self.kps_num, self.feat_h * self.feat_w),
                fill_value=-1,
            )  # noqa

            kps_label_weight = torch.zeros_like(kps_label)
            kps_pos_offset = box.new_zeros(
                (num_boxes * self.kps_num, 2, self.feat_h * self.feat_w),
            )
            kps_pos_offset_weight = torch.zeros_like(kps_pos_offset)

            if len(keep) > 0:
                num_keep_pos = 0
                for keep_i in keep:
                    score_map = self.get_score_map(kps_xy[keep_i])
                    reg_map_x, reg_map_y, keep_pos = self.get_reg_map(
                        score_map,
                        kps_xy[keep_i],
                    )
                    if len(keep_pos) > 0:
                        kps_label[keep_i] = 0
                        kps_label_weight[keep_i] = 1.0
                        kps_label[keep_i] = score_map
                        kps_pos_offset[keep_i, 0, keep_pos] = reg_map_x
                        kps_pos_offset[keep_i, 1, keep_pos] = reg_map_y
                        kps_pos_offset_weight[keep_i, 0, keep_pos] = 1.0
                        kps_pos_offset_weight[keep_i, 1, keep_pos] = 1.0
                        num_keep_pos += len(keep_pos)

            kps_label = kps_label.view(
                (num_boxes, self.kps_num, self.feat_h, self.feat_w)
            )
            kps_label_weight = kps_label_weight.view_as(kps_label)

            kps_pos_offset = kps_pos_offset.view(
                (num_boxes, self.kps_num * 2, self.feat_h, self.feat_w)
            )
            kps_pos_offset_weight = kps_pos_offset_weight.view_as(
                kps_pos_offset
            )

            cls_labels[i] = kps_label
            cls_label_weight[i] = kps_label_weight
            reg_offset[i] = kps_pos_offset
            reg_offset_weight[i] = kps_pos_offset_weight

        return OrderedDict(
            kps_cls_label=cls_labels,
            kps_cls_label_weight=cls_label_weight,
            kps_reg_label=reg_offset,
            kps_reg_label_weight=reg_offset_weight,
        )


@OBJECT_REGISTRY.register
class RCNNBinDetLabelFromMatch(nn.Module):
    """RCNN bin detection label encoder.

    Bin detection is the detection task in the areas of
    bins which are parents boxes.

    Get label by anchor match. For example if anchor is matched by A,
    then A's class label is GT's class label.

    Args:
        roi_h_zoom_scale: Zoom scale of roi's height.
        roi_w_zoom_scale: Zoom scale of roi's width.
        feature_h: Roi featuremap's height.
        feature_w: Roi featuremap's width.
        num_classes: Num of classes.
        cls_on_hard: Classification on hard label only.
        allow_low_quality_heatmap: Whether to allow low quality heatmap.
    """

    def __init__(
        self,
        roi_h_zoom_scale,
        roi_w_zoom_scale,
        feature_h,
        feature_w,
        num_classes,
        cls_on_hard,
        allow_low_quality_heatmap=False,
    ):
        super(RCNNBinDetLabelFromMatch, self).__init__()
        self.cls_on_hard = cls_on_hard
        self.num_classes = num_classes
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.roi_zoom_scale_wh = (roi_w_zoom_scale, roi_h_zoom_scale)
        self.allow_low_quality_heatmap = allow_low_quality_heatmap

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        ig_flag: Optional[torch.Tensor] = None,
    ):
        """Forward.

        Args:
            boxes: With shape (N, 4+) or (B, N, 4+),
                where 4 represents (x1, y1, x2, y2)
            gt_boxes: With shape (B, num_gt_box, 5+),
                where 5 represents (x1, y1, x2, y2, class_id)
            match_pos_flag: With shape (B, num_anchors),
                value 1: pos, 0: neg, -1: ignore
            match_gt_id: With shape (B, num_anchors),
                the best matched gt box id, -1 means unavailable
            ig_flag: With shape (B, N), ignore matched result
                of each predicted box

        Returns:
            non_neg_match_label: With shape (B, num_anchors, 1)
                match_pos_flag > 0: label > 0 or label <0, depends on roi label
                match_pos_flag == 0: label == 0
                match_pos_flag < 0: label < 0

            label_map: With shape (B * num_anchors, num_classes, w, h)
            offset: With shape (B * num_anchors, 4, w, h)
            mask: With shape (B * num_anchors, num_classes)
        """

        gt_anchor_box = take_row(gt_boxes, match_gt_id)
        box_label = torch.reshape(
            gt_anchor_box[..., [4]], (gt_anchor_box.shape[0], -1)
        )

        label_map, offset = self.get_label(boxes, gt_anchor_box)

        pos_match = match_pos_flag > 0
        # get non-neg match label, and set neg label to 0
        non_neg_match_label = torch.where(
            match_pos_flag != 0, box_label, torch.zeros_like(box_label)
        )

        mask_one_dim = (
            pos_match
            if self.cls_on_hard
            else pos_match * (non_neg_match_label > 0)
        )

        mask = torch.cat(
            [
                torch.ones_like(box_label).unsqueeze(-1) * i
                for i in range(self.num_classes)
            ],
            dim=2,
        )

        mask = torch.where(
            mask == torch.abs(box_label).unsqueeze(-1).expand(mask.shape) - 1,
            torch.unsqueeze(mask_one_dim, -1).expand(mask.shape),
            torch.zeros_like(box_label)
            .unsqueeze(-1)
            .expand(-1, -1, self.num_classes)
            .bool(),
        )

        label_map = torch.reshape(
            label_map, (-1, self.num_classes, self.feature_w, self.feature_h)
        )
        offset = torch.reshape(offset, (-1, 4, self.feature_w, self.feature_h))
        mask = torch.reshape(mask, (-1, self.num_classes))

        if ig_flag is not None:
            mask = mask * (ig_flag != 1).view(mask.shape)

        return OrderedDict(label_map=label_map, offset=offset, mask=mask)

    def get_label(self, anchors, gt_anchor_box):
        """Get label.

        Args:
            anchors: With shape (B, num_anchors, 4+)
            gt_anchor_box: With shape (B, num_anchors, 5)

        Returns:
            labelmap_onehot_label: With shape (B, num_anchors, w, h)
            relative_box: gt_box(subbox) relative to rois(anchors)

            labelmap: With shape (B, num_anchors, num_classes, w, h)
            offset: With shape (B, num_anchors, 4, w, h)
        """

        anchors = zoom_boxes(anchors, self.roi_zoom_scale_wh)
        box_label = gt_anchor_box[..., [4]].reshape(gt_anchor_box.shape[0], -1)

        anchors_xmin = anchors[..., [0]]
        anchors_ymin = anchors[..., [1]]
        anchors_xmax = anchors[..., [2]]
        anchors_ymax = anchors[..., [3]]
        relative_box = gt_anchor_box[..., [0, 1, 2, 3]] - torch.cat(
            (anchors_xmin, anchors_ymin, anchors_xmin, anchors_ymin), dim=-1
        )

        relative_box_xmin = relative_box[..., [0]]
        relative_box_ymin = relative_box[..., [1]]
        relative_box_xmax = relative_box[..., [2]]
        relative_box_ymax = relative_box[..., [3]]

        relative_box_w = relative_box_xmax - relative_box_xmin
        relative_box_h = relative_box_ymax - relative_box_ymin
        relative_box_center_x = (relative_box_xmin + relative_box_xmax) / 2
        relative_box_center_y = (relative_box_ymin + relative_box_ymax) / 2

        anchors_w = anchors_xmax - anchors_xmin
        anchors_h = anchors_ymax - anchors_ymin
        strides_w = anchors_w / self.feature_w
        strides_h = anchors_h / self.feature_h

        ind_w_feature = torch.cat(
            [
                torch.ones_like(box_label)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, self.feature_h, 1)
                * i
                for i in range(self.feature_w)
            ],
            dim=3,
        )

        ind_h_feature = torch.cat(
            [
                torch.ones_like(box_label)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, 1, self.feature_w)
                * i
                for i in range(self.feature_h)
            ],
            dim=2,
        )

        # int position of future map(feature_h* feature_w)
        def gen_gaussian_label_map(ind_w_feature, ind_h_feature):
            # sigma: (w/2) or (h/2)
            w_sigma = (
                torch.div(relative_box_w / 2, strides_w)
                .unsqueeze(-1)
                .expand(-1, -1, self.feature_w, self.feature_h)
            )
            h_sigma = (
                torch.div(relative_box_h / 2, strides_h)
                .unsqueeze(-1)
                .expand(-1, -1, self.feature_w, self.feature_h)
            )

            if self.allow_low_quality_heatmap:
                w_sigma_for_score = torch.clamp(w_sigma, min=1.5)
                h_sigma_for_score = torch.clamp(h_sigma, min=1.5)
                w_sigma_for_cond = torch.clamp(w_sigma, min=0.5)
                h_sigma_for_cond = torch.clamp(h_sigma, min=0.5)
            else:
                w_sigma_for_score = w_sigma
                h_sigma_for_score = h_sigma
                w_sigma_for_cond = w_sigma
                h_sigma_for_cond = h_sigma

            label_map_position_w = (
                torch.div(relative_box_center_x, strides_w)
                .unsqueeze(-1)
                .expand(-1, -1, self.feature_w, self.feature_h)
            )
            label_map_position_h = (
                torch.div(relative_box_center_y, strides_h)
                .unsqueeze(-1)
                .expand(-1, -1, self.feature_w, self.feature_h)
            )

            w_term = torch.square(
                (label_map_position_w - ind_w_feature - 0.5)
                / w_sigma_for_score
            )
            h_term = torch.square(
                (label_map_position_h - ind_h_feature - 0.5)
                / h_sigma_for_score
            )

            gaussian_label_map = torch.exp(-(w_term + h_term))

            cond_w = torch.abs(ind_w_feature + 0.5 - label_map_position_w)
            cond_h = torch.abs(ind_h_feature + 0.5 - label_map_position_h)
            cond = torch.logical_and(
                cond_w < w_sigma_for_cond, cond_h < h_sigma_for_cond
            )

            gaussian_label_map = torch.where(
                cond, gaussian_label_map, torch.zeros_like(gaussian_label_map)
            )

            # return gaussian_label_map
            return gaussian_label_map

        labelmap_onehot_label = gen_gaussian_label_map(
            ind_w_feature, ind_h_feature
        )

        label_map_offset_x1 = torch.div(
            relative_box_xmin, strides_w
        ).unsqueeze(-1).expand(ind_w_feature.shape) - (ind_w_feature + 0.5)
        label_map_offset_y1 = torch.div(
            relative_box_ymin, strides_h
        ).unsqueeze(-1).expand(ind_h_feature.shape) - (ind_h_feature + 0.5)
        label_map_offset_x2 = torch.div(
            relative_box_xmax, strides_w
        ).unsqueeze(-1).expand(ind_w_feature.shape) - (ind_w_feature + 0.5)
        label_map_offset_y2 = torch.div(
            relative_box_ymax, strides_h
        ).unsqueeze(-1).expand(ind_h_feature.shape) - (ind_h_feature + 0.5)

        offset = torch.cat(
            (
                label_map_offset_x1.unsqueeze(2),
                label_map_offset_y1.unsqueeze(2),
                label_map_offset_x2.unsqueeze(2),
                label_map_offset_y2.unsqueeze(2),
            ),
            dim=2,
        )

        offset = torch.reshape(
            offset,
            (
                offset.shape[0],
                offset.shape[1],
                -1,
                self.feature_w,
                self.feature_h,
            ),
        )

        label_map = torch.cat(
            [
                torch.ones_like(box_label)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, self.feature_w, self.feature_h)
                * i
                for i in range(self.num_classes)
            ],
            dim=2,
        )
        cond = (
            label_map
            == torch.abs(box_label)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(label_map.shape)
            - 1
        )

        x = torch.reshape(
            labelmap_onehot_label,
            (
                labelmap_onehot_label.shape[0],
                -1,
                1,
                self.feature_w,
                self.feature_h,
            ),
        ).expand(label_map.shape)

        y = (
            torch.zeros_like(box_label)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.num_classes, self.feature_w, self.feature_h)
        )

        label_map = torch.where(cond, x, y)

        def process_small_stride(stride, num_channel, data):
            cond = (
                torch.unsqueeze(stride, -1)
                .unsqueeze(-1)
                .expand(-1, -1, num_channel, self.feature_w, self.feature_h)
                < 1e-2
            )
            return torch.where(cond, torch.zeros_like(data), data)

        label_map = process_small_stride(
            strides_w, self.num_classes, label_map
        )
        label_map = process_small_stride(
            strides_h, self.num_classes, label_map
        )
        offset = process_small_stride(strides_w, 4, offset)
        offset = process_small_stride(strides_h, 4, offset)

        return label_map, offset


@OBJECT_REGISTRY.register
class MatchLabelGroundLineEncoder(nn.Module):
    """RCNN vehicle ground line label encoder.

    This class encodes gt and matching results to separate
    bbox and class labels.

    Args:
        limit_reg_length: Whether to limit the length of regression.
        cls_use_pos_only: Whether to use positive labels only during
            encoding. Default is False.
        cls_on_hard: Whether to classification on hard label only.
            Default is False.
        reg_on_hard: Whether to regression on hard label only.
            Default is False.
    """

    def __init__(
        self,
        limit_reg_length: bool = False,
        cls_use_pos_only: bool = False,
        cls_on_hard: bool = False,
        reg_on_hard: bool = False,
    ):
        super().__init__()
        self.limit_reg_length = limit_reg_length
        self.cls_use_pos_only = cls_use_pos_only
        self.reg_on_hard = reg_on_hard
        self.cls_on_hard = cls_on_hard

    @staticmethod
    def get_intersections_to_vertical(points, coord_x1, coord_x2):
        """Intersection coordinates."""
        x1, y1, x2, y2 = torch.split(points, 1, dim=-1)
        delta_x = x1 - x2
        delta_y = y1 - y2
        coord_y1 = (delta_y / delta_x) * (coord_x1 - x2) + y2
        coord_y2 = (delta_y / delta_x) * (coord_x2 - x2) + y2
        coord_y1[delta_x == 0] = 0
        coord_y2[delta_x == 0] = 0
        return coord_y1, coord_y2, delta_x != 0

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_flanks: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:  # noqa: D205,D400

        matched_gt_boxes = take_row(gt_boxes, match_gt_id)
        matched_gt_flanks = take_row(gt_flanks, match_gt_id)

        cls_label = matched_gt_boxes[..., 4]
        matched_gdls = matched_gt_flanks[..., :4]
        gdls_cls = matched_gt_flanks[..., 8]

        pos_match = match_pos_flag > 0
        # regress on any positive match even when label is negative
        # (hard inst) or only on moderate cases
        reg_label_mask = (
            pos_match * (gdls_cls != 0)
            if self.reg_on_hard
            else pos_match * (cls_label > 0)
        )
        reg_label_mask = reg_label_mask[..., None].flatten(
            start_dim=1, end_dim=-2
        )

        # set cls label of negative matchings to 0
        gdls_cls[match_pos_flag == 0] = 0
        if self.cls_on_hard:
            gdls_cls.abs_()

        gdls_cls[match_pos_flag < 0] = -1

        if self.cls_use_pos_only:
            gdls_cls[match_pos_flag == 0] = -1

        gdls_cls[cls_label == 0] = -1

        gdls_cls = gdls_cls[..., None]
        cls_label_mask = gdls_cls >= 0

        out_dict = OrderedDict(
            cls_label=gdls_cls,
            cls_label_mask=cls_label_mask,
        )

        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = torch.split(boxes, 1, dim=-1)
        coords_y1, coords_y2, inter_mask = self.get_intersections_to_vertical(
            matched_gdls,
            boxes_x1,
            boxes_x2,
        )

        boxes_width = boxes_x2 - boxes_x1
        boxes_height = boxes_y2 - boxes_y1
        boxes_mask = torch.logical_and(boxes_width > 0, boxes_height > 0)
        mask = torch.logical_and(boxes_mask, inter_mask)

        if self.limit_reg_length:
            delta_left = coords_y1 - boxes_y1
            delta_right = coords_y2 - boxes_y2
            delta_mask = torch.logical_and(
                delta_left > 0, delta_right > 0
            )  # noqa
            mask = torch.logical_and(mask, delta_mask)
            reg_label = torch.log(
                torch.cat([delta_left, delta_right], dim=-1), boxes_height
            )
        else:
            coords_ys = torch.cat([coords_y1, coords_y2], dim=-1)
            reg_label = (coords_ys - boxes_y2) / boxes_height

        reg_label_mask *= mask

        out_dict.update(
            reg_label=reg_label,
            reg_label_mask=reg_label_mask.expand_as(reg_label),
        )

        return out_dict


@OBJECT_REGISTRY.register
class RCNN3DLabelFromMatch(RCNNKPSLabelFromMatch):
    """RCNN 3d label encoder.

    Args:
        feat_h: Roi featuremap's height.
        feat_w: Roi featuremap's width.
        kps_num: number of keypoints to be predicted, its value
            must be 1 due to the center of box.
        gauss_threshold: a threshold of score_map.
        gauss_3d_threshold: a threshold of 3d offset reg map.
        gauss_depth_threshold: a threshold of depth reg map.
        gauss_dim_threshold: a threshold of 3d dim reg map.
        undistort_depth_uv: whether depth label is undistort into depth_u/v.
        roi_expand_param: a ratio of rois which need to be expanded.
    """

    def __init__(
        self,
        feat_h: int,
        feat_w: int,
        kps_num: int,
        gauss_threshold: float,
        gauss_3d_threshold: float,
        gauss_depth_threshold: float,
        undistort_depth_uv: bool = False,
        roi_expand_param: Optional[float] = 1.0,
    ):

        assert kps_num == 1
        super().__init__(
            feat_h, feat_w, kps_num, (0,), roi_expand_param, gauss_threshold
        )

        self.undistort_depth_uv = undistort_depth_uv
        self.gauss_3d_threshold = gauss_3d_threshold
        self.gauss_depth_threshold = gauss_depth_threshold

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
    ):

        matched_gt_boxes = take_row(gt_boxes, match_gt_id)

        # (batch_size_per_ctx, proposal_num, 4)
        batch_size_per_ctx, proposal_num = boxes.shape[:2]

        pos_match_label = (match_pos_flag > 0).float()
        # (batch_size_per_ctx, proposal_num)
        pos_match_label = torch.repeat_interleave(
            pos_match_label, repeats=self.kps_num, axis=1
        )

        cls_labels = boxes.new_full(
            (
                batch_size_per_ctx,
                proposal_num,
                self.kps_num,
                self.feat_h,
                self.feat_w,
            ),
            fill_value=-1,
        )
        cls_label_weight = torch.zeros_like(cls_labels, dtype=torch.float32)

        # (batch_size_per_ctx, proposal_num, 14/15)
        if self.undistort_depth_uv:
            dim_loc_r_y = matched_gt_boxes[:, :, 8:].clone()
        else:
            dim_loc_r_y = matched_gt_boxes[:, :, 7:].clone()
        assert dim_loc_r_y.shape[-1] == 7, "{}".format(dim_loc_r_y.shape)

        reg_2d_offsets = boxes.new_zeros(
            (
                batch_size_per_ctx,
                proposal_num,
                self.kps_num * 2,
                self.feat_h,
                self.feat_w,
            ),
        )
        reg_2d_offset_weights = torch.zeros_like(reg_2d_offsets)

        reg_3d_offsets = boxes.new_zeros(
            (
                batch_size_per_ctx,
                proposal_num,
                self.kps_num * 2,
                self.feat_h,
                self.feat_w,
            ),
        )
        reg_3d_offset_weights = torch.zeros_like(reg_3d_offsets)

        if self.undistort_depth_uv:
            reg_depth_u = boxes.new_zeros(
                (
                    batch_size_per_ctx,
                    proposal_num,
                    self.kps_num,
                    self.feat_h,
                    self.feat_w,
                ),
            )
            reg_depth_v = boxes.new_zeros(
                (
                    batch_size_per_ctx,
                    proposal_num,
                    self.kps_num,
                    self.feat_h,
                    self.feat_w,
                ),
            )
            reg_depth_weights = torch.zeros_like(reg_depth_u)
        else:
            reg_depths = boxes.new_zeros(
                (
                    batch_size_per_ctx,
                    proposal_num,
                    self.kps_num,
                    self.feat_h,
                    self.feat_w,
                ),
            )
            reg_depth_weights = torch.zeros_like(reg_depths)

        for idx in range(batch_size_per_ctx):
            pos_indices = torch.where(pos_match_label[idx] == 1)[0]
            if len(pos_indices) == 0:
                continue

            # pos_num_boxes, 4
            box = zoom_boxes(
                boxes[idx], (self.expand_param, self.expand_param)
            )
            pos_gt_roi_boxes = matched_gt_boxes[idx]  # 128,14
            # pos_num_boxes, self.kps_num * 4
            gt_2d_bbox = pos_gt_roi_boxes[:, : self.kps_num * 4]
            # pos_num_boxes, self.kps_num * 2
            centers = (gt_2d_bbox[:, :2] + gt_2d_bbox[:, 2:]) / 2.0
            location_offsets = pos_gt_roi_boxes[:, 4:6]
            if self.undistort_depth_uv:
                depth_u = pos_gt_roi_boxes[:, 6:7]
                depth_v = pos_gt_roi_boxes[:, 7:8]
            else:
                depth = pos_gt_roi_boxes[:, 6:7]

            num_boxes = box.shape[0]
            # num_boxes * 2
            centers = centers.reshape((num_boxes, 2))

            scales_xy = torch.zeros((num_boxes, 2), dtype=torch.float32).to(
                centers.device
            )
            scales_xy[:, 0] = self.feat_w / (box[:, 2] - box[:, 0] + 1)
            scales_xy[:, 1] = self.feat_h / (box[:, 3] - box[:, 1] + 1)
            # (num_boxes, num_kps*2)
            scales_xy = torch.tile(scales_xy, (1, self.kps_num))
            # (num_boxes * num_kps, 2)
            scales_xy = scales_xy.reshape((num_boxes * self.kps_num, 2))
            # (num_boxes, 2)
            offsets_xy = box[:, :2]
            # (num_boxes, num_kps*2)
            offsets_xy = torch.tile(offsets_xy, (1, self.kps_num))
            # (num_boxes * num_kps, 2)
            offsets_xy = offsets_xy.reshape((num_boxes * self.kps_num, 2))
            # (num_boxes * num_kps, 2)
            kps_xy = (centers[:, :2] - offsets_xy) * scales_xy
            keep = list(pos_indices)
            kps_label = torch.full(
                (num_boxes * self.kps_num, self.feat_h * self.feat_w),
                fill_value=-1,
                dtype=torch.float32,
            )  # noqa
            kps_label_weight = torch.zeros(
                kps_label.shape, dtype=torch.float32
            )

            kps_pos_offset = kps_xy.new_zeros(
                (num_boxes * self.kps_num, 2, self.feat_h * self.feat_w)
            )
            kps_pos_offset_weight = torch.zeros(
                kps_pos_offset.shape, dtype=torch.float32
            )

            kps_3d_offset = kps_xy.new_zeros(
                (num_boxes * self.kps_num, 2, self.feat_h * self.feat_w)
            )
            kps_3d_offset_weight = torch.zeros(
                kps_3d_offset.shape, dtype=torch.float32
            )

            if self.undistort_depth_uv:
                kps_depth_u = torch.zeros(
                    (num_boxes * self.kps_num, 1, self.feat_h * self.feat_w),
                    dtype=torch.float32,
                )
                kps_depth_v = torch.zeros(
                    (num_boxes * self.kps_num, 1, self.feat_h * self.feat_w),
                    dtype=torch.float32,
                )
                kps_depth_weight = torch.zeros(
                    kps_depth_u.shape, dtype=torch.float32
                )
            else:
                kps_depth = torch.zeros(
                    (num_boxes * self.kps_num, 1, self.feat_h * self.feat_w),
                    dtype=torch.float32,
                )
                kps_depth_weight = torch.zeros(
                    kps_depth.shape, dtype=torch.float32
                )

            if len(keep) > 0:
                # 平均到每个proposal
                # num_keep_per_box = float(len(keep)) / num_boxes
                num_keep_pos = 0
                for keep_i in keep:
                    score_map = self.get_score_map(kps_xy[keep_i])
                    reg_map_x, reg_map_y, keep_pos = self.get_reg_map(
                        score_map,
                        kps_xy[keep_i],
                    )
                    if len(keep_pos) > 0:
                        kps_label[keep_i] = 0
                        kps_label_weight[keep_i] = 1.0
                        kps_label[keep_i] = score_map
                        kps_pos_offset[keep_i, 0, keep_pos] = reg_map_x
                        kps_pos_offset[keep_i, 1, keep_pos] = reg_map_y
                        kps_pos_offset_weight[keep_i, 0, keep_pos] = 1.0
                        kps_pos_offset_weight[keep_i, 1, keep_pos] = 1.0
                        num_keep_pos += len(keep_pos)

                    (
                        reg_map_3d_x,
                        reg_map_3d_y,
                        keep_pos_3d,
                    ) = self.get_3d_offset_reg_map(
                        score_map,
                        location_offsets[keep_i],
                    )
                    if len(keep_pos_3d) > 0:
                        kps_3d_offset[keep_i, 0, keep_pos_3d] = reg_map_3d_x
                        kps_3d_offset[keep_i, 1, keep_pos_3d] = reg_map_3d_y
                        kps_3d_offset_weight[keep_i, 0, keep_pos_3d] = 1.0
                        kps_3d_offset_weight[keep_i, 1, keep_pos_3d] = 1.0

                    if self.undistort_depth_uv:
                        depth_u_map, keep_pos_depth = self.get_depth_reg_map(
                            score_map,
                            depth_u[keep_i],
                        )
                        depth_v_map, keep_pos_depth = self.get_depth_reg_map(
                            score_map,
                            depth_v[keep_i],
                        )
                        if len(keep_pos_depth) > 0:
                            kps_depth_u[keep_i] = 0
                            kps_depth_v[keep_i] = 0
                            kps_depth_weight[keep_i, 0, keep_pos_depth] = 1.0
                            kps_depth_u[keep_i] = depth_u_map
                            kps_depth_v[keep_i] = depth_v_map
                    else:
                        depth_map, keep_pos_depth = self.get_depth_reg_map(
                            score_map,
                            depth[keep_i],
                        )
                        if len(keep_pos_depth) > 0:
                            kps_depth[keep_i] = 0
                            kps_depth_weight[keep_i, 0, keep_pos_depth] = 1.0
                            kps_depth[keep_i] = depth_map

            kps_label = kps_label.view(
                (num_boxes, self.kps_num, self.feat_h, self.feat_w)
            )

            kps_label_weight = kps_label_weight.view_as(kps_label)

            kps_pos_offset = kps_pos_offset.view(
                (num_boxes, self.kps_num * 2, self.feat_h, self.feat_w)
            )
            kps_pos_offset_weight = kps_pos_offset_weight.view_as(
                kps_pos_offset
            )

            kps_3d_offset = kps_3d_offset.view(
                (num_boxes, self.kps_num * 2, self.feat_h, self.feat_w)
            )
            kps_3d_offset_weight = kps_3d_offset_weight.view_as(kps_3d_offset)
            if self.undistort_depth_uv:
                kps_depth_u = kps_depth_u.view(
                    (num_boxes, self.kps_num, self.feat_h, self.feat_w)
                )
                kps_depth_v = kps_depth_v.view(
                    (num_boxes, self.kps_num, self.feat_h, self.feat_w)
                )
                kps_depth_weight = kps_depth_weight.view_as(kps_depth_u)
                reg_depth_u[idx] = kps_depth_u
                reg_depth_v[idx] = kps_depth_v
            else:
                kps_depth = kps_depth.view(
                    (num_boxes, self.kps_num, self.feat_h, self.feat_w)
                )
                kps_depth_weight = kps_depth_weight.view_as(kps_depth)
                reg_depths[idx] = kps_depth

            cls_labels[idx] = kps_label
            cls_label_weight[idx] = kps_label_weight
            reg_2d_offsets[idx] = kps_pos_offset
            reg_2d_offset_weights[idx] = kps_pos_offset_weight
            reg_3d_offsets[idx] = kps_3d_offset
            reg_3d_offset_weights[idx] = kps_3d_offset_weight
            reg_depth_weights[idx] = kps_depth_weight

        labels = OrderedDict(
            kps_cls_label=cls_labels,
            kps_cls_label_weight=cls_label_weight,
            kps_2d_offset=reg_2d_offsets,
            kps_2d_offset_weight=reg_2d_offset_weights,
            kps_3d_offset=reg_3d_offsets,
            kps_3d_offset_weight=reg_3d_offset_weights,
            kps_depth_weight=reg_depth_weights,
            dim_loc_r_y=dim_loc_r_y,
            pos_match=pos_match_label,
            rois=boxes,
        )
        if self.undistort_depth_uv:
            labels.update(
                OrderedDict(
                    kps_depth_u=reg_depth_u,
                    kps_depth_v=reg_depth_v,
                )
            )
        else:
            labels.update(
                OrderedDict(
                    kps_depth=reg_depths,
                )
            )

        return labels

    def get_depth_reg_map(self, score_map, depth):
        # get ksp_xy and bin's distance.
        reg_depth = torch.zeros_like(score_map)
        # get pos_indices.
        assert isinstance(self.gauss_depth_threshold, float)
        keep_pos = torch.where(score_map >= self.gauss_depth_threshold)[0]
        reg_depth[keep_pos] = depth
        return reg_depth, keep_pos

    def get_3d_offset_reg_map(
        self,
        score_map,
        loc_offset,
    ):
        reg_y = torch.zeros_like(score_map)
        reg_x = torch.zeros_like(score_map)
        # get pos_indices.
        assert isinstance(self.gauss_3d_threshold, float)
        keep_pos = torch.where(score_map >= self.gauss_3d_threshold)[0]
        reg_y[keep_pos] = loc_offset[1]
        reg_x[keep_pos] = loc_offset[0]
        return reg_x[keep_pos], reg_y[keep_pos], keep_pos

    def filter_kps(self, keypoints):
        return torch.logical_and(keypoints[:, 2] > 0, keypoints[:, 2] < 3)


@OBJECT_REGISTRY.register
class RCNNMultiBinDetLabelFromMatch(RCNNBinDetLabelFromMatch):
    """RCNN Multi bin detection label one_hot encoder.

    Args:
        feature_h: Height of feature map.
        feature_w: Width of feature map.
        num_classes: Number of class.
        max_subbox_num: Max number of subbox.
        cls_on_hard: Whether classify on hard label.
            Defaults to False.
        roi_h_zoom_scale: Zoom scale of roi's height.
            Defaults to 1.
        roi_w_zoom_scale: Zoom scale of roi's width.
            Defaults to 1.
    """

    def __init__(
        self,
        feature_h: int,
        feature_w: int,
        num_classes: int,
        max_subbox_num: int,
        cls_on_hard: bool = False,
        reg_on_hard: bool = False,
        use_ig_region: bool = True,
        roi_h_zoom_scale=1,
        roi_w_zoom_scale=1,
    ):
        super(RCNNMultiBinDetLabelFromMatch, self).__init__(
            roi_h_zoom_scale,
            roi_w_zoom_scale,
            feature_h,
            feature_w,
            num_classes,
            cls_on_hard,
        )
        self.max_subbox_num = max_subbox_num
        self.reg_on_hard = reg_on_hard
        self.use_ig_region = use_ig_region

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        ig_regions: torch.Tensor = None,
        ig_regions_num: torch.Tensor = None,
    ):
        """Forward.

        Args:
            boxes: With shape (B, N, 4+),
                where 4 represents (x1, y1, x2, y2)
            gt_boxes: With shape (B, max_num_box, max_num_sub, 5+),
                where 5 represents (x1, y1, x2, y2, class_id)
            match_pos_flag: With shape (B, N),
                value 1: pos, 0: neg, -1: ignore
            match_gt_id: With shape (B, N),
                the best matched gt box id, -1 means unavailable
            ig_regions: With shape (B, max_num_box, max_num_sub, 5+),
                where 5 represents (x1, y1, x2, y2, class_id)
            ig_regions_num: With shape (B, max_num_box)

        Returns:
            non_neg_match_label: With shape (B, num_anchors, 1)
                match_pos_flag > 0: label > 0 or label <0, depends on roi label
                match_pos_flag == 0: label == 0
                match_pos_flag < 0: label < 0

            label_map: With shape (B * num_anchors, num_classes, w, h)
            offset: With shape (B * num_anchors, 4, w, h)
            label_mask: With shape (B * num_anchors,
                    num_classes, feature_h, feature_w)
            offset_mask: With shape (B * num_anchors, 4, feature_h, feature_w)
        """
        gt_boxes = gt_boxes.flatten(start_dim=2)
        # with shape (B,N,max_num_sub,5+)
        gt_boxes = take_row(gt_boxes, match_gt_id).unflatten(
            -1, (self.max_subbox_num, -1)
        )
        # with shape: (B,num_anchors,num_class,H,W),
        # (B,num_anchors,4,H,W), (B,num_anchors,num_class,H,W)
        label_map, offset, hard_mask = self.get_label(
            boxes, gt_boxes, match_pos_flag
        )

        # with shape (B*num_anchors,num_class,H,W)
        label_map = label_map.reshape(
            -1, self.num_classes, self.feature_h, self.feature_w
        )
        # with shape (B*num_anchors,4,H,W)
        offset = offset.reshape(-1, 4, self.feature_h, self.feature_w)
        # with shape (B*num_anchors,num_class,H,W)
        hard_mask = hard_mask.reshape(
            -1, self.num_classes, self.feature_h, self.feature_w
        )
        # with shape (B*num_anchors,1,1,1)
        pos_mask = (
            (match_pos_flag > 0)
            .flatten()
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        # with shape (B*num_anchors,num_class,H,W)
        label_mask = pos_mask.expand_as(label_map)
        if not self.cls_on_hard:
            label_mask = torch.logical_and(label_mask, hard_mask <= 0)

        # with shape (B*num_anchors,4,H,W)
        offset_mask = torch.logical_and(
            pos_mask.expand_as(offset),
            (label_map > 0).any(dim=1, keepdim=True).expand_as(offset),
        )
        if not self.reg_on_hard:
            offset_mask = torch.logical_and(
                offset_mask,
                (hard_mask <= 0)
                .any(dim=1, keepdim=True)
                .expand_as(offset_mask),
            )

        if self.use_ig_region and ig_regions is not None:
            ig_regions = ig_regions.flatten(start_dim=2)
            # with shape (B,N,max_num_sub,5+)
            ig_regions = take_row(ig_regions, match_gt_id).unflatten(
                -1, (self.max_subbox_num, -1)
            )
            ig_label, _, _ = self.get_label(boxes, ig_regions)
            # with shape (B*num_anchors,num_class,H,W)
            ig_label = label_map.reshape(
                -1, self.num_classes, self.feature_h, self.feature_w
            )
            ig_mask = torch.logical_and(
                ig_label > 0,
                (match_pos_flag > 0)
                .flatten()
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1),
            )
            # set negative position mask to zero
            label_mask = torch.where(
                ig_mask,
                torch.logical_or(ig_mask, label_mask),
                label_mask,
            )
            ig_mask = ig_mask.any(dim=1, keepdim=True).expand_as(offset_mask)
            offset_mask = torch.where(
                ig_mask,
                torch.logical_or(ig_mask, offset_mask),
                offset_mask,
            )

        return OrderedDict(
            label_map=label_map,
            offset=offset,
            label_mask=label_mask,
            offset_mask=offset_mask,
        )

    def get_label(self, anchors, gt_boxes, match_pos_flag=None):
        """Get label.

        Args:
            anchors: With shape (B, num_anchors, 4+)
            gt_boxes: With shape (B,num_anchors,max_num_sub,5+)
            match_pos_flag: With shape (B,num_anchors); pass this for debug

        Returns:
            label_tgt: With shape (B, num_anchors, num_class, H, W).
            offset_tgt: With shape (B, num_anchors, 4, H, W).
            hard_mash: With shape (B, num_anchors, num_class, H, W).
        """
        batch_size, num_anchors, *_ = gt_boxes.shape
        anchors: torch.Tensor = zoom_boxes(anchors, self.roi_zoom_scale_wh)
        # with shape (B,num_anchors,max_num_sub,1)
        gt_mask = ~(gt_boxes == 0).all(-1, keepdim=True)
        # with shape (B,num_anchors,max_num_sub,4+)
        anchors = anchors[:, :, None].expand((-1, -1, self.max_subbox_num, -1))
        # with shape (B,num_anchors,max_num_sub,1)
        ori_box_label = gt_boxes[..., 4:5]
        if match_pos_flag is not None:
            # with shape (B,num_anchors,1,1)
            match_pos_flag = match_pos_flag.unsqueeze(-1).unsqueeze(-1)
            gt_mask = torch.logical_and(gt_mask, match_pos_flag)
            ori_box_label = torch.where(
                match_pos_flag > 0,
                ori_box_label,
                torch.zeros_like(ori_box_label),
            )
        box_label = torch.abs(ori_box_label)

        # with shape (B,num_anchors,max_num_sub,1)
        anchors_xmin = anchors[..., [0]]
        anchors_ymin = anchors[..., [1]]
        anchors_xmax = anchors[..., [2]]
        anchors_ymax = anchors[..., [3]]
        relative_box = gt_boxes[..., :4] - torch.cat(
            (anchors_xmin, anchors_ymin, anchors_xmin, anchors_ymin), dim=-1
        )

        relative_box_xmin = relative_box[..., [0]]
        relative_box_ymin = relative_box[..., [1]]
        relative_box_xmax = relative_box[..., [2]]
        relative_box_ymax = relative_box[..., [3]]

        relative_box_w = relative_box_xmax - relative_box_xmin
        relative_box_center_x = (relative_box_xmin + relative_box_xmax) / 2
        relative_box_center_y = (relative_box_ymin + relative_box_ymax) / 2

        strides_w = (anchors_xmax - anchors_xmin) / self.feature_w
        strides_h = (anchors_ymax - anchors_ymin) / self.feature_h

        # with shape (B,num_anchors,num_class,H,W)
        hard_mask = torch.zeros(
            (
                batch_size,
                num_anchors,
                self.num_classes,
                self.feature_h,
                self.feature_w,
            ),
            device=anchors.device,
        )
        # with shape (B,num_anchors,num_class,H,W)
        label_tgt = torch.zeros_like(hard_mask)
        # with shape (B,num_anchors,4,H,W)
        offset_tgt = torch.zeros(
            (batch_size, num_anchors, 4, self.feature_h, self.feature_w),
            device=anchors.device,
        )
        # with shape (1,1,1,feat_h,feat_w)
        ys, xs = torch.meshgrid(
            torch.arange(self.feature_h).to(anchors.device),
            torch.arange(self.feature_w).to(anchors.device),
        )
        ys = ys.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        xs = xs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # with shape (1,1,num_class,1,1)
        label_inds = (
            torch.arange(self.num_classes, device=anchors.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        # with shape (B,num_anchors,max_num_sub,1)
        arg_w_inds = torch.argsort(relative_box_w, dim=-2)
        for i in range(self.max_subbox_num):
            # with shape (B,num_anchors,1,1)
            inds = arg_w_inds[:, :, i : i + 1, :]
            # with shape (B,num_anchors,1,1,1)
            inst_gt_mask = gt_mask.gather(-2, inds).unsqueeze(-1)
            # with shape (B,num_anchors,1,1,1)
            relative_cx = relative_box_center_x.gather(-2, inds).unsqueeze(-1)
            relative_cy = relative_box_center_y.gather(-2, inds).unsqueeze(-1)
            # with shape (B,num_anchors,1,1,1)
            relatvie_label = box_label.gather(-2, inds).unsqueeze(-1)
            ori_relatvie_label = ori_box_label.gather(-2, inds).unsqueeze(-1)
            # with shape (B,num_anchors,1,1,1)
            stride_w = strides_w.gather(-2, inds).unsqueeze(-1)
            stride_h = strides_h.gather(-2, inds).unsqueeze(-1)
            # with shape (B,num_anchors,num_class,1,1)
            label_mask = (relatvie_label - 1) == label_inds
            hard_label_mask = (-ori_relatvie_label - 1) == label_inds
            grid_x = torch.div(relative_cx, stride_w)
            grid_y = torch.div(relative_cy, stride_h)
            grid_x = torch.floor(
                torch.clamp(grid_x, 0, self.feature_w - 1)
            ).long()
            grid_y = torch.floor(
                torch.clamp(grid_y, 0, self.feature_h - 1)
            ).long()
            # with shape (B,num_anchors,1,H,W)
            position_mask = torch.logical_and(
                torch.logical_and(grid_x == xs, grid_y == ys),
                inst_gt_mask,
            )
            # with shape (B,num_anchors,num_class,H,W)
            label = torch.logical_and(position_mask, label_mask).float()
            # with shape (B,num_anchors,num_class,H,W)
            hard = torch.logical_and(position_mask, hard_label_mask).float()

            # with shape (B,num_anchors,1,1,1)
            relative_x1 = relative_box_xmin.gather(-2, inds).unsqueeze(-1)
            relative_x2 = relative_box_xmax.gather(-2, inds).unsqueeze(-1)
            relative_y1 = relative_box_ymin.gather(-2, inds).unsqueeze(-1)
            relative_y2 = relative_box_ymax.gather(-2, inds).unsqueeze(-1)
            # with shape (B,num_anchors,1,1,1)
            offset_x1 = relative_x1 / stride_w - (grid_x.float() + 0.5)
            offset_x2 = relative_x2 / stride_w - (grid_x.float() + 0.5)
            offset_y1 = relative_y1 / stride_h - (grid_y.float() + 0.5)
            offset_y2 = relative_y2 / stride_h - (grid_y.float() + 0.5)
            # with shape (B,num_anchors,4,H,W)
            offset = torch.cat(
                [offset_x1, offset_y1, offset_x2, offset_y2],
                dim=2,
            ).expand(-1, -1, -1, self.feature_h, self.feature_w)

            # with shape (B,num_anchors,num_class,H,W)
            label_tgt = torch.where(
                label > 0,
                label,
                label_tgt,
            )
            # with shape (B,num_anchors,4,H,W)
            offset_tgt = torch.where(
                (label > 0).any(dim=2, keepdim=True).expand_as(offset),
                offset,
                offset_tgt,
            )
            # with shape (B,num_anchrs,num_class,H,W)
            hard_mask = torch.where(
                hard > 0,
                hard,
                hard_mask,
            )

        return label_tgt, offset_tgt, hard_mask


@OBJECT_REGISTRY.register
class MatchLabelFlankEncoder(nn.Module):
    """RCNN vehicle flank corners label encoder.

    This class encodes gt and matching results to separate
    bbox and class labels.
    Args:
        cls_on_hard: classification on hard label only.
            Default is False.
        use_cls: use loss of classification.
            Default is True.
        reg_on_hard: regression on hard label only.
            Default is False.
        reg_on_inside_only: whether only reg flank corners
            inside of image. Default is False.
    """

    def __init__(
        self,
        cls_on_hard: bool = False,
        use_cls: bool = True,
        reg_on_hard: bool = False,
        reg_on_inside_only: bool = False,
    ):
        super().__init__()
        self.cls_on_hard = cls_on_hard
        self.reg_on_hard = reg_on_hard
        self.reg_on_inside_only = reg_on_inside_only
        self.use_cls = use_cls

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_flanks: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:  # noqa: D205,D400

        # with shape [batch_size, num_instance, 5]
        # 5 item: <x1,y1,x2,y2,bbox_cls>
        matched_gt_boxes = take_row(gt_boxes, match_gt_id)
        # with shape [batch_size, num_instance, num_points, 3]
        matched_gt_flanks = take_row(gt_flanks, match_gt_id)

        # encode classification target
        # [batch_size,num_instance]
        matched_gt_bbox_cls = matched_gt_boxes[..., 4]
        # [batch_size,num_instance,num_corners]
        matched_gt_flank_cls = matched_gt_flanks[..., 2]
        # [batch_size,num_instance,1]
        pos_match = (match_pos_flag > 0)[..., None]
        if self.reg_on_hard:
            reg_label_mask = pos_match * (matched_gt_flank_cls != 0)
        else:
            reg_label_mask = pos_match * (matched_gt_flank_cls > 0)
        if self.reg_on_inside_only:
            reg_label_mask = reg_label_mask * (matched_gt_flank_cls == 1)

        if self.cls_on_hard:
            matched_gt_flank_cls = torch.abs(matched_gt_flank_cls)
        matched_gt_flank_cls[match_pos_flag <= 0] = -1
        matched_gt_flank_cls[matched_gt_bbox_cls == 0] = -1
        pos_mask = torch.all(matched_gt_flank_cls > 0, dim=-1)
        neg_mask = torch.any(matched_gt_flank_cls == 0, dim=-1)
        ign_mask = torch.any(matched_gt_flank_cls < 0, dim=-1)
        cls_label = torch.zeros_like(pos_mask).float()
        cls_label[pos_mask] = 1.0
        cls_label[neg_mask] = 0.0
        cls_label[ign_mask] = -1
        # with shape [batch_size,num_instance]
        if self.use_cls:
            cls_label_mask = (cls_label >= 0).float()
        else:
            cls_label_mask = torch.zeros_like(pos_mask, dtype=torch.float)

        out_dict = OrderedDict(
            cls_label=cls_label,
            cls_label_mask=cls_label_mask,
        )

        # encode the regression target
        # [batch_size,num_instance,1]
        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = torch.split(boxes, 1, dim=-1)
        boxes_cx = (boxes_x1 + boxes_x2) * 0.5
        boxes_width = boxes_x2 - boxes_x1
        boxes_height = boxes_y2 - boxes_y1
        boxes_mask = torch.logical_and(boxes_width > 0, boxes_height > 0)
        # [batch_size,num_instance,num_corners]
        height = matched_gt_flanks[..., 1] - boxes_y1
        height_mask = torch.logical_and(boxes_mask, height > 0)
        height_tgt = torch.where(
            height_mask,
            torch.log(height / boxes_height),
            torch.zeros_like(height),
        )
        horizon_delta = torch.where(
            height_mask,
            (matched_gt_flanks[..., 0] - boxes_cx) / boxes_width,
            torch.zeros_like(matched_gt_flanks[..., 0]),
        )
        # [batch_size,num_instance,num_corners,2]
        reg_label = torch.cat(
            [horizon_delta[..., None], height_tgt[..., None]], dim=-1
        )
        reg_label_mask = torch.logical_and(reg_label_mask, height_mask)
        reg_label_mask = reg_label_mask[..., None]
        dims = [
            int(x / y) for x, y in zip(reg_label.size(), reg_label_mask.size())
        ]
        # with shape [batch_size,num_instance,num_corners,2]
        reg_label_mask = torch.tile(reg_label_mask, dims)

        out_dict.update(
            reg_label=reg_label,
            reg_label_mask=reg_label_mask,
        )

        return out_dict
