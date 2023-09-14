# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection

import itertools
from collections import OrderedDict

import torch
from torch import nn

from cap.core.box_utils import bbox_overlaps
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import multi_apply

__all__ = ["FCOSTarget", "get_points", "distance2bbox"]

INF = 1e8


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (torch.Tensor): Shape (n, 2), [x, y].
        distance (torch.Tensor): Distance from the given point to 4 boundaries
            (left, top, right, bottom).
        max_shape (tuple, optional): Shape of the image, used to clamp decoded
            bbox in max_shape range.

    Returns:
        torch.Tensor: Decoded bbox, with shape (n, 4).

    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def get_points(feat_sizes, strides, dtype, device, flatten=False):
    """Generate points according to feat_sizes.

    Args:
        feat_sizes (list[tuple]): Multi-level feature map sizes, the value is
            the HW of a certain layer.
        dtype (torch.dtype): Type of points should be.
        device (torch.device): Device of points should be.
        flatten (bool): Whether to flatten 2D coordinates into 1D dimension.

    Returns:
        list[torch.Tensor]: Points of multiple levels belong to each image,
            the value in mlvl_points is [Tensor(H1W1, 2), Tensor(H2W2, 2), ...]
    """

    def _get_points_single(feat_size, stride):
        """Get points of a single scale level.

        Args:
            feat_size (tuple): Shape of feature map, (h, w).
            stride (int): The stride corresponding to the feature map.

        Returns:
            torch.Tensor: Points of a single scale level, with shape (HW, 2).
        """
        h, w = feat_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        points = (
            torch.stack(
                (x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1
            )
            + stride // 2
        )
        return points

    mlvl_points = []
    for i in range(len(feat_sizes)):
        mlvl_points.append(_get_points_single(feat_sizes[i], strides[i]))
    return mlvl_points


@OBJECT_REGISTRY.register
class FCOSTarget(object):
    """Generate cls and reg targets for FCOS in training stage.

    Args:
        strides (Sequence[int]): Strides of points in multiple feature levels.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        cls_out_channels (int): Out_channels of cls_score.
        background_label (int): Label ID of background, set as num_classes.
        center_sampling (bool): If true, use center sampling.
        center_sample_radius: Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides.
        use_iou_replace_ctrness (bool): If true, use iou as box quality
            assessment method, else use ctrness. Default: false.
        task_batch_list ([int, int]): two datasets use same head, so we
            generate mask
    """

    def __init__(
        self,
        strides,
        regress_ranges,
        cls_out_channels,
        background_label,
        norm_on_bbox=True,
        center_sampling=True,
        center_sample_radius=1.5,
        use_iou_replace_ctrness=False,
        task_batch_list=None,
    ):
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.cls_out_channels = cls_out_channels
        self.background_label = background_label
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.use_iou_replace_ctrness = use_iou_replace_ctrness
        self.norm_on_bbox = norm_on_bbox
        self.task_batch_list = task_batch_list

    @staticmethod
    def _get_ignore_bboxes(gt_bboxes_list, gt_labels_list):
        # Currently, the box corresponding to label <0 indicates that it needs
        # to be ignored
        gt_bboxes_ignore_list = [None] * len(gt_bboxes_list)
        for ii, (gt_bboxes, gt_labels) in enumerate(
            zip(gt_bboxes_list, gt_labels_list)
        ):
            if gt_bboxes.shape[0] > 0:
                gt_bboxes_ignore = gt_bboxes[gt_labels < 0]
                if len(gt_bboxes_ignore.shape) == 1:
                    gt_bboxes_ignore = gt_bboxes_ignore.unsqueeze(0)
                gt_bboxes_ignore_list[ii] = gt_bboxes_ignore
        return gt_bboxes_ignore_list

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore,
        points,
        regress_ranges,
        num_points_per_lvl,
        background_label,
    ):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        # If the gt label is full of -1, do not calculate the image loss
        num_pos_gts = sum(gt_labels != -1)
        num_ignore = 0
        if gt_bboxes_ignore is not None:
            num_ignore = gt_bboxes_ignore.size(0)
        if num_pos_gts == 0:
            return (
                gt_labels.new_full((num_points,), background_label),
                gt_bboxes.new_zeros((num_points, 4)),
                gt_bboxes.new_zeros((num_points,)),
            )

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]
        )
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2
        )
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_ignore != 0:
            gt_bboxes_ignore = gt_bboxes_ignore[None].expand(
                num_points, num_ignore, 4
            )
            xs_ignore = points[:, 0][:, None].expand(num_points, num_ignore)
            ys_ignore = points[:, 1][:, None].expand(num_points, num_ignore)

            left = xs_ignore - gt_bboxes_ignore[..., 0]
            right = gt_bboxes_ignore[..., 2] - xs_ignore
            top = ys_ignore - gt_bboxes_ignore[..., 1]
            bottom = gt_bboxes_ignore[..., 3] - ys_ignore
            ignore_bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(
                x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0]
            )
            center_gts[..., 1] = torch.where(
                y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1]
            )
            center_gts[..., 2] = torch.where(
                x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs
            )
            center_gts[..., 3] = torch.where(
                y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs
            )

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
            )
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]
        ) & (max_regress_distance <= regress_ranges[..., 1])

        # If gt_bboxes_ignore is not none, condition: limit the regression
        # range out of gt_bboxes_ignore
        if num_ignore != 0:
            inside_ignore_bbox_mask = ignore_bbox_targets.min(-1)[0] > 0

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        label_weights = labels.new_ones(labels.shape[0], dtype=torch.float)
        if num_ignore != 0:
            inside_ignore, _ = inside_ignore_bbox_mask.max(dim=1)
            label_weights[inside_ignore == 1] = 0.0

        labels[min_area == INF] = background_label  # set as BG
        labels = labels.long()
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets, label_weights

    @staticmethod
    def _centerness_target(pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (torch.Tensor): BBox targets of positive bboxes,
                with shape (num_pos, 4).

        Returns:
            torch.Tensor:
        """
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]
        ) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def __call__(self, label, pred, *args):
        assert len(pred) == 3
        if isinstance(pred, dict):
            # assert order is cls_scores, bbox_preds, centernesses
            assert isinstance(pred, OrderedDict)
            cls_scores, bbox_preds, centernesses = pred.values()
        else:
            assert isinstance(pred, (list, tuple))
            cls_scores, bbox_preds, centernesses = pred

        # flatten and concat head output
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        gt_bboxes_list = label["gt_bboxes"]
        gt_labels_list = label["gt_classes"]
        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        assert len(all_level_points) == len(self.regress_ranges)
        num_levels = len(all_level_points)
        num_imgs = len(gt_bboxes_list)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            all_level_points[i]
            .new_tensor(self.regress_ranges[i])[None]
            .expand_as(all_level_points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, dim=0)

        # the number of points per lvl, all imgs are equal
        num_points = [center.size(0) for center in all_level_points]
        gt_bboxes_ignore_list = self._get_ignore_bboxes(
            gt_bboxes_list, gt_labels_list
        )

        labels_list, bbox_targets_list, label_weights_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
            background_label=self.background_label,
        )

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        label_weights_list = [
            label_weights.split(num_points, 0)
            for label_weights in label_weights_list
        ]  # noqa
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_label_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list])
            )
            concat_lvl_label_weights.append(
                torch.cat(
                    [label_weights[i] for label_weights in label_weights_list]
                )
            )  # noqa
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list]
            )
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        # generate bbox targets and centerness targets of positive points
        flatten_labels = torch.cat(concat_lvl_labels)
        flatten_label_weights = torch.cat(concat_lvl_label_weights)
        flatten_bbox_targets = torch.cat(concat_lvl_bbox_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        )

        pos_inds = (
            (
                (flatten_labels >= 0)
                & (flatten_labels != self.background_label)
                & (flatten_label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )
        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)

        if num_pos == 0:
            # generate fake pos_inds
            pos_inds = flatten_labels.new_zeros((1,))

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]
        # decode pos_bbox_targets to bbox for calculating IOU-like loss
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)

        # fix loss conflict
        points_per_strides = []
        for feat_size in feat_sizes:
            points_per_strides.append(feat_size[0] * feat_size[1])
        valid_classes_list = None
        if self.task_batch_list is not None:
            valid_classes_list = []
            accumulate_list = list(itertools.accumulate(self.task_batch_list))
            task_id = 0
            for ii in range(num_imgs):
                if ii >= accumulate_list[task_id]:
                    task_id += 1
                valid_classes = []
                for cls in label["gt_classes"][ii].unique():
                    if cls >= 0:
                        valid_classes.append(cls.item())
                if len(valid_classes) == 0:
                    valid_classes.append(task_id)
                valid_classes_list.append(valid_classes)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos == 0:
            pos_centerness_targets = torch.tensor([0.0], device=device)
            centerness_weight = torch.tensor([0.0], device=device)  # noqa
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
        else:
            centerness_weight = None
            if not self.use_iou_replace_ctrness:
                pos_centerness_targets = self._centerness_target(
                    pos_bbox_targets
                )  # noqa
                bbox_weight = pos_centerness_targets
                bbox_avg_factor = pos_centerness_targets.sum()
            else:
                pos_centerness_targets = bbox_overlaps(
                    pos_decoded_bbox_preds.detach(),  # noqa
                    pos_decoded_targets,
                    is_aligned=True,
                )
                bbox_weight = None
                bbox_avg_factor = None

        cls_target = {
            "pred": flatten_cls_scores,
            "target": flatten_labels,
            "weight": flatten_label_weights,
            "avg_factor": cls_avg_factor,
            "points_per_strides": points_per_strides,
            "valid_classes_list": valid_classes_list,
        }
        giou_target = {
            "pred": pos_decoded_bbox_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }
        centerness_target = {
            "pred": pos_centerness,
            "target": pos_centerness_targets,
            "weight": centerness_weight,
        }
        return cls_target, giou_target, centerness_target


@OBJECT_REGISTRY.register
class DynamicFcosTarget(nn.Module):
    """Generate cls and reg targets for FCOS in training \
stage base on dynamic losses.

    Args:
        strides (Sequence[int]): Strides of points in multiple feature levels.
        topK(int): Number of postive sample for each ground trouth to keep.
        cls_out_channels (int): Out_channels of cls_score.
        background_label (int): Label ID of background, set as num_classes.
        loss_cls (nn.Module): Loss for cls to choose positive target.
        loss_reg (nn.Module): Loss for reg to choose positive target.
    """

    def __init__(
        self,
        strides,
        topK,
        loss_cls,
        loss_reg,
        cls_out_channels,
        background_label,
    ):
        super(DynamicFcosTarget, self).__init__()
        self.strides = strides
        self.topK = topK
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.cls_out_channels = cls_out_channels
        self.background_label = background_label

    def _get_iou(self, flatten_bbox_preds, bbox_targets, points):
        decoded_targets = distance2bbox(points, bbox_targets)
        decoded_preds = distance2bbox(points, flatten_bbox_preds)
        return bbox_overlaps(decoded_preds, decoded_targets, is_aligned=True)

    def _get_cost(
        self, cls_preds, cls_targets, bbox_preds, bbox_targets, points
    ):

        cls_loss = list(self.loss_cls(cls_preds, cls_targets.long()).values())[
            0
        ]
        cls_loss = cls_loss.sum(2)
        bbox_preds_decoded = distance2bbox(points, bbox_preds)
        bbox_targets_decoded = distance2bbox(points, bbox_targets)
        reg_loss = list(
            self.loss_reg(bbox_preds_decoded, bbox_targets_decoded).values()
        )[0]
        return cls_loss + reg_loss

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        flatten_cls_scores,
        flatten_bbox_preds,
        strides,
        points,
        device,
    ):
        num_gts = gt_bboxes.shape[0]
        if num_gts == 0:
            bbox_targets = torch.zeros_like(flatten_bbox_preds, device=device)
            cls_targets = (
                torch.ones((flatten_cls_scores.shape[0]), device=device)
                * self.background_label
            )
            centerness_targets = torch.zeros(
                (flatten_bbox_preds.shape[0]), device=device
            )
            label_weights = torch.ones_like(
                cls_targets, dtype=torch.float, device=device
            )
            return (
                cls_targets.long(),
                bbox_targets,
                centerness_targets,
                label_weights,
            )

        gt_labels = gt_labels.view(1, -1).repeat(
            flatten_cls_scores.shape[0], 1
        )
        bbox_targets = torch.zeros(
            (flatten_bbox_preds.shape[0], num_gts, 4), device=device
        )

        bbox_targets[..., 0:2] = points[:, None, :] - gt_bboxes[None, :, 0:2]
        bbox_targets[..., 2:4] = gt_bboxes[None, :, 2:4] - points[:, None, :]
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        ignore_labels = gt_labels < 0
        ignore_labels[inside_gt_bbox_mask != 1] = 0

        inside_gt_bbox_mask[ignore_labels] = -1
        bbox_targets /= strides[:, None, :]
        points = points / strides[..., 0:2]
        points = points.view(-1, 1, 2).repeat(1, num_gts, 1)
        flatten_bbox_preds = flatten_bbox_preds.view(-1, 1, 4).repeat(
            1, num_gts, 1
        )
        flatten_cls_scores = flatten_cls_scores.view(
            -1, 1, self.cls_out_channels
        ).repeat(1, num_gts, 1)

        ious = self._get_iou(flatten_bbox_preds, bbox_targets, points)
        ious[inside_gt_bbox_mask != 1] = 0.0

        cost = self._get_cost(
            flatten_cls_scores,
            gt_labels,
            flatten_bbox_preds,
            bbox_targets,
            points,
        )
        cost[inside_gt_bbox_mask != 1] = INF
        cost = cost.permute(1, 0).contiguous()
        n_candidate_k = min(10, ious.size(0))
        topk_ious, _ = torch.topk(ious, n_candidate_k, dim=0)

        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        matching_matrix = torch.zeros_like(
            cost, device=flatten_bbox_preds.device
        )
        for gt_idx in range(num_gts):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        matching_gt = matching_matrix.sum(0)

        if (matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, matching_gt > 1], dim=0)
            matching_matrix[:, matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, matching_gt > 1] = 1.0

        matching_matrix = matching_matrix.permute(1, 0).contiguous()
        matching_matrix[inside_gt_bbox_mask != 1] = 0.0

        gt_labels[matching_matrix == 0] = self.background_label
        argmin = gt_labels.argmin(1).view(-1)

        cls_targets = gt_labels[torch.arange(gt_labels.shape[0]), argmin]

        bbox_targets = bbox_targets[
            torch.arange(bbox_targets.shape[0]), argmin
        ]
        centerness_targets = ious[torch.arange(bbox_targets.shape[0]), argmin]
        if (inside_gt_bbox_mask == -1).sum().cpu().numpy() != 0:
            print((inside_gt_bbox_mask == -1).sum())
        label_weights = torch.ones_like(gt_labels, dtype=torch.float)
        label_weights[inside_gt_bbox_mask == -1] = 0
        label_weights, _ = label_weights.min(-1)
        return cls_targets, bbox_targets, centerness_targets, label_weights

    def forward(self, label, pred, *args):
        cls_scores, bbox_preds, centernesses = pred

        gt_bboxes_list = label["gt_bboxes"]
        gt_labels_list = label["gt_classes"]
        num_imgs = len(gt_bboxes_list)
        # flatten and concat head output
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.cls_out_channels
            )
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for centerness in centernesses
        ]

        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        flatten_strides = [
            torch.tensor([s, s, s, s], device=device)
            .view(-1, 4)
            .repeat(flatten_bbox_pred.shape[1], 1)
            for flatten_bbox_pred, s in zip(flatten_bbox_preds, self.strides)
        ]
        flatten_strides = torch.cat(flatten_strides)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_centerness = torch.cat(flatten_centerness, dim=1)

        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        # expand regress ranges to align with points
        concat_points = torch.cat(all_level_points, dim=0)

        with torch.no_grad():
            (
                cls_targets_list,
                bbox_targets_list,
                centerness_targets_list,
                label_weights_list,
            ) = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                flatten_cls_scores,
                flatten_bbox_preds,
                strides=flatten_strides,
                points=concat_points,
                device=device,
            )
        cls_targets = torch.cat(cls_targets_list)
        bbox_targets = torch.cat(bbox_targets_list)
        centerness_targets = torch.cat(centerness_targets_list)

        label_weights = torch.cat(label_weights_list)

        pos_inds = (
            (
                (cls_targets >= 0)
                & (cls_targets != self.background_label)
                & (label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )

        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)
        if num_pos == 0:
            pos_inds = flatten_cls_scores.new_zeros((1,)).long()
        pos_bbox_targets = bbox_targets[pos_inds]
        if num_pos == 0:
            centerness_weight = torch.tensor([0.0], device=device)  # noqa
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
        else:
            centerness_weight = None
            bbox_weight = FCOSTarget._centerness_target(
                pos_bbox_targets
            )  # noqa
            bbox_avg_factor = bbox_weight.sum()

        flatten_cls_scores = flatten_cls_scores.view(-1, self.cls_out_channels)
        flatten_bbox_preds = flatten_bbox_preds.view(-1, 4)
        flatten_centerness = flatten_centerness.view(-1)

        cls_target = {
            "pred": flatten_cls_scores,
            "target": cls_targets.long(),
            "weight": label_weights,
            "avg_factor": cls_avg_factor,
        }

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_points = torch.zeros((2), device=device)
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)
        pos_decoded_preds = distance2bbox(pos_points, pos_bbox_preds)

        giou_target = {
            "pred": pos_decoded_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }

        pos_centerness = flatten_centerness[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds]
        centerness_target = {
            "pred": pos_centerness,
            "target": pos_centerness_targets,
            "weight": centerness_weight,
        }
        return cls_target, giou_target, centerness_target
