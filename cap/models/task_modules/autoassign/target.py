# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection

import itertools
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.core.box_utils import bbox_overlaps
from cap.models.task_modules.fcos.target import distance2bbox, get_points
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import multi_apply

__all__ = ["AutoAssignTarget", "CenterPrior"]


@OBJECT_REGISTRY.register
class CenterPrior(nn.Module):
    """Center Weighting module to adjust the category-specific prior \
    distributions.

    Args:
        force_topk (bool): When no point falls into gt_bbox, forcibly
            select the k points closest to the center to calculate
            the center prior. Defaults to False.
        topk (int): The number of points used to calculate the
            center prior when no point falls in gt_bbox. Only work when
            force_topk if True. Defaults to 9.
        num_classes (int): The class number of dataset. Defaults to 80.
        strides (tuple[int]): The stride of each input feature map. Defaults
            to (8, 16, 32, 64, 128).
    """

    def __init__(
            self,
            force_topk=False,
            topk=9,
            num_classes=80,
            strides=(8, 16, 32, 64, 128),
    ):
        super(CenterPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
        self.strides = strides
        self.force_topk = force_topk
        self.topk = topk

    def forward(self, anchor_points_list, gt_bboxes, labels,
                inside_gt_bbox_mask):
        """Get the center prior of each point on the feature map for each
        instance.

        Args:
            anchor_points_list (list[Tensor]): list of coordinate
                of points on feature map. Each with shape
                (num_points, 2).
            gt_bboxes (Tensor): The gt_bboxes with shape of
                (num_gt, 4).
            labels (Tensor): The gt_labels with shape of (num_gt).
            inside_gt_bbox_mask (Tensor): Tensor of bool type,
                with shape of (num_points, num_gt), each
                value is used to mark whether this point falls
                within a certain gt.

        Returns:
            tuple(Tensor):

                - center_prior_weights(Tensor): Float tensor with shape
                    of (num_points, num_gt). Each value represents
                    the center weighting coefficient.
                - inside_gt_bbox_mask (Tensor): Tensor of bool type,
                    with shape of (num_points, num_gt), each
                    value is used to mark whether this point falls
                    within a certain gt or is the topk nearest points for
                    a specific gt_bbox.
        """
        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(labels)
        num_points = sum([len(item) for item in anchor_points_list])
        if num_gts == 0:
            return (
                gt_bboxes.new_zeros(num_points, num_gts),
                inside_gt_bbox_mask,
            )
        center_prior_list = []
        for slvl_points, stride in zip(anchor_points_list, self.strides):
            # slvl_points: points from single level in FPN, has shape (h*w, 2)
            # single_level_points has shape (h*w, num_gt, 2)
            single_level_points = slvl_points[:, None, :].expand(
                (slvl_points.size(0), len(gt_bboxes), 2))
            gt_center_x = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
            gt_center_y = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None]
            device = single_level_points.device
            # instance_center has shape (1, num_gt, 2)
            instance_center = self.mean[labels][None].to(device)
            # instance_sigma has shape (1, num_gt, 2)
            instance_sigma = self.sigma[labels][None].to(device)
            # distance has shape (num_points, num_gt, 2)
            distance = ((single_level_points - gt_center) / float(stride) -
                        instance_center)**2
            center_prior = torch.exp(-distance /
                                     (2 * instance_sigma**2)).prod(dim=-1)
            center_prior_list.append(center_prior)
        center_prior_weights = torch.cat(center_prior_list, dim=0)

        if self.force_topk:
            gt_inds_no_points_inside = torch.nonzero(
                inside_gt_bbox_mask.sum(0) == 0).reshape(-1)
            if gt_inds_no_points_inside.numel():
                topk_center_index = center_prior_weights[:,
                                                         gt_inds_no_points_inside].topk(
                                                             self.topk,
                                                             dim=0)[1]
                temp_mask = inside_gt_bbox_mask[:, gt_inds_no_points_inside]
                inside_gt_bbox_mask[:,
                                    gt_inds_no_points_inside] = torch.scatter(
                                        temp_mask,
                                        dim=0,
                                        index=topk_center_index,
                                        src=torch.ones_like(topk_center_index,
                                                            dtype=torch.bool),
                                    )

        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights, inside_gt_bbox_mask


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@OBJECT_REGISTRY.register
class AutoAssignTarget(object):
    """Generate pos, neg and center target for AutoAssign in training stage.

    Args:
        strides (Sequence[int]): Strides of points in multiple feature levels.
        center_prior (dict): center prior Modu.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides.
        max_avg_num_gt (int): Memory usage of auto assign has direct relation
            to number of gt. To avoid cuda oom with extreme data which has
            hundreds of gt, restrict maximum average number of gt to
            'max_avg_num_gt'. According to statistics of training dataset and
            experiments, 16 to 24 is recommended. if None, no restrict applied.
        cls_out_channels (int): Out_channels of cls_score.
        task_batch_list ([int, int]): two datasets use same head, so we
            generate mask.
    """

    def __init__(
        self,
        strides,
        norm_on_bbox,
        center_prior,
        cls_out_channels,
        max_avg_num_gt=None,
        task_batch_list=None,
    ):
        self.strides = strides
        self.norm_on_bbox = norm_on_bbox
        self.task_batch_list = task_batch_list
        self.max_avg_num_gt = max_avg_num_gt
        self.cls_out_channels = cls_out_channels
        self.center_prior = center_prior

    @staticmethod
    def _get_ignore_mask(gt_labels_list, max_avg_num_gt):
        # Currently, the box corresponding to label < 0 indicates that it needs
        # to be ignored. if max_avg_num_gt is not None, random ignore redundant
        # gt to make total number of gt not exceed batch * max_avg_num_gt
        assert (max_avg_num_gt is None
                or max_avg_num_gt > 0), "max_avg_num_gt \
            should be None or large than zero"

        gt_ignore_mask_list = [None] * len(gt_labels_list)
        pruning_num_list = [0] * len(gt_labels_list)
        if max_avg_num_gt is not None:
            valid_num_gt_list = [(gt_labels >= 0).sum()
                                 for gt_labels in gt_labels_list]
            total_valid_num_gt = sum(valid_num_gt_list)
            total_pruning_num = total_valid_num_gt - max_avg_num_gt * len(
                gt_labels_list)
            if total_pruning_num > 0:
                offset_num_list = [
                    valid_num_gt - max_avg_num_gt
                    for valid_num_gt in valid_num_gt_list
                ]
                deficient_num = sum([
                    -offset_num for offset_num in offset_num_list
                    if offset_num < 0
                ])
                redundant_num = sum([
                    offset_num for offset_num in offset_num_list
                    if offset_num > 0
                ])
                ratio = deficient_num / redundant_num.float()
                pruning_index = []
                for index, offset_num in enumerate(offset_num_list):
                    if offset_num > 0:
                        pruning_num_list[index] = offset_num - int(
                            offset_num * ratio)
                        pruning_index.append(index)
                if sum(pruning_num_list) != total_pruning_num:
                    for index in random.sample(
                            pruning_index,
                            sum(pruning_num_list) - total_pruning_num,
                    ):
                        pruning_num_list[index] -= 1
        for ii, (gt_labels, pruning_num) in enumerate(
                zip(gt_labels_list, pruning_num_list)):
            if len(gt_labels) > 0:
                gt_ignore_mask = gt_labels < 0
                if pruning_num > 0:
                    ignore_redundant = random.sample(
                        range(valid_num_gt_list[ii]), pruning_num)
                    gt_tmp = gt_ignore_mask[~gt_ignore_mask]
                    gt_tmp[ignore_redundant] = True
                    gt_ignore_mask[~gt_ignore_mask] = gt_tmp
                gt_ignore_mask_list[ii] = gt_ignore_mask
        return gt_ignore_mask_list

    def get_targets(self, points, gt_bboxes_list, gt_ignore_mask_list,
                    valid_classes_list):
        """Compute regression targets and each point inside or outside gt_bbox \
        in multiple images.

        Args:
            points (list[Tensor]): Points of all fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_ignore_mask_list (list[Tensor]): mask indicate each Ground truth
                is ignored or not
            valid_classes_list (list[list[int]]): valid_classes of each image

        Returns:
            tuple(list[Tensor]):

                - inside_gt_bbox_mask_list (list[Tensor]): Each
                  Tensor is with bool type and shape of
                  (num_points, num_gt), each value
                  is used to mark whether this point falls
                  within a certain gt.
                - bbox_targets_list (list[Tensor]): BBox
                  targets of each image. Each tensor has shape
                  (num_points, num_gt, 4).
                - label_weights_list (list[Tensor]): Label weights
                  of each image. Each tensor has shape
                  (num_points, num_classes).
        """

        concat_points = torch.cat(points, dim=0)
        # the number of points per img, per lvl
        (
            inside_gt_bbox_mask_list,
            bbox_targets_list,
            label_weights_list,
        ) = multi_apply(  # noqa
            self._get_target_single,
            gt_bboxes_list,
            gt_ignore_mask_list,
            valid_classes_list,
            points=concat_points,
        )
        return inside_gt_bbox_mask_list, bbox_targets_list, label_weights_list

    def _get_target_single(self, gt_bboxes, gt_ignore_mask, valid_class,
                           points):
        """Compute regression targets and each point inside or outside gt_bbox \
        for a single image.

        Args:
            gt_bboxes (Tensor): gt_bbox of single image, has shape
                (num_gt, 4).
            gt_ignore_mask (Tensor): mask of single image's gt, True means
                ignored, False means not ignored, has shape (num_gt).
            valid_class (list[int]): valid class of current image.
            points (Tensor): Points of all fpn level, has shape
                (num_points, 2).

        Returns:
            tuple[Tensor]: Containing the following Tensors:

                - inside_gt_bbox_mask (Tensor): Bool tensor with shape
                  (num_points, num_gt), each value is used to mark
                  whether this point falls within a certain gt.
                - bbox_targets (Tensor): BBox targets of each points with
                  each gt_bboxes, has shape (num_points, num_gt, 4).
                - label_weights (Tensor): Label weight of each points, has
                  shape (num_points, num_classes).
        """

        num_points = points.size(0)
        label_weights = gt_bboxes.new_zeros(
            (num_points, self.cls_out_channels), dtype=torch.float)
        label_weights[:, valid_class] = 1.0
        if gt_ignore_mask is None:
            inside_gt_bbox_mask = gt_bboxes.new_zeros((num_points, 0),
                                                      dtype=torch.bool)  # noqa
            bbox_targets = gt_bboxes.new_zeros((num_points, 0, 4))

            return inside_gt_bbox_mask, bbox_targets, label_weights

        num_gts = (~gt_ignore_mask).sum()
        num_ignore = (gt_ignore_mask).sum()
        if num_gts == 0:
            inside_gt_bbox_mask = gt_bboxes.new_zeros((num_points, 0),
                                                      dtype=torch.bool)  # noqa
            bbox_targets = gt_bboxes.new_zeros((num_points, 0, 4))
        else:
            gt_bboxes_ = gt_bboxes[~gt_ignore_mask][None].expand(
                num_points, num_gts, 4)
            xs, ys = points[:, 0], points[:, 1]
            xs = xs[:, None]
            ys = ys[:, None]
            left = xs - gt_bboxes_[..., 0]
            right = gt_bboxes_[..., 2] - xs
            top = ys - gt_bboxes_[..., 1]
            bottom = gt_bboxes_[..., 3] - ys
            bbox_targets = torch.stack((left, top, right, bottom), -1)
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        if num_ignore != 0:
            gt_bboxes_ignore = gt_bboxes[gt_ignore_mask][None].expand(
                num_points, num_ignore, 4)
            xs_ignore = points[:, 0][:, None].expand(num_points, num_ignore)
            ys_ignore = points[:, 1][:, None].expand(num_points, num_ignore)

            left = xs_ignore - gt_bboxes_ignore[..., 0]
            right = gt_bboxes_ignore[..., 2] - xs_ignore
            top = ys_ignore - gt_bboxes_ignore[..., 1]
            bottom = gt_bboxes_ignore[..., 3] - ys_ignore
            ignore_bbox_targets = torch.stack((left, top, right, bottom), -1)
            inside_ignore_bbox_mask = ignore_bbox_targets.min(-1)[0] > 0
            inside_ignore, _ = inside_ignore_bbox_mask.max(dim=1)
            label_weights[inside_ignore == 1] = 0.0

        return inside_gt_bbox_mask, bbox_targets, label_weights

    @autocast(enabled=False)
    def __call__(self, label, pred, *args):
        assert len(pred) == 3
        if isinstance(pred, dict):
            # assert order is cls_scores, bbox_preds, centernesses
            assert isinstance(pred, OrderedDict)
            cls_scores, bbox_preds, centernesses = pred.values()
        else:
            assert isinstance(pred, (list, tuple))
            cls_scores, bbox_preds, centernesses = pred

        cls_scores = [score.float() for score in cls_scores]
        bbox_preds = [pred.float() for pred in bbox_preds]
        centernesses = [centerness.float() for centerness in centernesses]

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        gt_bboxes_list = label["gt_bboxes"]
        gt_labels_list = label["gt_classes"]
        all_num_gt = sum([len(item) for item in gt_bboxes_list])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        assert len(all_level_points) == len(self.strides)
        num_levels = len(all_level_points)
        num_points = [center.size(0) for center in all_level_points]
        num_imgs = len(gt_bboxes_list)

        gt_ignore_mask_list = self._get_ignore_mask(gt_labels_list,
                                                    self.max_avg_num_gt)
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
        valid_classes_list = (valid_classes_list if valid_classes_list
                              is not None else [[0]] * num_imgs)
        (
            inside_gt_bbox_mask_list,
            bbox_targets_list,
            label_weights_list,
        ) = self.get_targets(  # noqa
            all_level_points,
            gt_bboxes_list,
            gt_ignore_mask_list,
            valid_classes_list,
        )
        tmp_gt_labels_list = []
        tmp_gt_bboxes_list = []
        for gt_label, gt_bboxe, gt_ignore_mask in zip(gt_labels_list,
                                                      gt_bboxes_list,
                                                      gt_ignore_mask_list):
            if gt_ignore_mask is None:
                tmp_gt_labels_list.append(gt_label)
                tmp_gt_bboxes_list.append(gt_bboxe)
            else:
                tmp_gt_labels_list.append(gt_label[~gt_ignore_mask])
                tmp_gt_bboxes_list.append(gt_bboxe[~gt_ignore_mask])
        gt_labels_list = tmp_gt_labels_list
        gt_bboxes_list = tmp_gt_bboxes_list
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        if self.norm_on_bbox:
            tmp_bbox_targets_list = []
            for bbox_targets in bbox_targets_list:
                bbox_targets = list(bbox_targets)
                for i in range(num_levels):
                    bbox_targets[i] = bbox_targets[i] / self.strides[i]
                tmp_bbox_targets_list.append(bbox_targets)
            bbox_targets_list = tmp_bbox_targets_list
        bbox_targets_list = [
            torch.cat(bbox_targets) for bbox_targets in bbox_targets_list
        ]

        num_ignores = 0
        for item in gt_ignore_mask_list:
            if item is not None:
                num_ignores = num_ignores + item.sum()

        num_pos_gt = all_num_gt - num_ignores

        center_prior_weight_list = []
        temp_inside_gt_bbox_mask_list = []
        for gt_bboxe, gt_label, inside_gt_bbox_mask in zip(
                gt_bboxes_list, gt_labels_list, inside_gt_bbox_mask_list):
            center_prior_weight, inside_gt_bbox_mask = self.center_prior(
                all_level_points, gt_bboxe, gt_label, inside_gt_bbox_mask)
            center_prior_weight_list.append(center_prior_weight)
            temp_inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
        inside_gt_bbox_mask_list = temp_inside_gt_bbox_mask_list

        mlvl_points = torch.cat(all_level_points, dim=0)
        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        centernesses = levels_to_images(centernesses)

        ious_list = []
        decoded_target_preds_list = []
        decoded_bbox_preds_list = []
        num_points = len(mlvl_points)
        for bbox_pred, gt_bboxe, inside_gt_bbox_mask in zip(
                bbox_preds, bbox_targets_list, inside_gt_bbox_mask_list):
            temp_num_gt = gt_bboxe.size(1)
            expand_mlvl_points = (mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2))
            gt_bboxe = gt_bboxe.reshape(-1, 4)
            expand_bbox_pred = (bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 4).reshape(-1, 4))
            decoded_bbox_preds = distance2bbox(expand_mlvl_points,
                                               expand_bbox_pred)
            decoded_target_preds = distance2bbox(expand_mlvl_points, gt_bboxe)
            decoded_target_preds_list.append(decoded_target_preds)
            decoded_bbox_preds_list.append(decoded_bbox_preds)

            ious = bbox_overlaps(
                decoded_bbox_preds.detach(),
                decoded_target_preds,
                is_aligned=True,
            )
            ious = ious.reshape(num_points, temp_num_gt)
            if temp_num_gt:
                ious = ious.max(dim=-1,
                                keepdim=True).values.repeat(1, temp_num_gt)
            else:
                ious = ious.new_zeros(num_points, temp_num_gt)
            ious[~inside_gt_bbox_mask] = 0
            ious_list.append(ious)

        pos_avg_factor = bbox_pred.new_tensor(num_pos_gt).clamp_(min=1)
        neg_avg_factor = sum(
            (item * label_weight[:, valid_classes]).data.sum()
            for item, label_weight, valid_classes in zip(
                center_prior_weight_list,
                label_weights_list,
                valid_classes_list,
            )).clamp_(min=1)  # noqa

        target = {
            "cls_scores": cls_scores,
            "objectnesses": centernesses,
            "decoded_bbox_preds_list": decoded_bbox_preds_list,
            "decoded_target_preds_list": decoded_target_preds_list,
            "gt_labels_list": gt_labels_list,
            "ious_list": ious_list,
            "center_prior_weight_list": center_prior_weight_list,
            "inside_gt_bbox_mask_list": inside_gt_bbox_mask_list,
            "label_weights_list": label_weights_list,
            "valid_classes_list": valid_classes_list,
            "num_pos_gt": num_pos_gt,
            "avg_factor": [pos_avg_factor, neg_avg_factor],
        }
        return (target, )
