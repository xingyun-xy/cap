# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import multi_apply

__all__ = ["AutoAssignLoss", "PosLoss", "NegLoss", "CenterLoss"]

EPS = 1e-12


@OBJECT_REGISTRY.register
class AutoAssignLoss(torch.nn.Module):
    """
    AutoAssignLoss wrapper.

    Args:
        loss_name: The loss name should in pos/neg/center order.
        pos_loss: Positive loss Module.
        neg_loss: Negative loss Module.
        center_loss: Center prior loss Module.

    Returns:
        dict: A dict containing three calculated loss dicts, the key of inside
        loss dict is loss_name.

    Note:
        This class is not universe. Make sure you know this class limit before
        using it.
    """

    def __init__(
        self,
        loss_name: list,
        pos_loss: torch.nn.Module,
        neg_loss: torch.nn.Module,
        center_loss: torch.nn.Module,
    ):
        super(AutoAssignLoss, self).__init__()
        self.loss_name = loss_name
        self.pos_loss = pos_loss
        self.neg_loss = neg_loss
        self.center_loss = center_loss

    @autocast(enabled=False)
    def forward(self, pred: Tuple, target: Tuple[Dict]) -> Dict:
        target = target[0]
        cls_scores = target["cls_scores"]
        objectnesses = target["objectnesses"]
        decoded_bbox_preds_list = target["decoded_bbox_preds_list"]
        decoded_target_preds_list = target["decoded_target_preds_list"]
        gt_labels_list = target["gt_labels_list"]
        ious_list = target["ious_list"]
        center_prior_weight_list = target["center_prior_weight_list"]
        inside_gt_bbox_mask_list = target["inside_gt_bbox_mask_list"]
        label_weights_list = target["label_weights_list"]
        valid_classes_list = target["valid_classes_list"]
        num_pos_gt = target["num_pos_gt"]
        pos_avg_factor, neg_avg_factor = target["avg_factor"]

        decoded_bbox_preds_list = [
            pred.float() for pred in decoded_bbox_preds_list
        ]
        cls_scores = [item.float() for item in cls_scores]
        objectnesses = [item.float() for item in objectnesses]

        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]

        pos_target = {
            "cls_scores": cls_scores,
            "objectnesses": objectnesses,
            "decoded_bbox_preds_list": decoded_bbox_preds_list,
            "decoded_target_preds_list": decoded_target_preds_list,
            "gt_labels_list": gt_labels_list,
            "center_prior_weight_list": center_prior_weight_list,
            "label_weights_list": label_weights_list,
            "valid_classes_list": valid_classes_list,
            "num_pos_gt": num_pos_gt,
            "avg_factor": pos_avg_factor,
        }

        neg_target = {
            "cls_scores": cls_scores,
            "objectnesses": objectnesses,
            "gt_labels_list": gt_labels_list,
            "ious_list": ious_list,
            "label_weights_list": label_weights_list,
            "inside_gt_bbox_mask_list": inside_gt_bbox_mask_list,
            "avg_factor": neg_avg_factor,
        }

        center_target = {
            "gt_labels_list": gt_labels_list,
            "center_prior_weight_list": center_prior_weight_list,
            "label_weights_list": label_weights_list,
            "valid_classes_list": valid_classes_list,
            "objectnesses": objectnesses,
            "num_pos_gt": num_pos_gt,
        }
        res = {}
        # `pred` is in target tuple, so we get pred from target.
        # assume autoassign target is in pos/neg/center order.
        res[self.loss_name[0]] = self.pos_loss(**pos_target)
        res[self.loss_name[1]] = self.neg_loss(**neg_target)
        res[self.loss_name[2]] = self.center_loss(**center_target)
        return res


@OBJECT_REGISTRY.register
class PosLoss(nn.Module):
    """Calculate the positive loss of all points in gt_bboxes.

    Args:
        loss_weight (float): Global weight of loss. Defaults is 1.0.
        eps (float): A small value to avoid zero denominator.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        torch.Tensor: The value of positive loss.
    """

    def __init__(self, reg_loss, loss_weight=1.0, eps=1e-6, reduction="mean"):
        super(PosLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reg_loss = reg_loss
        self.eps = eps
        self.reduction = reduction

    def get_pos_loss_single(
        self,
        cls_score,
        objectness,
        reg_loss,
        gt_labels,
        center_prior_weights,
        label_weights,
        valid_classes,
    ):
        """Calculate the positive loss for a single image.

        Args:
            cls_score (Tensor): All category scores for each point on
                the feature map. The shape is (num_points, num_class).
            objectness (Tensor): Foreground probability of all points,
                has shape (num_points, 1).
            reg_loss (Tensor): The regression loss of each gt_bbox and each
                prediction box, has shape of (num_points, num_gt).
            gt_labels (Tensor): The zeros based gt_labels of all gt
                with shape of (num_gt,).
            center_prior_weights (Tensor): Float tensor with shape
                of (num_points, num_gt). Each value represents
                the center weighting coefficient.

        Returns:
            tuple[Tensor]:

                - pos_loss (Tensor): The positive loss of a single image.
        """
        gt_labels_weight = torch.ones_like(gt_labels).float()
        gt_labels_weight[gt_labels < 0] = 0.0
        gt_labels[gt_labels < 0] = 0
        # p_loc: localization confidence
        p_loc = torch.exp(-reg_loss)
        # p_cls: classification confidence
        p_cls = (cls_score * objectness)[:, gt_labels]
        # p_pos: joint confidence indicator
        p_pos = p_cls * p_loc

        # 3 is a hyper-parameter to control the contributions of high and
        # low confidence locations towards positive losses.
        confidence_weight = torch.exp(p_pos * 3)
        label_weights = label_weights[:, valid_classes]
        label_weights = label_weights.expand_as(confidence_weight)
        p_pos_weight = (
            confidence_weight * center_prior_weights * label_weights
        ) / (
            (confidence_weight * center_prior_weights * label_weights).sum(
                0, keepdim=True
            )
        ).clamp(
            min=EPS
        )
        reweighted_p_pos = (p_pos * p_pos_weight).sum(0)
        gt_labels_weight[reweighted_p_pos == 0] = 0.0
        pos_loss = F.binary_cross_entropy(
            reweighted_p_pos,
            torch.ones_like(reweighted_p_pos),
            weight=gt_labels_weight,
            reduction="none",
        )
        pos_loss = pos_loss.sum() * self.loss_weight
        return pos_loss

    @autocast(enabled=False)
    def forward(
        self,
        cls_scores,
        objectnesses,
        decoded_bbox_preds_list,
        decoded_target_preds_list,
        gt_labels_list,
        center_prior_weight_list,
        label_weights_list,
        valid_classes_list,
        num_pos_gt,
        avg_factor,
    ):
        if num_pos_gt == 0:
            pos_loss = decoded_bbox_preds_list[0].sum() * 0
        else:
            temp_num_gt_list = [len(item) for item in gt_labels_list]
            num_points = cls_scores[0].shape[0]
            reg_loss_list = []
            for decoded_bbox_preds, decoded_target_preds, temp_num_gt in zip(
                decoded_bbox_preds_list,
                decoded_target_preds_list,
                temp_num_gt_list,
            ):
                loss_bbox = self.reg_loss(
                    decoded_bbox_preds,
                    decoded_target_preds,
                )
                reg_loss_list.append(
                    loss_bbox[self.reg_loss.loss_name].reshape(
                        num_points, temp_num_gt
                    )
                )

            pos_loss_list = multi_apply(
                self.get_pos_loss_single,
                cls_scores,
                objectnesses,
                reg_loss_list,
                gt_labels_list,
                center_prior_weight_list,
                label_weights_list,
                valid_classes_list,
            )
            pos_loss = sum(pos_loss_list) / avg_factor

        return pos_loss


@OBJECT_REGISTRY.register
class NegLoss(nn.Module):
    """Calculate the negative loss.

    Args:
        loss_weight (float): Global weight of loss. Defaults is 1.0.
        eps (float): A small value to avoid zero denominator.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        torch.Tensor: The value of negative loss.
    """

    def __init__(self, loss_weight=1.0, eps=1e-6, reduction="mean"):
        super(NegLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction

    def get_neg_loss_single(
        self,
        cls_score,
        objectness,
        gt_labels,
        ious,
        label_weights,
        inside_gt_bbox_mask,
    ):
        """Calculate the negative loss for a single image.

        Args:
            cls_score (Tensor): All category scores for each point on
                the feature map. The shape is (num_points, num_class).
            objectness (Tensor): Foreground probability of all points
                and is shape of (num_points, 1).
            gt_labels (Tensor): The zeros based label of all gt with shape of
                (num_gt).
            ious (Tensor): Float tensor with shape of (num_points, num_gt).
                Each value represent the iou of pred_bbox and gt_bboxes.
            inside_gt_bbox_mask (Tensor): Tensor of bool type,
                with shape of (num_points, num_gt), each
                value is used to mark whether this point falls
                within a certain gt.

        Returns:
            tuple[Tensor]:

                - neg_loss (Tensor): The negative loss of a single image.
        """
        num_gts = len(gt_labels)
        joint_conf = cls_score * objectness
        p_neg_weight = torch.ones_like(joint_conf)
        if num_gts > 0:
            # the order of dinmension would affect the value of
            # p_neg_weight, we strictly follow the original
            # implementation.
            inside_gt_bbox_mask = inside_gt_bbox_mask.permute(1, 0)
            ious = ious.permute(1, 0)

            foreground_idxs = torch.nonzero(inside_gt_bbox_mask, as_tuple=True)
            temp_weight = (1 / (1 - ious[foreground_idxs]).clamp_(EPS)).float()

            def normalize(x):
                return (x - x.min() + EPS) / (x.max() - x.min() + EPS)

            for instance_idx in range(num_gts):
                idxs = foreground_idxs[0] == instance_idx
                if idxs.any():
                    temp_weight[idxs] = normalize(temp_weight[idxs])

            # Convert to the same type to avoid errors in AMP mode
            p_neg_weight[foreground_idxs[1], gt_labels[foreground_idxs[0]]] = (
                1 - temp_weight
            ).to(p_neg_weight.dtype)

        logits = joint_conf * p_neg_weight
        neg_loss = logits ** 2 * F.binary_cross_entropy(
            logits,
            torch.zeros_like(logits),
            weight=label_weights,
            reduction="none",
        )
        neg_loss = neg_loss.sum() * self.loss_weight
        return neg_loss

    @autocast(enabled=False)
    def forward(
        self,
        cls_scores,
        objectnesses,
        gt_labels_list,
        ious_list,
        label_weights_list,
        inside_gt_bbox_mask_list,
        avg_factor,
    ):

        neg_loss_list = multi_apply(
            self.get_neg_loss_single,
            cls_scores,
            objectnesses,
            gt_labels_list,
            ious_list,
            label_weights_list,
            inside_gt_bbox_mask_list,
        )

        neg_loss = sum(neg_loss_list) / avg_factor

        return neg_loss


@OBJECT_REGISTRY.register
class CenterLoss(nn.Module):
    """Calculate the center prior loss.

    Args:
        loss_weight (float): Global weight of loss. Defaults is 1.0.
        eps (float): A small value to avoid zero denominator.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        torch.Tensor: The value of center prior loss.
    """

    def __init__(self, loss_weight=1.0, eps=1e-12, reduction="mean"):
        super(CenterLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction

    @autocast(enabled=False)
    def forward(
        self,
        gt_labels_list,
        center_prior_weight_list,
        label_weights_list,
        valid_classes_list,
        objectnesses,
        num_pos_gt,
    ):
        if num_pos_gt == 0:
            center_loss = objectnesses[0].sum() * 0
        else:
            img_num = len(gt_labels_list)
            center_loss = []
            for i in range(img_num):
                label_weight = label_weights_list[i]
                valid_classes = valid_classes_list[i]
                label_weight = label_weight[:, valid_classes].expand_as(
                    center_prior_weight_list[i]
                )
                valid_center_prior_weight_list = (
                    center_prior_weight_list[i] * label_weight
                )
                gt_normal_num = (
                    valid_center_prior_weight_list.sum(0) > 0
                ).sum()
                if gt_normal_num > 0:
                    center_loss.append(
                        gt_normal_num
                        / valid_center_prior_weight_list.sum().clamp_(
                            min=self.eps
                        )
                    )
                # when width or height of gt_bbox is smaller than stride of p3
                else:
                    center_loss.append(center_prior_weight_list[i].sum() * 0)
            center_loss = torch.stack(center_loss).mean() * self.loss_weight

        return center_loss
