# Copyright (c) Changan Auto. All rights reserved.

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from cap.models.base_modules import DynamicWeight
from cap.registry import OBJECT_REGISTRY

__all__ = ["Real3DLoss"]


DEFAULT_EPS = 1e-5
STAGE_AWARE_EPS = {
    "float": 1e-5,
    "float_freeze_bn": 1e-5,
    "qat": 1e-4,
}
training_step = os.getenv("CAP_TRAINING_STEP", "float")
EPS = STAGE_AWARE_EPS.get(training_step, DEFAULT_EPS)


def sigmoid_and_clip(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def hm_focal_loss(pred, gt, ignore_mask, gamma=2, beta=1, alpha=4):

    pos_mask = (gt == 1).float() * (1.0 - ignore_mask)
    neg_mask = (gt < 1).float() * (1.0 - ignore_mask)

    neg_weights = torch.pow(1 - gt, alpha)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * pos_mask
    neg_loss = beta * torch.log(1 - pred) * torch.pow(pred, gamma) * neg_mask
    neg_loss = neg_loss * neg_weights

    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    num_pos = torch.max(num_pos, torch.ones_like(num_pos))
    loss = -(pos_loss + neg_loss) / num_pos
    return loss


def hm_l1_loss(output, target, weight_mask, ignore_mask, heatmap_type):
    if heatmap_type == "dense":
        weight_mask = weight_mask.gt(0).float()
    elif heatmap_type == "point":
        weight_mask = weight_mask.eq(1).float()

    if ignore_mask is not None:
        weight_mask = weight_mask * (1.0 - ignore_mask)

    loss = (F.l1_loss(output, target, reduction="none") * weight_mask).sum()
    loss = loss / (EPS + weight_mask.sum())
    return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _compute_box_3d(dim, loc, rot_y):
    rot_y = rot_y.unsqueeze(2)
    c, s = torch.cos(rot_y), torch.sin(rot_y)
    zeros, ones = torch.zeros_like(c), torch.ones_like(c)
    R = torch.cat(
        [c, zeros, s, zeros, ones, zeros, -s, zeros, c], dim=-1
    ).unsqueeze(2)

    l, w, h = dim[:, :, [2]], dim[:, :, [1]], dim[:, :, [0]]

    x_corners = torch.cat(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=-1
    ).unsqueeze(3)
    y_corners = torch.cat(
        [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=-1
    ).unsqueeze(3)
    z_corners = torch.cat(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=-1
    ).unsqueeze(3)
    corners = torch.cat([x_corners, y_corners, z_corners], dim=-1)

    corners_3d = R * corners.repeat(1, 1, 1, 3)
    corners_3d = torch.cat(
        [
            torch.sum(corners_3d[:, :, :, :3], dim=-1, keepdim=True),
            torch.sum(corners_3d[:, :, :, 3:6], dim=-1, keepdim=True),
            torch.sum(corners_3d[:, :, :, 6:], dim=-1, keepdim=True),
        ],
        dim=-1,
    )

    corners_3d = corners_3d + loc.unsqueeze(2)

    return corners_3d


def rot_corner_loss(
    output_rot, batch_rot_mask, batch_ind, batch_loc, batch_dim, batch_rot_y
):

    pred_rot = _transpose_and_gather_feat(output_rot, batch_ind.detach())
    pred_alpha = torch.atan(pred_rot[:, :, 0] / (pred_rot[:, :, 1] + EPS))
    cos_pos_mask = pred_rot[:, :, 1] >= 0
    cos_neg_mask = pred_rot[:, :, 1] < 0
    pred_alpha[cos_pos_mask] -= np.pi / 2
    pred_alpha[cos_neg_mask] += np.pi / 2

    mask = batch_rot_mask
    pred_alpha = pred_alpha * mask

    rays = torch.atan(batch_loc[:, :, 0] / (batch_loc[:, :, 2] + EPS))
    pred_rot_y = pred_alpha + rays
    larger_mask = pred_rot_y > np.pi
    small_mask = pred_rot_y < -np.pi
    pred_rot_y[larger_mask] -= 2 * np.pi
    pred_rot_y[small_mask] += 2 * np.pi
    pred_rot_y = mask * pred_rot_y

    pred_box3d = _compute_box_3d(batch_dim, batch_loc, pred_rot_y)
    gt_box3d = _compute_box_3d(batch_dim, batch_loc, batch_rot_y)
    mask = mask[:, :, None, None]
    loss = torch.abs(pred_box3d * mask - gt_box3d * mask)
    loss = loss.mean(dim=(2, 3))
    N = mask.sum()
    loss = loss.sum() / torch.max(N, torch.ones_like(N))

    return loss


def mask_l1_loss(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Calculate L1 loss with mask.

    Args:
        pred (torch.Tensor): predictions
        gt (torch.Tensor): ground truth
        mask (torch.Tensor): mask

    Returns:
        torch.Tensor: l1 loss
    """
    loss = F.l1_loss(pred, gt, reduction="none") * mask
    loss = loss.sum() / max(mask.sum(), 1.0)
    return loss


def angle_multibin_loss(
    pred_cls: torch.Tensor,
    pred_offset: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_offset: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Calculate multiple bin loss.

    Args:
        pred_cls (torch.Tensor): predicted bin class, [b,k,num_bin]
        pred_offset (torch.Tensor): predicted bin offset, [b,k,num_bin*2]
        gt_cls (torch.Tensor): ground truth of bin classes, [b,k,num_bin]
        gt_offset (torch.Tensor): ground truth of bin offsets, [b,k,num_bin]
        mask (torch.Tensor): positive mask

    Returns:
        torch.Tensor: mean loss
    """

    assert pred_cls.ndim - mask.ndim in [0, 1]
    if pred_cls.ndim - mask.ndim == 1:
        mask = mask[..., None]

    b, n, c = pred_cls.shape
    p_cls = pred_cls.sigmoid_()
    p_off = pred_offset.reshape((b, n, -1, 2))
    p_off = F.normalize(p_off, dim=-1)

    # bin cls loss
    cls_loss = F.binary_cross_entropy(p_cls, gt_cls, reduction="none")
    cls_loss = (cls_loss * mask).sum() / max(mask.sum(), 1.0)

    # regression loss
    sin_loss = mask_l1_loss(p_off[..., 0], torch.sin(gt_offset), mask)
    cos_loss = mask_l1_loss(p_off[..., 1], torch.cos(gt_offset), mask)
    return cls_loss + cos_loss + sin_loss


def dep_l1_loss(
    pred,
    target,
    weight_mask,
    ignore_mask=None,
    heatmap_type="weighted",
    max_dep=150,
):

    if heatmap_type == "dense":
        weight_mask = weight_mask.float()
    elif heatmap_type == "point":
        weight_mask = (weight_mask == 1).float()

    if ignore_mask is not None:
        weight_mask = weight_mask * (1.0 - ignore_mask)

    max_dep_mask = target < max_dep
    weight_mask = weight_mask * max_dep_mask

    l1_depth_error = torch.abs(pred - target) * weight_mask
    rel_norm = 0.1 + 1.2 * (((max_dep - target) * max_dep_mask) / max_dep) ** 2
    quad_rel_depth_error = rel_norm * l1_depth_error
    loss = torch.sum(quad_rel_depth_error / (EPS + torch.sum(weight_mask)))

    return loss


@OBJECT_REGISTRY.register
class Real3DLoss(nn.Module):
    def __init__(
        self,
        max_dep=150.0,
        loss_weights=None,
        heatmap_type=None,
        use_dynamic_weight=False,
        init_values=None,
        norm_type="direct",
        use_multibin=False,
        num_multibin=4,
    ):
        """Real3D loss function.

           Classification uses focal loss.
           Orientation uses corner loss.
           Depth uses L1 loss weighted by ground truth depth.
           The other regressions use L1 loss.

        Args:
            max_dep (float): default: 150.0
                Maximum depth value to weigh depth loss.
            loss_weights: dict, default: None
                Global loss weight for each sub-loss. Default weight is 1.0.
            heatmap_type: dict, default: None
                Heatmap type for loss weight used by each regression loss,
                except orientation. Default is 'heatmap'.
                'heatmap': gaussian loss weight.
                'point': only samples in center point will be used,
                        same as CenterNet.
                'dense': all samples use the same loss weight, which is 1.0.
            use_dynamic_weight: bool, default: False
                Whether to use dynamic loss weights.
            init_values: dict, default: None
                Initialization values for dynamic loss weights. If it was
                `None`, we will initialize with `0.0` for every loss.
            norm_type: str, default: `l1`
                Dynamic loss weights normlize method, includes `l1`, `direct`,
                `exp`, `softmax`.
            use_multibin: bool, whether use multi-bin to get rotation.
            num_multibin: int, default: 4
                Number of alpha multi-bin.
        """
        super(Real3DLoss, self).__init__()
        self.max_dep = max_dep
        self.use_dynamic_weight = use_dynamic_weight
        self.norm_type = norm_type
        self.loss_weights = defaultdict(lambda: 1.0)
        self.heatmap_type = defaultdict(lambda: "heatmap")
        self.init_values = defaultdict(lambda: 0.0)
        self.use_multibin = use_multibin
        self.num_multibin = num_multibin

        if loss_weights is not None:
            self.loss_weights.update(**loss_weights)
        if heatmap_type is not None:
            self.heatmap_type.update(**heatmap_type)
        if init_values is not None:
            self.init_values.update(**init_values)

        # Dynamic loss weights initialization
        if self.use_dynamic_weight:
            self.dynamic_weights = nn.ModuleDict()
            for k in self.loss_weights.keys():
                self.dynamic_weights[k] = DynamicWeight(self.init_values[k])

    @autocast(enabled=False)
    def forward(self, pred, target):
        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()

        # K = target["calibration"]
        target = target["target"]
        all_losses = {}
        if "ignore_mask" in target:
            ignore_mask = target["ignore_mask"].float()
        else:
            ignore_mask = None

        hm_weight = self.loss_weights["hm"]
        hm = sigmoid_and_clip(pred["hm"])
        hm_loss = hm_focal_loss(hm, target["hm"], ignore_mask)
        if self.use_dynamic_weight:
            hm_loss = self.dynamic_weights["hm"](hm_loss)
        hm_loss = hm_loss * hm_weight
        all_losses["hm_loss"] = hm_loss

        reg_keys = ["wh", "dim", "loc_offset", "center_offset"]
        for key in reg_keys:
            if key not in pred:
                continue
            weight = self.loss_weights[key]
            loss = hm_l1_loss(
                pred[key],
                target[key],
                target["weight_hm"],
                ignore_mask,
                heatmap_type=self.heatmap_type[key],
            )
            if self.use_dynamic_weight:
                loss = self.dynamic_weights[key](loss)
            loss = loss * weight
            all_losses["{}_loss".format(key)] = loss

        rot = pred["rot"]
        rot_weight = self.loss_weights["rot"]
        if self.use_multibin:
            rot = _transpose_and_gather_feat(rot, target["ind_"])
            rot_loss = angle_multibin_loss(
                rot[..., : self.num_multibin],
                rot[..., self.num_multibin :],
                target["bin_cls_"],
                target["bin_offset_"],
                target["ind_mask_"],
            )
        else:
            ind_mask_, ind_, loc_, dim_, rot_y_ = (
                target["ind_mask_"],
                target["ind_"],
                target["loc_"],
                target["dim_"],
                target["rot_y_"],
            )
            rot_loss = rot_corner_loss(
                rot, ind_mask_, ind_, loc_, dim_, rot_y_
            )
        if self.use_dynamic_weight:
            rot_loss = self.dynamic_weights["rot"](rot_loss)
        rot_loss = rot_loss * rot_weight
        all_losses["rot_loss"] = rot_loss

        dep = pred["dep"]
        dep_weight = self.loss_weights["dep"]
        dep = 1.0 / (dep.sigmoid() + EPS) - 1.0
        dep_loss = dep_l1_loss(
            dep,
            target["dep"],
            target["weight_hm"],
            ignore_mask,
            heatmap_type=self.heatmap_type["dep"],
            max_dep=self.max_dep,
        )
        if self.use_dynamic_weight:
            dep_loss = self.dynamic_weights["dep"](dep_loss)
        dep_loss = dep_loss * dep_weight
        all_losses["dep_loss"] = dep_loss

        # Normlize dynamic loss weights
        if self.use_dynamic_weight:
            if self.norm_type == "softmax":
                sum_weights = sum(
                    [torch.exp(v.scale) for v in self.dynamic_weights.values()]
                )
            for k, v in self.dynamic_weights.items():
                if self.norm_type == "direct":
                    all_losses[k + "_weight_loss"] = v.scale
                elif self.norm_type == "l1":
                    all_losses[k + "_weight_loss"] = torch.abs(v.scale)
                elif self.norm_type == "exp":
                    all_losses[k + "_weight_loss"] = torch.exp(v.scale)
                elif self.norm_type == "softmax":
                    all_losses[k + "_weight_loss"] = (
                        torch.exp(v.scale) / sum_weights
                    )
                else:
                    raise NotImplementedError
        return all_losses
