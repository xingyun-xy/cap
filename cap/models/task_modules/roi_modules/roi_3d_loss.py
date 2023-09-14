# cloneright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.models.task_modules.real3d.camera3d_loss import (
    alpha2rot_y_simplified,
    compute_box_3d,
    get_alpha_simplified,
)
from cap.registry import OBJECT_REGISTRY

__all__ = ["RCNNSparse3DLoss"]


def undistort_points(points, cam_mat_1, dist_coeffs, cam_mat_2):
    points_np = points.cpu().numpy()
    cam_mat_1_np = cam_mat_1.cpu().numpy()
    dist_coeffs_np = dist_coeffs.cpu().numpy()
    cam_mat_2_np = cam_mat_2.cpu().numpy()
    points_undistorted_np = cv2.undistortPoints(
        points_np, cam_mat_1_np, dist_coeffs_np, None, cam_mat_2_np
    ).reshape(-1, 2)
    points_undistorted = torch.from_numpy(points_undistorted_np).to(
        points.device
    )
    return points_undistorted


def equivalent_focallength(points, eq_fu_mat, eq_fv_mat):
    x = points[:, 0].clip(min=0, max=eq_fu_mat.shape[1] - 1).long()
    y = points[:, 1].clip(min=0, max=eq_fv_mat.shape[0] - 1).long()
    eq_fu = eq_fu_mat[y, x]
    eq_fv = eq_fv_mat[y, x]
    return eq_fu, eq_fv


@OBJECT_REGISTRY.register
class RCNNSparse3DLoss(nn.Module):
    """RCNN sparse 3d loss.

    Args:
        num_classes: number of classes should be predicted.
        proposal_num: number of rpn proposals.
        kps_loss: kps label loss module.
        offset_2d_loss: offset_2d loss module.
        offset_3d_loss: offset_3d loss module.
        depth_loss: depth loss module.
        dim_loss: dimension loss module.
        rot_weight: weight of the rotation loss.
        focal_length_default: the default value of focal length.
        valid_posmatch_iou: whether to match iou loss values
            according to pos_match.
        iou_loss_scale: the scale value of iou loss.
        feat_w: the width of the output feature.
        feat_h: the height of the output feature.
        undistort_depth_uv: whether to undistort depth branch into depth_u/v.
    """

    def __init__(
        self,
        num_classes: int,
        proposal_num: int,
        kps_loss: nn.Module,
        offset_2d_loss: nn.Module,
        offset_3d_loss: nn.Module,
        depth_loss: nn.Module,
        dim_loss: nn.Module,
        rot_weight: float,
        focal_length_default: float,
        valid_posmatch_iou: bool = False,
        iou_loss_scale: float = 1.0,
        feat_w: int = 8,
        feat_h: int = 8,
        undistort_depth_uv: bool = False,
    ):
        super(RCNNSparse3DLoss, self).__init__()
        self.num_classes = num_classes
        self.proposal_num = proposal_num
        self.feat_w = feat_w
        self.feat_h = feat_h
        self.kps_num = 1
        self.rot_weight = rot_weight
        self.valid_posmatch_iou = valid_posmatch_iou
        self.focal_length_default = focal_length_default
        self.iou_grad_scale = iou_loss_scale
        self.undistort_depth_uv = undistort_depth_uv
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.kps_loss = kps_loss
        self.offset_2d_loss = offset_2d_loss
        self.offset_3d_loss = offset_3d_loss
        self.depth_loss = depth_loss
        self.dim_loss = dim_loss

    @autocast(enabled=False)
    def forward(
        self, pred: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ):

        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()

        pre_rot_cp = pred["rot_pred"].clone()
        pre_cls_map_cp = pred["cls_pred"].clone()
        pre_2d_offset_cp = pred["offset_2d_pred"].clone()
        pre_3d_offset_cp = pred["offset_3d_pred"].clone()
        pre_dim_cp = pred["dims_pred"].clone()
        if self.undistort_depth_uv:
            pre_depth_u_cp = pred["depth_u_pred"].clone()
            pre_depth_v_cp = pred["depth_v_pred"].clone()
        else:
            pre_depth_cp = pred["depth_pred"].clone()

        cls_pred = pred["cls_pred"]
        cls_label = targets["kps_cls_label"].view(
            -1, self.kps_num, self.feat_h, self.feat_w
        )
        cls_weight = targets["kps_cls_label_weight"].view(
            -1, self.kps_num, self.feat_h, self.feat_w
        )
        kps_label_loss = self.kps_loss(
            pred=cls_pred,
            target=cls_label,
            weight=cls_weight,
            avg_factor=(cls_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )

        offset_2d_pred = pred["offset_2d_pred"]
        offset_2d_label = targets["kps_2d_offset"].view(
            -1, self.kps_num * 2, self.feat_h, self.feat_w
        )
        offset_2d_weight = targets["kps_2d_offset_weight"].view(
            -1, self.kps_num * 2, self.feat_h, self.feat_w
        )
        offset_2d_loss = self.offset_2d_loss(
            pred=offset_2d_pred,
            target=offset_2d_label,
            weight=offset_2d_weight,
            avg_factor=(offset_2d_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )

        offset_3d_pred = pred["offset_3d_pred"]
        offset_3d_label = targets["kps_3d_offset"].view(
            -1, self.kps_num * 2, self.feat_h, self.feat_w
        )
        offset_3d_weight = targets["kps_3d_offset_weight"].view(
            -1, self.kps_num * 2, self.feat_h, self.feat_w
        )
        offset_3d_loss = self.offset_3d_loss(
            pred=offset_3d_pred,
            target=offset_3d_label,
            weight=offset_3d_weight,
            avg_factor=(offset_3d_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )

        depth_weight = targets["kps_depth_weight"].view(
            -1, self.kps_num, self.feat_h, self.feat_w
        )
        if self.undistort_depth_uv:
            depth_u_pred = pred["depth_u_pred"]
            depth_u_pred = (1.0 / depth_u_pred.sigmoid() + 1e-6) - 1.0
            depth_u_label = targets["kps_depth_u"].view(
                -1, self.kps_num, self.feat_h, self.feat_w
            )
            depth_u_loss = self.depth_loss(
                pred=depth_u_pred,
                target=depth_u_label,
                weight=depth_weight,
                avg_factor=(depth_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
            )
            depth_v_pred = pred["depth_v_pred"]
            depth_v_pred = (1.0 / depth_v_pred.sigmoid() + 1e-6) - 1.0
            depth_v_label = targets["kps_depth_v"].view(
                -1, self.kps_num, self.feat_h, self.feat_w
            )
            depth_v_loss = self.depth_loss(
                pred=depth_v_pred,
                target=depth_v_label,
                weight=depth_weight,
                avg_factor=(depth_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
            )
        else:
            depth_pred = pred["depth_pred"]
            depth_pred = (1.0 / depth_pred.sigmoid() + 1e-6) - 1.0
            depth_label = targets["kps_depth"].view(
                -1, self.kps_num, self.feat_h, self.feat_w
            )
            depth_loss = self.depth_loss(
                pred=depth_pred,
                target=depth_label,
                weight=depth_weight,
                avg_factor=(depth_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
            )

        # (batch_size, proposal_num)
        batch_mask = targets["pos_match"].clone()
        # (batch_size*proposal_num, 1)
        pos_match = targets["pos_match"].view(-1, 1)

        batch_dim_label = targets["dim_loc_r_y"][..., [0, 1, 2]]
        batch_location_label = targets["dim_loc_r_y"][..., [3, 4, 5]]
        batch_rot_y_label = targets["dim_loc_r_y"][..., [6]]

        dim_pred = pred["dims_pred"].view(-1, 3)
        dim_label = batch_dim_label.view(-1, 3)
        dim_loss = self.dim_loss(
            pred=dim_pred,
            target=dim_label,
            weight=pos_match,
            avg_factor=(
                torch.abs((dim_pred - dim_label) * pos_match) > 0
            ).sum()
            + 1e-6,
        )

        batch_rot_pred = pred["rot_pred"].view(-1, self.proposal_num, 2)

        rot_loss = (
            corner_loss_v2(
                pred_rot=batch_rot_pred,
                batch_rot_mask=batch_mask,
                batch_loc=batch_location_label,
                batch_dim=batch_dim_label,
                batch_rot_y=batch_rot_y_label,
            )
            * self.rot_weight
        )

        calib = targets["calib"]
        trans_mat = targets["trans_mat"]
        dist_coeffs = targets["distCoeffs"]
        rois = targets["rois"]
        pre_cls_map_cp = pre_cls_map_cp.view(
            -1, self.proposal_num, 1, self.feat_h, self.feat_w
        )
        pre_2d_offset_cp = pre_2d_offset_cp.view(
            -1, self.proposal_num, 2, self.feat_h, self.feat_w
        )
        pre_3d_offset_cp = pre_3d_offset_cp.view(
            -1, self.proposal_num, 2, self.feat_h, self.feat_w
        )
        pre_dim_cp = pre_dim_cp.view(-1, self.proposal_num, 3)
        pre_rot_cp = pre_rot_cp.view(-1, self.proposal_num, 2)
        if self.undistort_depth_uv:
            batch_eq_fu = targets["eq_fu"]
            batch_eq_fv = targets["eq_fv"]
            pre_depth_u_cp = pre_depth_u_cp.view(
                -1, self.proposal_num, 1, self.feat_h, self.feat_w
            )
            pre_depth_v_cp = pre_depth_v_cp.view(
                -1, self.proposal_num, 1, self.feat_h, self.feat_w
            )
            iou_label = cal_bev_iou_with_eqfl(
                batch_location_label,
                batch_dim_label,
                batch_rot_y_label,
                rois,
                pre_cls_map_cp.detach(),
                pre_2d_offset_cp.detach(),
                pre_3d_offset_cp.detach(),
                pre_depth_u_cp.detach(),
                pre_depth_v_cp.detach(),
                pre_dim_cp.detach(),
                pre_rot_cp.detach(),
                calib,
                trans_mat,
                self.focal_length_default,
                dist_coeffs,
                batch_eq_fu,
                batch_eq_fv,
            ).view(-1, 1)
        else:
            pre_depth_cp = pre_depth_cp.view(
                -1, self.proposal_num, 1, self.feat_h, self.feat_w
            )
            iou_label = cal_bev_iou(
                batch_location_label,
                batch_dim_label,
                batch_rot_y_label,
                rois,
                pre_cls_map_cp.detach(),
                pre_2d_offset_cp.detach(),
                pre_3d_offset_cp.detach(),
                pre_depth_cp.detach(),
                pre_dim_cp.detach(),
                pre_rot_cp.detach(),
                calib,
                trans_mat,
                self.focal_length_default,
                dist_coeffs,
            ).view(-1, 1)
        iou_pred = pred["iou_pred"].view(-1, 1)

        if self.valid_posmatch_iou:
            iou_loss = self.bce_loss(iou_pred, iou_label) * pos_match
        else:
            iou_loss = self.bce_loss(iou_pred, iou_label).flatten() * pos_match

        iou_loss = (
            iou_loss.sum()
            * self.iou_grad_scale
            / ((iou_loss.detach() > 0).sum() + 1e-6)
        )

        if self.undistort_depth_uv:
            output_loss = OrderedDict(
                center_2d_loss=kps_label_loss,
                offset_2d_loss=offset_2d_loss,
                offset_3d_loss=offset_3d_loss,
                depth_u_loss=depth_u_loss,
                depth_v_loss=depth_v_loss,
                dim_loss=dim_loss,
                rot_loss=rot_loss,
                iou_loss=iou_loss,
            )
        else:
            output_loss = OrderedDict(
                center_2d_loss=kps_label_loss,
                offset_2d_loss=offset_2d_loss,
                offset_3d_loss=offset_3d_loss,
                depth_loss=depth_loss,
                dim_loss=dim_loss,
                rot_loss=rot_loss,
                iou_loss=iou_loss,
            )

        return output_loss


def corner_loss_v2(
    pred_rot, batch_rot_mask, batch_loc, batch_dim, batch_rot_y
):
    """Compute 3d box corner loss.

    Args:
        batch_ind: shape (batch, X).
        batch_loc: shape (batch, X, 3).
        batch_dim: shape (batch, X, 3).
        batch_rot_mask: shape (batch, X).
        batch_rot_y: shape (batch, X, 1).
    """

    pred_alpha = get_alpha_simplified(pred_rot)
    mask = batch_rot_mask.unsqueeze(2).expand(pred_alpha.shape)
    pred_alpha = pred_alpha * mask

    pred_rot_y = alpha2rot_y_simplified(pred_alpha, batch_loc)
    pred_rot_y = mask * pred_rot_y
    # Get 3D obj 8 corners
    batch_size = batch_loc.shape[0]
    loc_ = batch_loc.repeat(2, 1, 1)
    rot_ = torch.cat([pred_rot_y, batch_rot_y], dim=0)
    dim_ = batch_dim.repeat(2, 1, 1)
    corners3d = compute_box_3d(dim_, loc_, rot_)
    pred_corners_3d_rot = corners3d[:batch_size, :]
    gt_corners_3d = corners3d[batch_size:, :]

    mask = (
        batch_rot_mask.view(
            batch_rot_mask.shape
            + (
                1,
                1,
            )
        )
        .expand(gt_corners_3d.shape)
        .float()
    )
    rot_loss = torch.abs(pred_corners_3d_rot * mask - gt_corners_3d * mask)
    rot_loss = rot_loss.sum() / torch.max(mask.sum(), torch.tensor(1.0))

    return rot_loss


def affine_transform_nd(pt, t):
    new_pt = torch.cat([pt, torch.ones(pt.shape[0], 1).to(pt.device)], dim=1).T
    new_pt = torch.mm(t, new_pt).T
    return new_pt[:, :2]


def unproject_2d_to_3d_nd(pt_2d, depth, P):
    depth = depth.view(-1, 1)
    z = depth - P[2, 3]
    x = (pt_2d[:, 0:1] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:, 1:2] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
    pt_3d = torch.cat([x, y, z], dim=-1)

    return pt_3d


def cal_bev_iou(
    batch_loc,
    batch_dim,
    batch_rot_y,
    rois,
    pre_cls_map,
    pre_2d_offset,
    pre_3d_offset,
    pre_depth,
    pre_dim,
    pre_rot,
    calib,
    trans_mat,
    focal_length_default,
    dist_coeffs,
):

    batch_size, proposal_num, _, feat_width, feat_height = pre_cls_map.shape
    pre_cls_map = pre_cls_map.view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_2d_x = pre_2d_offset[:, :, :1, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_2d_y = pre_2d_offset[:, :, 1:, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_3d_x = pre_3d_offset[:, :, :1, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_3d_y = pre_3d_offset[:, :, 1:, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    depth = pre_depth.view(batch_size, proposal_num, feat_width * feat_height)
    max_inds = torch.argmax(pre_cls_map, dim=-1)
    max_inds_x = max_inds % feat_width
    max_inds_y = torch.floor(max_inds / feat_width)

    max_offset_2d_x = torch.gather(
        offset_2d_x,
        index=max_inds[..., None].expand(offset_2d_y.shape).long(),
        dim=-1,
    )[..., 0]
    max_offset_2d_y = torch.gather(
        offset_2d_y,
        index=max_inds[..., None].expand(offset_2d_y.shape).long(),
        dim=-1,
    )[..., 0]
    max_offset_3d_x = torch.gather(
        offset_3d_x,
        index=max_inds[..., None].expand(offset_3d_x.shape).long(),
        dim=-1,
    )[..., 0]
    max_offset_3d_y = torch.gather(
        offset_3d_y,
        index=max_inds[..., None].expand(offset_3d_y.shape).long(),
        dim=-1,
    )[..., 0]
    max_depth = torch.gather(
        depth, index=max_inds[..., None].expand(depth.shape).long(), dim=-1
    )[..., 0]

    scales_x = feat_width / (rois[:, :, 2] - rois[:, :, 0] + 1)
    scales_y = feat_height / (rois[:, :, 3] - rois[:, :, 1] + 1)
    offsets_x = rois[:, :, 0]
    offsets_y = rois[:, :, 1]

    pred_2d_x = (max_inds_x + max_offset_2d_x) / scales_x + offsets_x
    pred_2d_y = (max_inds_y + max_offset_2d_y) / scales_y + offsets_y

    center_2d = torch.stack([pred_2d_x, pred_2d_y], dim=-1)
    pad_row = trans_mat.new_tensor([[[0, 0, 1]]]).repeat(batch_size, 1, 1)

    trans_mat = torch.cat([trans_mat, pad_row], dim=1)
    trans_mat_inv = torch.inverse(trans_mat)
    affine_center_2d = center_2d.clone()
    max_depth = 1.0 / (max_depth.sigmoid() + 1e-6) - 1.0
    center_3d = trans_mat.new_zeros((batch_size, proposal_num, 3))
    max_offset_3d = torch.stack([max_offset_3d_x, max_offset_3d_y], dim=-1)
    for batch_i in range(batch_size):
        affine_center_2d[batch_i] = affine_transform_nd(
            center_2d[batch_i], trans_mat_inv[batch_i]
        )
        # undistort
        affine_center_2d[batch_i] = undistort_points(
            affine_center_2d[batch_i],
            calib[batch_i, :3, :3],
            dist_coeffs[batch_i],
            calib[batch_i, :3, :3],
        )

        max_depth[batch_i] = (
            max_depth[batch_i] * calib[batch_i, 0, 0] / focal_length_default
        )
        center_3d[batch_i] = unproject_2d_to_3d_nd(
            affine_center_2d[batch_i], max_depth[batch_i], calib[batch_i]
        )
        max_offset_3d[batch_i] = (
            max_offset_3d[batch_i]
            * calib[batch_i, 0, 0]
            / focal_length_default
        )
        center_3d[batch_i, :, :2] = (
            center_3d[batch_i, :, :2] + max_offset_3d[batch_i]
        )

    locations = center_3d.clone()
    locations[:, :, [1]] += pre_dim[:, :, [1]] / 2
    rot_sin = pre_rot[:, :, [0]]
    rot_cos = pre_rot[:, :, [1]]
    alpha = torch.arctan(rot_sin / (rot_cos + 1e-7))
    cos_pos_idx = rot_cos >= 0
    cos_neg_idx = rot_cos < 0
    alpha = alpha + (cos_neg_idx.float() - cos_pos_idx.float()) * np.pi / 2.0

    theta = torch.arctan(locations[:, :, [0]] / (locations[:, :, [2]] + 1e-7))
    rotation_y = alpha + theta
    mask_0 = rotation_y > np.pi
    mask_1 = rotation_y < -np.pi
    rotation_y = rotation_y + (mask_1.float() - mask_0.float()) * 2 * np.pi
    pred_corners3d = compute_box_3d(pre_dim, locations, rotation_y)
    gt_corners3d = compute_box_3d(batch_dim, batch_loc, batch_rot_y)
    pred_bev = pred_corners3d[:, :, :4, [0, 2]]
    gt_bev = gt_corners3d[:, :, :4, [0, 2]]
    pre_min_x = torch.min(pred_bev[:, :, :, [0]], dim=-2)[0]
    pre_min_y = torch.min(pred_bev[:, :, :, [1]], dim=-2)[0]
    pre_max_x = torch.max(pred_bev[:, :, :, [0]], dim=-2)[0]
    pre_max_y = torch.max(pred_bev[:, :, :, [1]], dim=-2)[0]

    gt_min_x = torch.min(gt_bev[:, :, :, [0]], dim=-2)[0]
    gt_min_y = torch.min(gt_bev[:, :, :, [1]], dim=-2)[0]
    gt_max_x = torch.max(gt_bev[:, :, :, [0]], dim=-2)[0]
    gt_max_y = torch.max(gt_bev[:, :, :, [1]], dim=-2)[0]
    s1 = (pre_max_x - pre_min_x) * (pre_max_y - pre_min_y)
    s2 = (gt_max_x - gt_min_x) * (gt_max_y - gt_min_y)

    w = torch.minimum(pre_max_x, gt_max_x) - torch.maximum(pre_min_x, gt_min_x)
    w = torch.where(w < 0, torch.zeros_like(w), w)
    h = torch.minimum(pre_max_y, gt_max_y) - torch.maximum(pre_min_y, gt_min_y)
    h = torch.where(h < 0, torch.zeros_like(h), h)

    overlap = w * h
    iou = overlap / (s1 + s2 - overlap + 1e-6)

    return iou


def cal_bev_iou_with_eqfl(
    batch_loc,
    batch_dim,
    batch_rot_y,
    rois,
    pre_cls_map,
    pre_2d_offset,
    pre_3d_offset,
    pre_depth_u,
    pre_depth_v,
    pre_dim,
    pre_rot,
    calib,
    trans_mat,
    focal_length_default,
    dist_coeffs,
    batch_eq_fu,
    batch_eq_fv,
):

    batch_size, proposal_num, _, feat_width, feat_height = pre_cls_map.shape
    pre_cls_map = pre_cls_map.view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_2d_x = pre_2d_offset[:, :, :1, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_2d_y = pre_2d_offset[:, :, 1:, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_3d_x = pre_3d_offset[:, :, :1, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    offset_3d_y = pre_3d_offset[:, :, 1:, :, :].view(
        batch_size, proposal_num, feat_width * feat_height
    )
    depth_u = pre_depth_u.view(
        batch_size, proposal_num, feat_width * feat_height
    )  # noqa
    depth_v = pre_depth_v.view(
        batch_size, proposal_num, feat_width * feat_height
    )  # noqa
    max_inds = torch.argmax(pre_cls_map, dim=-1)
    max_inds_x = max_inds % feat_width
    max_inds_y = torch.floor(max_inds / feat_width)

    max_offset_2d_x = torch.gather(
        offset_2d_x,
        index=max_inds[..., None].expand(offset_2d_y.shape).long(),
        dim=-1,
    )[..., 0]
    max_offset_2d_y = torch.gather(
        offset_2d_y,
        index=max_inds[..., None].expand(offset_2d_y.shape).long(),
        dim=-1,
    )[..., 0]
    max_offset_3d_x = torch.gather(
        offset_3d_x,
        index=max_inds[..., None].expand(offset_3d_x.shape).long(),
        dim=-1,
    )[..., 0]
    max_offset_3d_y = torch.gather(
        offset_3d_y,
        index=max_inds[..., None].expand(offset_3d_y.shape).long(),
        dim=-1,
    )[..., 0]
    max_depth_u = torch.gather(
        depth_u, index=max_inds[..., None].expand(depth_u.shape).long(), dim=-1
    )[..., 0]
    max_depth_v = torch.gather(
        depth_v, index=max_inds[..., None].expand(depth_v.shape).long(), dim=-1
    )[..., 0]

    scales_x = feat_width / (rois[:, :, 2] - rois[:, :, 0] + 1)
    scales_y = feat_height / (rois[:, :, 3] - rois[:, :, 1] + 1)
    offsets_x = rois[:, :, 0]
    offsets_y = rois[:, :, 1]

    pred_2d_x = (max_inds_x + max_offset_2d_x) / scales_x + offsets_x
    pred_2d_y = (max_inds_y + max_offset_2d_y) / scales_y + offsets_y

    center_2d = torch.stack([pred_2d_x, pred_2d_y], dim=-1)
    pad_row = trans_mat.new_tensor([[[0, 0, 1]]]).repeat(batch_size, 1, 1)

    trans_mat = torch.cat([trans_mat, pad_row], dim=1)
    trans_mat_inv = torch.inverse(trans_mat)
    affine_center_2d = center_2d.clone()
    max_depth_u = 1.0 / (max_depth_u.sigmoid() + 1e-6) - 1.0
    max_depth_v = 1.0 / (max_depth_v.sigmoid() + 1e-6) - 1.0
    center_3d = trans_mat.new_zeros((batch_size, proposal_num, 3))
    max_offset_3d = torch.stack([max_offset_3d_x, max_offset_3d_y], dim=-1)
    for batch_i in range(batch_size):
        affine_center_2d[batch_i] = affine_transform_nd(
            center_2d[batch_i], trans_mat_inv[batch_i]
        )
        # undistort
        affine_center_2d[batch_i] = undistort_points(
            affine_center_2d[batch_i],
            calib[batch_i, :3, :3],
            dist_coeffs[batch_i],
            calib[batch_i, :3, :3],
        )
        # depth
        eq_fu_mat = batch_eq_fu[batch_i]
        eq_fv_mat = batch_eq_fv[batch_i]
        eq_fu, eq_fv = equivalent_focallength(
            center_2d[batch_i], eq_fu_mat, eq_fv_mat
        )  # noqa
        max_depth = (
            max_depth_u[batch_i] * eq_fu / focal_length_default
            + max_depth_v[batch_i] * eq_fv / focal_length_default
        )  # noqa
        max_depth = max_depth / 2

        center_3d[batch_i] = unproject_2d_to_3d_nd(
            affine_center_2d[batch_i], max_depth, calib[batch_i]
        )
        max_offset_3d[batch_i] = (
            max_offset_3d[batch_i]
            * calib[batch_i, 0, 0]
            / focal_length_default
        )
        center_3d[batch_i, :, :2] = (
            center_3d[batch_i, :, :2] + max_offset_3d[batch_i]
        )

    locations = center_3d.clone()
    locations[:, :, [1]] += pre_dim[:, :, [1]] / 2
    rot_sin = pre_rot[:, :, [0]]
    rot_cos = pre_rot[:, :, [1]]
    alpha = torch.arctan(rot_sin / (rot_cos + 1e-7))
    cos_pos_idx = rot_cos >= 0
    cos_neg_idx = rot_cos < 0
    alpha = alpha + (cos_neg_idx.float() - cos_pos_idx.float()) * np.pi / 2.0

    theta = torch.arctan(locations[:, :, [0]] / (locations[:, :, [2]] + 1e-7))
    rotation_y = alpha + theta
    mask_0 = rotation_y > np.pi
    mask_1 = rotation_y < -np.pi
    rotation_y = rotation_y + (mask_1.float() - mask_0.float()) * 2 * np.pi
    pred_corners3d = compute_box_3d(pre_dim, locations, rotation_y)
    gt_corners3d = compute_box_3d(batch_dim, batch_loc, batch_rot_y)
    pred_bev = pred_corners3d[:, :, :4, [0, 2]]
    gt_bev = gt_corners3d[:, :, :4, [0, 2]]
    pre_min_x = torch.min(pred_bev[:, :, :, [0]], dim=-2)[0]
    pre_min_y = torch.min(pred_bev[:, :, :, [1]], dim=-2)[0]
    pre_max_x = torch.max(pred_bev[:, :, :, [0]], dim=-2)[0]
    pre_max_y = torch.max(pred_bev[:, :, :, [1]], dim=-2)[0]

    gt_min_x = torch.min(gt_bev[:, :, :, [0]], dim=-2)[0]
    gt_min_y = torch.min(gt_bev[:, :, :, [1]], dim=-2)[0]
    gt_max_x = torch.max(gt_bev[:, :, :, [0]], dim=-2)[0]
    gt_max_y = torch.max(gt_bev[:, :, :, [1]], dim=-2)[0]
    s1 = (pre_max_x - pre_min_x) * (pre_max_y - pre_min_y)
    s2 = (gt_max_x - gt_min_x) * (gt_max_y - gt_min_y)

    w = torch.minimum(pre_max_x, gt_max_x) - torch.maximum(pre_min_x, gt_min_x)
    w = torch.where(w < 0, torch.zeros_like(w), w)
    h = torch.minimum(pre_max_y, gt_max_y) - torch.maximum(pre_min_y, gt_min_y)
    h = torch.where(h < 0, torch.zeros_like(h), h)

    overlap = w * h
    iou = overlap / (s1 + s2 - overlap + 1e-6)

    return iou
