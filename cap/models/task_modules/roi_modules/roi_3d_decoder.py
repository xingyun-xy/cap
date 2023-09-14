from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp2d

from cap.core.data_struct.base_struct import DetBoxes3D
from cap.core.undistort_lut import getUnDistortPoints
from cap.registry import OBJECT_REGISTRY

__all__ = ["ROI3DDecoder"]

GlobalEQFUMap = defaultdict()
GlobalEQFVMap = defaultdict()


def affine_transform_nd(pt, t):
    new_pt = torch.cat([pt, pt.new_ones(pt.shape[0], 1)], dim=1).T  # noqa
    new_pt = torch.matmul(t, new_pt).T
    return new_pt[:, :2]


def undistort_by_map(affine_center_2d, calib, distCoeffs, im_hw, scale=2.0):

    ii, jj = np.meshgrid(
        np.arange(im_hw[1]), np.arange(im_hw[0]), indexing="ij"
    )

    index = cv2.undistortPoints(
        np.column_stack([ii.flatten(), jj.flatten()])[None] * scale,
        calib[:3, :3],
        distCoeffs,
        None,
        calib[:3, :3],
    )  # noqa
    index = index.reshape(-1, 2) / scale

    map_xy = np.zeros(shape=(im_hw[1], im_hw[0], 2))
    map_xy = index.reshape(im_hw[1], im_hw[0], 2).astype(int)

    center = affine_center_2d.cpu().numpy().astype(int)
    center = np.maximum(np.minimum(center, (im_hw[::-1] - 1)[None]), 0)
    center = map_xy[center[:, 0], center[:, 1]]
    return affine_center_2d.new_tensor(center)


def unproject_2d_to_3d_nd(pt_2d, depth, P):
    # pts_2d: (n, 2)
    # depth: (n, 1)
    # P: 3 x 4
    # return: (n, 3)
    depth = depth.reshape(-1, 1)
    z = depth - P[2, 3]
    x = (pt_2d[:, 0:1] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:, 1:2] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
    pt_3d = torch.cat([x, y, z], dim=-1)

    return pt_3d


def cal_equivalent_focal_length(
    uv_points,
    calib,
    dist,
    img_wh,
    undistort_by_cv=False,
):  # noqa: D205, D400
    """[summary]
    Args:
        uv_points (numpy.NDArray): Distorted uv points,\
        in shape (num_points, 1, 2)
        calib (numpy.NDArray): Calibration mat
        dist (numpy.NDArray): Distort mat
        name (str):  Name of this metric instance for display.
    """
    calib = calib[:3, :3]

    fu = calib[0, 0]
    fv = calib[1, 1]
    f = (fu + fv) / 2
    cu = calib[0, 2]
    cv = calib[1, 2]
    dist_u = uv_points[:, 0, 0]
    dist_v = uv_points[:, 0, 1]

    if undistort_by_cv:
        undist_uv = cv2.undistortPointsIter(
            src=uv_points,
            cameraMatrix=calib,
            distCoeffs=dist,
            R=None,
            P=calib,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER, 4, 0.1),
        )
        undist_u = undist_uv[:, 0, 0]
        undist_v = undist_uv[:, 0, 1]
    else:
        undist_uv = getUnDistortPoints(
            points=uv_points[:, 0, :],
            calib=calib,
            distCoeffs=dist,
            img_wh=img_wh,
        )
        undist_u = undist_uv[:, 0]
        undist_v = undist_uv[:, 1]

    x = (undist_u - cu) / fu
    y = (undist_v - cv) / fv
    e_x = (dist_u - cu) / fu
    e_y = (dist_v - cv) / fv

    r = np.sqrt(x * x + y * y)
    e_r = np.sqrt(e_x * e_x + e_y * e_y)

    e_f = e_r.clip(min=1e-4) * f / r.clip(min=1e-4)

    # image center points is not stable
    e_f[e_r < 1e-3] = f
    e_f = e_f.clip(min=f * 0.1, max=f)

    return e_f


def cal_equivalent_focal_length_uv_mat(
    width,
    height,
    calib,
    dist,
    downsample=4,
    undistort_by_cv=False,
):
    if isinstance(dist, list):
        dist = np.array(dist, dtype=np.float32)
    hash_distcoeffs = dist.tobytes()
    if hash_distcoeffs in GlobalEQFUMap.keys():
        eq_fu = GlobalEQFUMap[hash_distcoeffs]
        eq_fv = GlobalEQFVMap[hash_distcoeffs]
    else:
        ori_width = width
        ori_height = height
        width //= downsample
        height //= downsample

        u_pos = np.arange(0, ori_width, downsample, dtype=np.float32)
        u_pos = np.tile(u_pos, (height, 1))
        v_pos = np.arange(0, ori_height, downsample, dtype=np.float32)
        v_pos = np.tile(v_pos, (width, 1)).transpose()

        uv_points = np.stack([u_pos, v_pos]).reshape((2, -1)).transpose()
        uv_points = uv_points.reshape(-1, 1, 2)
        e_f = cal_equivalent_focal_length(
            uv_points, calib, dist, (ori_width, ori_height), undistort_by_cv
        )
        e_f_mat = e_f.reshape(height, width)

        cu = calib[0, 2]
        cv = calib[1, 2]
        dfu = np.zeros_like(e_f_mat)
        dfv = np.zeros_like(e_f_mat)
        offset_u = u_pos - cu
        offset_v = v_pos - cv

        dfu[:, 1:-1] = (e_f_mat[:, 2:] - e_f_mat[:, :-2]) / (2 * downsample)
        dfu[:, 0] = (e_f_mat[:, 1] - e_f_mat[:, 0]) / downsample
        dfu[:, -1] = (e_f_mat[:, -1] - e_f_mat[:, -2]) / downsample
        eq_fu = e_f_mat / (1 - dfu * offset_u / e_f_mat).clip(min=1)
        eq_fu.clip(min=1)

        dfv[1:-1] = (e_f_mat[2:] - e_f_mat[:-2]) / (2 * downsample)
        dfv[0] = (e_f_mat[1] - e_f_mat[0]) / downsample
        dfv[-1] = (e_f_mat[-1] - e_f_mat[-2]) / downsample
        eq_fv = e_f_mat / (1 - dfv * offset_v / e_f_mat).clip(min=1)
        eq_fv.clip(min=1)

        interp_eq_fu = interp2d(
            np.arange(0, ori_width, downsample),
            np.arange(0, ori_height, downsample),
            eq_fu,
            kind="linear",
        )
        interp_eq_fv = interp2d(
            np.arange(0, ori_width, downsample),
            np.arange(0, ori_height, downsample),
            eq_fv,
            kind="linear",
        )
        eq_fu = interp_eq_fu(np.arange(ori_width), np.arange(ori_height))
        eq_fv = interp_eq_fv(np.arange(ori_width), np.arange(ori_height))

        GlobalEQFUMap[hash_distcoeffs] = eq_fu
        GlobalEQFVMap[hash_distcoeffs] = eq_fv

    return eq_fu, eq_fv


@OBJECT_REGISTRY.register
class ROI3DDecoder(nn.Module):
    def __init__(
        self,
        focal_length_default: float,
        scale_wh: Tuple[float, float],
        input_padding: Sequence[int] = (0, 0, 0, 0),
        image_hw: Optional[Tuple[int, int]] = None,
        undistort_depth_uv: bool = False,
    ):
        super().__init__()
        assert focal_length_default is not None

        self.undistort_depth_uv = undistort_depth_uv
        self.focal_length_default = focal_length_default
        assert len(input_padding) == 4
        self.input_padding = input_padding
        if image_hw is not None:
            image_hw = np.array(image_hw)
        self.image_hw = image_hw
        affine_mat = torch.tensor(
            [[scale_wh[0], 0, 0], [0, scale_wh[1], 0], [0, 0, 1]]
        )  # noqa

        affine_inv = torch.linalg.pinv(affine_mat)
        self.register_buffer("affine_mat_inv", affine_inv, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        batch_rois: List[torch.Tensor],
        head_out: Dict[str, List[torch.Tensor]],
        calib: torch.Tensor,
        distCoeffs: torch.Tensor,
        im_hw: Optional[torch.Tensor] = None,
    ):
        if im_hw is None:
            assert self.image_hw is not None
            im_hw = [self.image_hw for _ in batch_rois]
        else:
            im_hw = im_hw.cpu().numpy()

        calib = calib.cpu().numpy()
        distCoeffs = distCoeffs.cpu().numpy()

        pre_cls_maps = head_out["cls_pred"]
        pre_2d_offsets = head_out["offset_2d_pred"]
        pre_3d_offsets = head_out["offset_3d_pred"]
        pre_dims = head_out["dims_pred"]
        pre_rots = head_out["rot_pred"]
        pre_ious = head_out["iou_pred"]

        try:
            batch_rois = torch.stack(
                [rois.dequantize() for rois in batch_rois]
            )
        except NotImplementedError:
            batch_rois = torch.stack(
                [rois.as_subclass(torch.Tensor) for rois in batch_rois]
            )

        batch_rois[..., ::2] -= self.input_padding[0]
        batch_rois[..., 1::2] -= self.input_padding[2]

        bs = batch_rois.shape[0]
        pre_cls_maps = pre_cls_maps.unflatten(0, (bs, -1))
        pre_2d_offsets = pre_2d_offsets.unflatten(0, (bs, -1))
        pre_3d_offsets = pre_3d_offsets.unflatten(0, (bs, -1))
        pre_dims = pre_dims.unflatten(0, (bs, -1))
        pre_rots = pre_rots.unflatten(0, (bs, -1))
        pre_ious = pre_ious.unflatten(0, (bs, -1))

        if self.undistort_depth_uv:
            pre_depth_us = head_out["depth_u_pred"]
            pre_depth_vs = head_out["depth_v_pred"]

            pre_depth_us = pre_depth_us.unflatten(0, (bs, -1))
            pre_depth_vs = pre_depth_vs.unflatten(0, (bs, -1))
            pre_depths = zip(pre_depth_us, pre_depth_vs)
        else:
            pre_depths = head_out["pre_depth"]
            pre_depths = pre_depths.unflatten(0, (bs, -1))

        pred_results = []
        for i, (
            rois,
            pre_cls_map,
            pre_2d_offset,
            pre_3d_offset,
            pre_depth,
            pre_dim,
            pre_rot,
            pre_iou,
        ) in enumerate(
            zip(
                batch_rois,
                pre_cls_maps,
                pre_2d_offsets,
                pre_3d_offsets,
                pre_depths,
                pre_dims,
                pre_rots,
                pre_ious,
            )
        ):
            _, _, feat_width, feat_height = pre_cls_map.shape

            pre_cls_map = pre_cls_map.view(-1, feat_width * feat_height)
            offset_2d_x = pre_2d_offset[:, :1].view(
                -1, feat_width * feat_height
            )  # noqa
            offset_2d_y = pre_2d_offset[:, 1:].view(
                -1, feat_width * feat_height
            )  # noqa
            offset_3d_x = pre_3d_offset[:, :1].view(
                -1, feat_width * feat_height
            )  # noqa
            offset_3d_y = pre_3d_offset[:, 1:].view(
                -1, feat_width * feat_height
            )  # noqa

            max_inds = torch.argmax(pre_cls_map, dim=1, keepdim=True)
            max_offset_2d_x = torch.gather(
                offset_2d_x, index=max_inds, dim=1
            ).squeeze(dim=1)
            max_offset_2d_y = torch.gather(
                offset_2d_y, index=max_inds, dim=1
            ).squeeze(dim=1)
            max_offset_3d_x = torch.gather(
                offset_3d_x, index=max_inds, dim=1
            ).squeeze(dim=1)
            max_offset_3d_y = torch.gather(
                offset_3d_y, index=max_inds, dim=1
            ).squeeze(dim=1)

            scales_x = feat_width / (rois[:, 2] - rois[:, 0] + 1)
            scales_y = feat_height / (rois[:, 3] - rois[:, 1] + 1)
            offsets_x = rois[:, 0]
            offsets_y = rois[:, 1]

            max_inds_x = max_inds.squeeze(dim=1) % feat_width
            max_inds_y = (max_inds.squeeze(dim=1) / feat_width).floor()
            pred_2d_x = (max_inds_x + max_offset_2d_x) / scales_x + offsets_x
            pred_2d_y = (max_inds_y + max_offset_2d_y) / scales_y + offsets_y

            raw_center_2d = torch.cat(
                [pred_2d_x.unsqueeze(dim=1), pred_2d_y.unsqueeze(dim=1)], dim=1
            )

            scale = self.affine_mat_inv[0, 0].item()
            ori_center_2d = affine_transform_nd(
                raw_center_2d, self.affine_mat_inv
            )
            affine_center_2d = getUnDistortPoints(
                ori_center_2d.cpu().numpy(),
                calib[i],
                distCoeffs[i],
                im_hw[i][::-1] * scale,
            )
            if affine_center_2d.shape[0] == 0:
                affine_center_2d = torch.zeros_like(raw_center_2d)
            else:
                affine_center_2d = torch.from_numpy(affine_center_2d).to(
                    rois.device
                )
            if self.undistort_depth_uv:
                img_h = int(im_hw[i][0] * self.affine_mat_inv[1, 1])
                img_w = int(im_hw[i][1] * self.affine_mat_inv[0, 0])

                eq_fu, eq_fv = cal_equivalent_focal_length_uv_mat(
                    img_w, img_h, calib[i], distCoeffs[i]
                )  # noqa

                eq_fu = torch.from_numpy(eq_fu).to(rois.device)
                eq_fv = torch.from_numpy(eq_fv).to(rois.device)

                dist_ct = (
                    affine_transform_nd(raw_center_2d, self.affine_mat_inv)
                    .long()
                    .T
                )

                dist_ct[0].clamp_(min=0, max=img_w - 1)
                dist_ct[1].clamp_(min=0, max=img_h - 1)
                eq_fu = eq_fu[dist_ct[1], dist_ct[0]]
                eq_fv = eq_fv[dist_ct[1], dist_ct[0]]

                pre_depth_u, pre_depth_v = pre_depth
                pre_depth_u = pre_depth_u.view(-1, feat_width * feat_height)
                pre_depth_v = pre_depth_v.view(-1, feat_width * feat_height)

                max_depth_u = torch.gather(
                    pre_depth_u, index=max_inds, dim=1
                ).squeeze(dim=1)
                max_depth_v = torch.gather(
                    pre_depth_v, index=max_inds, dim=1
                ).squeeze(dim=1)

                max_depth_u = 1.0 / (max_depth_u.sigmoid() + 1e-6) - 1.0
                max_depth_v = 1.0 / (max_depth_v.sigmoid() + 1e-6) - 1.0
                max_depth_u *= eq_fu
                max_depth_v *= eq_fv

                max_depth = (
                    (max_depth_u + max_depth_v) / self.focal_length_default / 2
                )

            else:
                depth = pre_depth.view(-1, feat_width * feat_height)
                max_depth = torch.gather(depth, index=max_inds, dim=1).squeeze(
                    dim=1
                )
                max_depth = 1.0 / (torch.sigmoid(max_depth) + 1e-6) - 1.0
                max_depth = max_depth * calib[0, 0] / self.focal_length_default

            center_3d = unproject_2d_to_3d_nd(
                affine_center_2d, max_depth, calib[i]
            )  # noqa
            max_offset_3d = torch.stack(
                [max_offset_3d_x, max_offset_3d_y], dim=1
            )
            max_offset_3d = (
                max_offset_3d * calib[i, 0, 0] / self.focal_length_default
            )
            center_3d[:, :2] = center_3d[:, :2] + max_offset_3d
            locations = center_3d.clone()

            pre_dim = pre_dim.view(-1, 3)
            locations[:, [1]] += pre_dim[:, [0]] / 2
            pre_rot = pre_rot.reshape(-1, 2)

            pre_rot /= torch.linalg.norm(pre_rot, dim=1, keepdim=True)

            rot_sin = pre_rot[:, 0]
            rot_cos = pre_rot[:, 1]
            alpha = torch.arctan(rot_sin / (rot_cos + 1e-7))
            cos_pos_idx = (rot_cos >= 0).float()
            cos_neg_idx = (rot_cos < 0).float()
            alpha += (cos_neg_idx - cos_pos_idx) * np.pi / 2.0

            theta = torch.arctan(locations[:, 0] / (locations[:, 2] + 1e-7))
            rotation_y = alpha + theta

            rotation_y = torch.where(
                rotation_y > np.pi, rotation_y - 2 * np.pi, rotation_y
            )  # noqa
            rotation_y = torch.where(
                rotation_y < -np.pi, rotation_y + 2 * np.pi, rotation_y
            )

            iou_score = pre_iou.sigmoid()

            pred_results.append(
                DetBoxes3D(
                    yaw=rotation_y,
                    x=locations[:, 0],
                    y=locations[:, 1],
                    z=locations[:, 2],
                    h=pre_dim[:, 0],
                    w=pre_dim[:, 1],
                    l=pre_dim[:, 2],
                    scores=iou_score[:, 0, 0, 0],
                    cls_idxs=torch.ones_like(rotation_y),
                )
            )

        return {"pred_roi_3d": pred_results}
