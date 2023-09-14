# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict, defaultdict
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from cap.models.losses.real3d_losses import (
    hm_focal_loss,
    hm_l1_loss,
    sigmoid_and_clip,
)
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .utils import weight_reduce_loss

__all__ = [
    "DepthLoss",
    "DepthPoseResflowLoss",
    "ConsistencyCrossEntropyLoss",
    "LossCalculationWrapper",
    "BEV3DLoss",
    "BEVSegLoss",
]


def rot_from_axisangle(vec: torch.Tensor):
    """Convert an axisangle rotation into a 4x4 transformation matrix.

    (adapted from https://github.com/Wallacoloo/printipi)

    Args:
        vec (torch.tensor): shape is (b,1,3).
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def transformation_from_parameters(
    axisangle: torch.Tensor,
    translation: torch.Tensor,
    residual_flow: Optional[torch.Tensor] = None,
    invert: bool = False,
    scale_coe: float = 0.01,
):  # noqa: D205,D400
    """Convert the network's (axisangle, translation) predict
        into a 4x4 transformation matrix.

    Args:
        axisangle (torch.tensor): shape is (b,1,3).
        translation (torch.tensor): shape is (b,1,3).
        residual_flow (torch.tensor,None): shape is (b,3,h,w).
        invert (bool): shape is (b,1,3). whether invert the rotation matrix
             and translation matrix.
        scale_coe (float): a scalar to scale rotation and translation.

    """
    axisangle = scale_coe * axisangle.mean(dim=(2, 3)).view(-1, 1, 3)
    translation = scale_coe * translation.mean(dim=(2, 3)).view(-1, 1, 3)

    R = rot_from_axisangle(axisangle)[:, :-1, :-1]  # [b,3,3]
    t = translation.view(-1, 3, 1)
    if residual_flow is not None:
        residual_flow = residual_flow.view(residual_flow.shape[0], 3, -1)
        t = t + residual_flow * scale_coe
    if invert:
        R = R.transpose(1, 2)  # [b,3,3]
        t = R @ t * -1
    # R:[b,3,3]，t:(b,3,hw) if residual_flow is not none else (b,3,1)
    return R, t


@OBJECT_REGISTRY.register
class DepthLoss(nn.Module):
    def __init__(
        self,
        low=0.001,
        high=150.0,
        loss_weight=1.0,
    ):
        """Calculate depth l1 loss.

        Args:
            low (float): low threshold value durning calculating.
            high (flaot): high threshold value durning calculating.
            loss_weight (float): loss weight.
        """
        super(DepthLoss, self).__init__()
        self.low = low
        self.high = high
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_depths: Union[torch.Tensor, Sequence[torch.Tensor]],
        gt_depth: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D205,D400
        """
        Args:
            pred_depths (tensor or list of tensor):
                multi-stride predict depth maps.
            gt_depth (tensor):
                gt depth map.
        """
        pred_depths = _as_list(pred_depths)
        gt_depth_mask = (gt_depth > self.low) * (gt_depth < self.high)
        valid_element = gt_depth_mask.sum() + 0.0001

        loss = 0
        for depth in pred_depths:
            diff = (depth - gt_depth).abs() * gt_depth_mask
            diff_rel = (
                1 + 1.2 * ((self.high - gt_depth) / self.high) ** 2
            ) * diff
            loss += diff_rel.sum() / valid_element
        return loss * self.loss_weight


@OBJECT_REGISTRY.register
class DepthConfidenceLoss(nn.Module):
    def __init__(
        self,
        low=0.001,
        high=150.0,
        loss_weight=1.0,
    ):
        """Calculate depth l1 loss and confidence loss.

        Args:
            low (float): low threshold value durning calculating.
            high (flaot): high threshold value durning calculating.
            loss_weight (float): loss weight.
        """
        super(DepthConfidenceLoss, self).__init__()
        self.low = low
        self.high = high
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_depths: Union[torch.Tensor, Sequence[torch.Tensor]],
        gt_depth: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D205,D400
        """
        Args:
            pred_depths (tensor or list of tensor):
                multi-stride predict depth maps.
            gt_depth (tensor):
                gt depth map.
        """
        pred_depths = _as_list(pred_depths)
        gt_depth_mask = (gt_depth > self.low) * (gt_depth < self.high)
        valid_element = gt_depth_mask.sum(dim=(1, 2, 3), keepdim=True) + 0.0001
        gt_depth = gt_depth * gt_depth_mask

        l1_loss = 0
        confidence_loss = 0
        result = OrderedDict()
        for depth_conf in pred_depths:
            depth_conf = depth_conf * gt_depth_mask
            depth, coef = torch.split(depth_conf, 1, dim=1)

            diff = (depth - gt_depth).abs()

            diff_cf = (
                torch.abs(coef - torch.exp(-diff / (gt_depth + 0.00001)))
                * gt_depth_mask
            )
            diff_rel = (
                1 + 1.2 * ((self.high - gt_depth) / self.high) ** 2
            ) * diff

            l1_loss += (
                diff_rel.sum(dim=(1, 2, 3), keepdim=True) / valid_element
            ).mean()
            confidence_loss += (
                diff_cf.sum(dim=(1, 2, 3), keepdim=True) / valid_element
            ).mean()
        result["depth_l1_loss"] = l1_loss * self.loss_weight
        result["depth_confidence_loss"] = confidence_loss * self.loss_weight

        return result


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images."""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (
            sigma_x + sigma_y + self.C2
        )

        ssim_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean(
            1, keepdim=True
        )
        return ssim_loss


class Project3D(nn.Module):
    """Transform coordinates in the camera frame to the pixel frame."""

    def __init__(self, eps=1e-3):
        super(Project3D, self).__init__()
        self.eps = eps

    def forward(self, cam_coords, rot, trans, intrinsics):  # noqa: D205,D400
        """
        Args:
            cam_coords: pixel coordinates defined in the first
                        camera coordinates system -- [B, 3, H, W]
            rot: rotation matrix of cameras -- [B, 3, 3]
            trans: translation vectors of cameras -- [B, 3, 1] or [B,3,HW]
            intrinsics:  intrinscis matrix
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        # depth:(n,1,h,w)
        b, _, h, w = cam_coords.shape
        cam_coords_flat = cam_coords.view(b, 3, -1)
        new_cam_points = torch.matmul(rot, cam_coords_flat)
        new_cam_points = new_cam_points + trans  # [B, 3, H*W]

        X = new_cam_points[:, 0:1]  # [B,1,H*W]
        Y = new_cam_points[:, 1:2]  # [B,1,H*W]
        Z = new_cam_points[:, 2:3].clamp(
            min=self.eps, max=float("inf")
        )  # [B,1,H*W]
        X, Y = X / Z, Y / Z  # normalized plane [B, 1,H*W]

        norm_points_dist = torch.cat((X, Y, torch.ones_like(X)), 1)
        pix_coords = torch.matmul(intrinsics, norm_points_dist)
        pix_coords = pix_coords[:, :2].view(b, 2, h, w)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        pix_coords[..., 0] /= w - 1
        pix_coords[..., 1] /= h - 1
        pix_coords = (pix_coords - 0.5) * 2

        valid_points = pix_coords.abs().max(dim=-1)[0].unsqueeze(1) <= 1

        return pix_coords, valid_points.float()


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud."""

    def __init__(
        self,
        height=256,
        width=480,
    ):
        super(BackprojectDepth, self).__init__()
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(width), range(height), indexing="xy")
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)  # (2,h,w)

        ones = np.ones((1, height, width), dtype="float32")
        pix_coords = np.concatenate([id_coords, ones], axis=0).reshape(
            (1, 3, -1)
        )

        self.pix_coords = nn.Parameter(
            torch.from_numpy(pix_coords), requires_grad=False
        )

    def forward(self, depth, intrinscis):
        # depth:(n,1,h,w),intrinscis:(n,3,3)
        inv_intrinscis = torch.inverse(intrinscis)
        cam_points = torch.matmul(inv_intrinscis, self.pix_coords)
        cam_points = cam_points.view(-1, 3, self.height, self.width) * depth
        return cam_points


@OBJECT_REGISTRY.register
class PoseResflowLoss(nn.Module):
    def __init__(
        self,
        scale_list=(2, 4, 8, 16),
        input_size=(512, 960),
        loss_weight=1.0,
        collect_vis=False,
    ):

        super(PoseResflowLoss, self).__init__()
        self.scale_list = scale_list
        self.input_size = input_size
        self.loss_weight = loss_weight
        self.collect_vis = collect_vis

        self.set_normalized_grid_with_iteration()
        self.project_3d = Project3D(eps=1e-3)
        self.ssim = SSIM()

    def set_normalized_grid_with_iteration(self):
        input_h, input_w = self.input_size
        self.backproject_depth = nn.ModuleDict()
        for scale_factor in self.scale_list:  # multi scale normalized grid
            h, w = input_h // scale_factor, input_w // scale_factor
            self.backproject_depth.update(
                {"%d" % scale_factor: BackprojectDepth(h, w)}
            )

    def inverse_warp_with_residualflow(
        self,
        img,
        depth,
        axisangle,
        translation,
        residual_flow,
        intrinsics,
        backproject_depth,
        invert=False,
        get_2d_flow=False,
    ):
        # back project pixel coords to camera coord
        cam_coords = backproject_depth(depth, intrinsics)

        R, t = transformation_from_parameters(
            axisangle, translation, residual_flow=residual_flow, invert=invert
        )
        # R:(b,3,3),t(b,3,1,1) or (b,3,h,w)

        src_pixel_coords, valid_points = self.project_3d(
            cam_coords, R, t, intrinsics
        )
        # without below will cause bug in amp
        img = img.to(src_pixel_coords.dtype)
        # (b,h,w,2)
        projected_img = F.grid_sample(
            img, src_pixel_coords, padding_mode="border", align_corners=False
        )
        flow_2d = None
        if get_2d_flow:
            flow_2d = src_pixel_coords - self.get_id_grid(src_pixel_coords)
        return projected_img, valid_points, flow_2d

    def get_id_grid(self, pixel_coords):
        b, h, w, _ = pixel_coords.shape

        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        meshgrid[1] = 2 * meshgrid[1] / (h - 1) - 1
        meshgrid[0] = 2 * meshgrid[0] / (w - 1) - 1
        id_coords = np.stack(meshgrid, axis=-1).astype(np.float32)  # (h,w,2)
        id_coords = np.expand_dims(id_coords, 0)  # (1,h,w,2)

        id_coords = torch.from_numpy(np.repeat(id_coords, b, 0)).to(
            pixel_coords.device
        )
        return id_coords

    def forward(
        self,
        pred_depths: Union[torch.Tensor, Sequence[torch.Tensor]],
        axisangle: Sequence[torch.Tensor],
        translation: Sequence[torch.Tensor],
        residual_flow: Sequence[Optional[torch.Tensor]],
        color_imgs: Sequence[torch.Tensor],
        intrinsics: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
        front_mask: Optional[torch.Tensor] = None,
        ud_coord: Optional[torch.Tensor] = None,
    ) -> Dict:  # noqa: D205,D400
        """
        Args:
            pred_depths :predict depth map of t,
            axisangle :predict two axisangle,
            translation :predict two translation,
            residual_flow :predict two residual flow,
            color_imgs : original imgs(normalized to [0,1]
                of t-1,t,t+1.
            intrinsics : camera intrinsics matrix.
            obj_mask : binary mask map. 1 means foreground
                pixel otherwise background pixel.
                shape is (b,1,h,w)
            front_mask: binary mask map. 0 means black boundaries in image,
                shape is (b,1,h,w)
            ud_coord: the remap image coordinates to undistort the outputs
                shape is (b,2,h,w)
        """

        pred_depths = _as_list(pred_depths)
        if ud_coord is not None:
            pred_depths = [
                F.grid_sample(
                    F.interpolate(
                        res,
                        ud_coord.shape[1:3],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    ud_coord,
                    padding_mode="border",
                )
                for res in pred_depths
            ]
            residual_flow = [
                F.grid_sample(
                    F.interpolate(
                        res,
                        ud_coord.shape[1:3],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    ud_coord,
                    padding_mode="border",
                )
                for res in residual_flow
            ]

        ref_imgs = [color_imgs[2], color_imgs[0]]  # [t-2,t]
        tgt_img = color_imgs[1]  # [t-1]
        loss = 0
        warp_imgs_vis, recon_loss_vis, flow_vis = None, None, None

        # TODO(wenming.meng, 0.5): too long function, move outsize forward. #
        def one_scale(scale_factor, depth):
            b, _, h, w = depth.shape
            depth = depth.clamp(min=0.01)
            downscale = tgt_img.shape[2] / h

            tgt_img_scaled = F.interpolate(
                tgt_img, [h, w], mode="bilinear", align_corners=False
            )

            obj_mask_scaled = (
                F.interpolate(obj_mask, [h, w], mode="nearest")
                if obj_mask is not None
                else None
            )

            front_mask_scaled = (
                F.interpolate(front_mask, [h, w], mode="nearest")
                if front_mask is not None
                else None
            )

            ref_imgs_scaled = [
                F.interpolate(
                    ref_img, [h, w], mode="bilinear", align_corners=False
                )
                for ref_img in ref_imgs
            ]
            intrinsics_scaled = intrinsics.clone()
            intrinsics_scaled[:, 0:2] /= downscale

            reproj_loss = []
            warped_imgs = []
            for i, ref_img in enumerate(ref_imgs_scaled):

                current_axisangle = axisangle[i]
                current_translation = translation[i]
                current_residual_flow = residual_flow[i]
                if current_residual_flow is not None:
                    current_residual_flow = F.interpolate(
                        current_residual_flow, [h, w]
                    )
                if (
                    obj_mask_scaled is not None
                    and current_residual_flow is not None
                ):
                    current_residual_flow *= obj_mask_scaled

                (
                    ref_img_warped,
                    valid_points,
                    total_flow,
                ) = self.inverse_warp_with_residualflow(
                    ref_img,
                    depth,
                    current_axisangle,
                    current_translation,
                    current_residual_flow,
                    intrinsics_scaled,
                    self.backproject_depth["%d" % scale_factor],
                    invert=i < 1,
                    get_2d_flow=False,
                )
                warped_imgs.append(ref_img_warped)
                _reproj_loss = self.ssim(ref_img_warped, tgt_img_scaled)

                if (
                    current_residual_flow is None
                    and obj_mask_scaled is not None
                ):
                    # do not calcu loss on object area
                    _reproj_loss = _reproj_loss * (1 - obj_mask_scaled)

                if front_mask_scaled is not None:
                    # do not calcu loss on black image boundaries
                    _reproj_loss = _reproj_loss * front_mask_scaled

                reproj_loss.append(_reproj_loss)

            reproj_loss_cat = torch.cat(reproj_loss, 1)  # b2hw
            to_optimise, idxs = torch.min(
                reproj_loss_cat, dim=1, keepdim=True
            )  # b1hw

            if self.collect_vis and scale_factor == self.scale_list[0]:
                s = 1

                r0 = F.interpolate(
                    ref_imgs_scaled[0], scale_factor=s, mode="bilinear"
                )
                # (b,3,h,w)
                r1 = F.interpolate(
                    ref_imgs_scaled[1], scale_factor=s, mode="bilinear"
                )
                # (b,3,h,w)

                tgt_img_scaled = F.interpolate(
                    tgt_img_scaled, scale_factor=s, mode="bilinear"
                )
                # (b,3,h,w)
                w0 = F.interpolate(
                    warped_imgs[0], scale_factor=s, mode="bilinear"
                )
                # (b,3,h,w)
                w1 = F.interpolate(
                    warped_imgs[1], scale_factor=s, mode="bilinear"
                )
                # (b,3,h,w)

                recon_loss0 = F.interpolate(
                    reproj_loss[0], scale_factor=s, mode="bilinear"
                )
                # (b,1,h,w)

                recon_min = F.interpolate(
                    to_optimise, scale_factor=s, mode="bilinear"
                )
                # (b,1,h,w)

                recon_loss1 = F.interpolate(
                    reproj_loss[1], scale_factor=s, mode="bilinear"
                )
                # (b,1,h,w)

                ori_img = torch.cat(
                    (r0[0], tgt_img_scaled[0], r1[0]), 1
                )  # (3,3h,w)
                warp_img = torch.cat(
                    (w0[0], tgt_img_scaled[0], w1[0]), 1
                )  # (3,3h,w)

                if residual_flow is not None:
                    _, _, rig_flow = self.inverse_warp_with_residualflow(
                        ref_imgs_scaled[1],
                        depth,
                        axisangle[1],
                        translation[1],
                        None,
                        intrinsics_scaled,
                        self.backproject_depth["%d" % scale_factor],
                        invert=False,
                        get_2d_flow=True,
                    )
                    _, _, res_flow = self.inverse_warp_with_residualflow(
                        ref_imgs_scaled[1],
                        depth,
                        torch.zeros_like(axisangle[1]).to(axisangle[1].device),
                        torch.zeros_like(translation[1]).to(
                            translation[1].device
                        ),
                        current_residual_flow,
                        intrinsics_scaled,
                        self.backproject_depth["%d" % scale_factor],
                        invert=False,
                        get_2d_flow=True,
                    )

                nonlocal warp_imgs_vis, recon_loss_vis, flow_vis
                warp_imgs_vis = torch.cat(
                    (ori_img, warp_img), dim=2
                )  # (3,3h,2w)
                recon_loss_vis = torch.cat(
                    (recon_loss0[0], recon_min[0], recon_loss1[0]), dim=1
                )
                # (1,3h,w)
                flow_vis = torch.cat(
                    (rig_flow[0], res_flow[0]), dim=1
                )  # (h,2w,2)

            return to_optimise.mean()

        for scale_idx, scale_factor in enumerate(self.scale_list):
            if scale_idx >= len(pred_depths):
                pred_depth = pred_depths[0]
            else:
                pred_depth = pred_depths[scale_idx]

            size = (
                self.input_size[0] // scale_factor,
                self.input_size[1] // scale_factor,
            )
            pred_depth = F.interpolate(
                pred_depth, size=size, mode="bilinear", align_corners=False
            )
            loss += one_scale(scale_factor, pred_depth)
        # TODO(wenming.meng, 0.5): return too many variable, need to refactor.
        return {
            "depth_pose_loss": loss * self.loss_weight,
            "warp_imgs_vis": warp_imgs_vis,
            "recon_vis": recon_loss_vis,
            "flow_vis": flow_vis,
            "depth_vis": pred_depths[0][0][0:1],
        }


@OBJECT_REGISTRY.register
class ConsistencyCrossEntropyLoss(nn.Module):
    def __init__(self, loss_name, loss_weight=1.0, reduction="mean"):
        """
        Calculate consistency CrossEntropy loss between two predict.

        Args:
            loss_weight (float): loss weight.
            reduction (str): The method used to reduce the loss. Options are
                [`none`, `mean`, `sum`].
        """
        super().__init__()
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        data1: torch.Tensor,
        data2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        """
        Calculate consistency CrossEntropy loss between two predict.

        Args:
            data1 (tensor): first data(without softmax).
            data2 (tensor): second predict(without softmax).
            weight (tensor or none): a pixel-wise weight tensor.
        """
        result_dict = {}
        data1_lsm = F.log_softmax(data1, 1)  # (b,c,h,w)
        data2_sm = F.softmax(data2, 1)  # (b,c,h,w)

        loss = -(data1_lsm * data2_sm).sum(1, keepdim=True)  # (b,c,h,w)

        if weight is not None:
            loss = loss * weight
        loss_reduce = weight_reduce_loss(
            loss, weight=None, reduction=self.reduction
        )
        result_dict[self.loss_name] = loss_reduce * self.loss_weight
        result_dict["consistency_vis"] = loss
        return result_dict


@OBJECT_REGISTRY.register
class DepthPoseResflowLoss(nn.Module):
    def __init__(
        self,
        pred_depths_name="pred_depths_frame0",
        gt_depth_name="gt_depth",
        axisangle_name="axisangle",
        translation_name="translation",
        residual_flow_name="residual_flow",
        color_imgs_name="color_imgs",
        intrinscis_name="intrinsics",
        obj_mask_name="obj_mask",
        front_mask_name="front_mask",
        ud_coord_name="ud_coord",
        depth_loss: Optional[torch.nn.Module] = None,
        pose_resflow_loss: Optional[torch.nn.Module] = None,
    ):
        super(DepthPoseResflowLoss, self).__init__()
        assert (
            depth_loss is not None or pose_resflow_loss is not None
        ), "depth loss and pose_resflow loss cannot be none at same time"
        self.pred_depths_name = pred_depths_name
        self.gt_depth_name = gt_depth_name
        self.axisangle_name = axisangle_name
        self.translation_name = translation_name
        self.residual_flow_name = residual_flow_name
        self.color_imgs_name = color_imgs_name
        self.intrinscis_name = intrinscis_name
        self.obj_mask_name = obj_mask_name
        self.front_mask_name = front_mask_name
        self.ud_coord_name = ud_coord_name

        self.depth_loss = depth_loss
        self.pose_resflow_loss = pose_resflow_loss

    @autocast(enabled=False)
    def forward(self, pred_dict, target_dict):
        result_dict = {}
        # cast to fp32
        if self.pred_depths_name in pred_dict:
            # when depth and pose_resflow in single output module.
            pred_depths = [
                data.float() for data in pred_dict[self.pred_depths_name]
            ]  # maybe contain depth and confidence
        else:
            # when depth and pose_resflow in different output module.
            # we need to get pred_depth from target_dict and pop it to avoid .
            pred_depths = target_dict.pop(self.pred_depths_name)
            pred_depths = [
                data.float() for data in pred_depths
            ]  # maybe contain depth and confidence

        if self.depth_loss:
            gt_depth = target_dict[self.gt_depth_name]
            result_dict["loss_depth"] = self.depth_loss(pred_depths, gt_depth)
        if self.pose_resflow_loss:
            axisangle = [
                data.float() for data in pred_dict[self.axisangle_name]
            ]
            translation = [
                data.float() for data in pred_dict[self.translation_name]
            ]
            residual_flow = [
                data.float() for data in pred_dict[self.residual_flow_name]
            ]

            color_imgs = target_dict[self.color_imgs_name]
            intrinsics = target_dict[self.intrinscis_name]
            obj_mask = target_dict.get(self.obj_mask_name, None)
            front_mask = target_dict.get(self.front_mask_name, None)
            ud_coord = target_dict.get(self.ud_coord_name, None)
            # get depth only to calculate pose_resflow_loss, depth is always
            # in first channel.
            pred_depths = [data[:, 0:1] for data in pred_depths]
            result_dict.update(
                self.pose_resflow_loss(
                    pred_depths,
                    axisangle,
                    translation,
                    residual_flow,
                    color_imgs,
                    intrinsics,
                    obj_mask,
                    front_mask,
                    ud_coord,
                )
            )
        return result_dict


@OBJECT_REGISTRY.register
class LossCalculationWrapper(nn.Module):  # noqa: D205,D400
    """Select varibles from dict and calculating loss with
    any type loss moulde builded.

    note: Make sure provided pred_names and tgt_names is proper for
    specific loss module. Because loss module will be called like
    this: loss_module(preds, targets).

    Args:
        pred_names (list(str)): predction`s names
        gt_names (list(str)): ground truth names.
        loss_name (str): loss name in result dict.
        loss_cfg (torch.nn.Module): Loss module.
    Returns:
        a dict contains loss.
    """

    def __init__(
        self,
        pred_names,
        tgt_names,
        loss_name,
        loss_cfg: Optional[torch.nn.Module] = None,
    ):
        super(LossCalculationWrapper, self).__init__()

        self.pred_names = _as_list(pred_names)
        self.tgt_names = _as_list(tgt_names)
        self.loss_name = loss_name

        self.loss = loss_cfg

    def forward(self, pred_dict, target_dict):
        result_dict = {}
        if self.loss:
            preds = []
            targets = []
            for pred_name in self.pred_names:
                preds.append(pred_dict.get(pred_name, None))
            for tgt_name in self.tgt_names:
                targets.append(target_dict.get(tgt_name, None))
            res = self.loss(*preds, *targets)
            if isinstance(res, Mapping):
                return res
            else:
                result_dict[self.loss_name] = res
        return result_dict


@OBJECT_REGISTRY.register
class BEV3DLoss(nn.Module):
    def __init__(
        self,
        loss_weights=None,
        gamma=2,
        beta=1,
    ):
        """Calculate bev3d losses.

            Classification uses focal loss.
            Regression (dimension, rotation, center_offset
                etc.) use L1 loss.

        Args:
            loss_weights:(dict), default: None
                Global loss weight for each sub-loss. Default
                weight is 1.0. (e.g. bev3d_hm:1, bev3d_dim:1,
                bev3d_rot:1 etc.).
            gamma (float): gamma paras of hm_focal_loss.
                Default: 2.
            beta (float): beta paras of hm_focal_loss.
                Default: 1.
        Returns:
            a dict contains loss.
        """

        super(BEV3DLoss, self).__init__()
        self.loss_weights = defaultdict(lambda: 1.0)
        self.gamma = gamma
        self.beta = beta
        if loss_weights is not None:
            self.loss_weights.update(**loss_weights)
        self.reg_keys = list(self.loss_weights.keys())
        if "bev3d_hm" in self.reg_keys:
            self.reg_keys.remove("bev3d_hm")

    @autocast(enabled=False)
    def forward(self, pred: Mapping, target: Mapping):
        """Calculate bev3d losses between pred and target items.

        Args:
            pred (Dict): predict bev3d output (e.g. bev3d_hm, bev3d_dim etc.)
            target (Dict): target contains the bev3d ground truth
        """

        assert "gt_bev_3d" in target
        target = target["gt_bev_3d"]

        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()

        all_losses = {}

        ignore_mask = target.get(
            "bev3d_ignore_mask", torch.zeros_like(target["bev3d_weight_hm"])
        )

        # heatmap use focal loss
        if "bev3d_hm" in self.loss_weights.keys():
            loss_weight = self.loss_weights["bev3d_hm"]
            pred_heatmap = sigmoid_and_clip(pred["bev3d_hm"])
            target_heatmap = target["bev3d_hm"]
            heatmap_loss = (
                hm_focal_loss(
                    pred=pred_heatmap,
                    gt=target_heatmap,
                    ignore_mask=ignore_mask,
                    gamma=self.gamma,
                    beta=self.beta,
                )
                * loss_weight
            )
            all_losses["bev3d_hm_loss"] = heatmap_loss

        # regression map use L1 loss
        for key in self.reg_keys:
            if key not in pred:
                continue
            loss_weight = self.loss_weights[key]
            loss = (
                hm_l1_loss(
                    pred[key],
                    target[key],
                    target["bev3d_weight_hm"],
                    ignore_mask,
                    heatmap_type=None,
                )
                * loss_weight
            )
            all_losses["{}_loss".format(key)] = loss
        return all_losses


@OBJECT_REGISTRY.register
class BEVSegLoss(nn.Module):  # noqa: D205,D400
    def __init__(
        self,
        pred_names,
        loss_names,
        loss_seg_cfg: Optional[torch.nn.Module] = None,
        loss_occlusion_cfg: Optional[torch.nn.Module] = None,
        conf_loss_weight: Optional[float] = None,
    ):
        """Calculate bev seg loss.

        BEV seg head and BEV occlusion head use cross entropy loss.
        BEV confidence head use L1 loss.
        Ground truth for confidence prediction is
        max(softmax(pred_seg_logit)) along channel dim.

        Args:
            pred_names (list(str)): prediction names.
            loss_names (list(str)): loss name for each output.
            loss_seg_cfg (torch.nn.Module): loss for seg.
            loss_occlusion_cfg (torch.nn.Module): loss for seg.
            conf_loss_weight (float): loss weight for confidence output.
        """
        super(BEVSegLoss, self).__init__()

        self.pred_names = _as_list(pred_names)
        self.loss_names = _as_list(loss_names)
        assert len(self.pred_names) == len(self.loss_names)

        self.loss_seg = loss_seg_cfg
        self.loss_occlusion = loss_occlusion_cfg
        self.conf_loss_weight = conf_loss_weight

    def forward(self, pred_dict, target_dict):

        assert "gt_bev_seg" in target_dict
        assert "pred_bev_segs_frame0" in pred_dict

        result_dict = {}
        for i, pred_name in enumerate(self.pred_names):
            if "bev_conf" in pred_name:
                # L1 loss for confidence map
                target_conf, _ = torch.max(
                    torch.softmax(pred_dict["pred_bev_segs_frame0"][0], dim=1),
                    dim=1,
                    keepdim=True,
                )
                res = (
                    hm_l1_loss(
                        pred_dict[pred_name][0],
                        target_conf,
                        torch.ones_like(target_conf),
                        torch.zeros_like(target_conf),
                        heatmap_type=None,
                    )
                    * self.conf_loss_weight
                )
            elif "bev_occulsion" in pred_name:
                res = self.loss_occlusion(
                    pred_dict[pred_name], target_dict["occlusion"]
                )
            elif "bev_segs" in pred_name:
                res = self.loss_seg(
                    pred_dict[pred_name], target_dict["gt_bev_seg"]
                )

            result_dict[self.loss_names[i]] = res

        return result_dict
