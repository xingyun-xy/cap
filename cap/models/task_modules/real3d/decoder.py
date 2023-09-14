# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection
import math
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cap.core.affine import get_affine_transform
from cap.models.utils import multi_class_nms
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import convert_numpy as to_numpy
from cap.core.data_struct.base_struct import DetBoxes3D
from cap.core.data_struct.app_struct import (
    DetObject,
    DetObjects,
    build_task_struct,
)

__all__ = ["Real3DDecoder"]


def _affine_transform_torch(pt, M):
    ones = torch.ones(pt.shape[0], pt.shape[1], 1).to(pt.device)
    new_pt = torch.cat([pt, ones], dim=2)
    new_pt = new_pt[:, :, :, None]
    M = M.permute(0, 2, 1).contiguous()[:, None]
    new_pt = (new_pt * M).sum(dim=2)
    return new_pt


def format_angle(
        angle: Union[torch.Tensor,
                     np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    angle[angle > np.pi] -= 2 * np.pi
    angle[angle < -np.pi] += 2 * np.pi
    return angle


def get_orientation(
    pred_alpha_z: torch.Tensor,
    location: torch.Tensor,
    bin_center: Sequence[float],
) -> Tuple[torch.Tensor]:
    """Calculate rotation y by predictions.

    Args:
        pred_alpha_z (torch.Tensor): predictd alpha z, shape is [B,N,C]
        location (torch.Tensor): predictd location XYZ, shape is [B,N,3]
        bin_center (Sequence[float]): multiple bin centers

    Returns:
        Tuple[torch.Tensor]: rotation y, alpha_x, alpha_z, theta
    """
    b, n, c = pred_alpha_z.shape
    bin_center = np.array(bin_center, np.float32)
    bin_size = len(bin_center)

    pred_bin_cls = pred_alpha_z[..., :bin_size].sigmoid_()  # [b,n,bin_size]
    bin_id = torch.argmax(pred_bin_cls, dim=-1)  # [b,n]

    pred_bin_off = pred_alpha_z[..., bin_size:].reshape((b, n, bin_size, 2))
    pred_bin_off = F.normalize(pred_bin_off, dim=-1)
    # [b,n,1]
    offset_sin = pred_bin_off[..., 0].gather(-1, bin_id[..., None])
    offset_cos = pred_bin_off[..., 1].gather(-1, bin_id[..., None])

    alpha_z_ct = torch.from_numpy(bin_center).to(offset_sin.device)
    alpha_z_ct = alpha_z_ct.reshape((1, 1, bin_size)).repeat((b, n, 1))
    alpha_z_ct = alpha_z_ct.gather(-1, bin_id[..., None])  # [b,n,1]
    alpha_z_offset = torch.atan2(offset_sin, offset_cos)  # [b,n,1]
    alpha_z = format_angle(alpha_z_offset + alpha_z_ct)[..., 0]
    alpha_x = format_angle(alpha_z - np.pi / 2.0)  # [b,n]

    # [b,n]
    theta = format_angle(-torch.atan2(location[..., 0], location[..., 2]))
    roty = format_angle(alpha_x - theta)

    return roty, alpha_x, alpha_z, theta


@OBJECT_REGISTRY.register
class Real3DDecoder(torch.nn.Module):
    """The decoder for Real3D.

    Args:
        focal_length_default (float): The default focal length of the camera
            which is used to project the bbox to world coordinates.
        topk (int): The maximum number of objects in a single
            image.
        max_pooling_kernel_size (int, default=3): The max pooling kernel
            that is used to do the nms.
        center_shift(list or tuple): coord shift(x, y) of center point in
            input image, x/y could be positive or negative, in pixel
        undistort(bool): whether undistort image points.
        fisheye(bool): whether process fisheye data.
        pre_resize_scale (float): Default is -1.0.
            If `pre_resize_scale` > 0, it will rescale `size` by
            pre_resize_scale, and then crop or padding.
        nms_kwargs (dict): NMS arguments, default is None
        use_multibin: bool, whether use multi-bin to get rotation.
        multibin_centers (tuple): tuple of float, alpha multi-bin centers.
            the domain is [-pi, +pi].
    """

    def __init__(self,
                 focal_length_default: float,
                 topk: int,
                 max_pooling_kernel_size=3,
                 center_shift=(0, 0),
                 undistort=False,
                 fisheye=False,
                 pre_resize_scale=-1.0,
                 nms_kwargs=None,
                 use_multibin=False,
                 multibin_centers=(0.0, math.pi / 2, math.pi, -math.pi / 2),
                 score_thres=0.2,
                 **kwargs):
        super(Real3DDecoder, self).__init__()
        self.focal_length_default = focal_length_default
        self.topk = topk
        self.max_pooling_kernel_size = max_pooling_kernel_size
        self.center_shift = center_shift
        self.undistort = undistort
        self.fisheye = fisheye
        self.pre_resize_scale = pre_resize_scale
        self.nms_kwargs = nms_kwargs
        self.use_multibin = use_multibin
        self.multibin_centers = np.array(multibin_centers, np.float32)
        self.score_thres = score_thres
        self.use_gt = kwargs.get('is_use_gt', False)

    def nms(self, pred, iou_threshold, replace=False):
        score = pred["score"].float()
        bbox = pred["bbox"].float()
        cate_id = pred["category_id"].float()
        batch, n = score.shape
        keep = score.new_zeros((batch, n), dtype=torch.int8)
        for bi in range(batch):
            bi_idx = multi_class_nms(bbox[bi], score[bi], cate_id[bi],
                                     iou_threshold)
            keep[bi] = keep[bi].scatter(0, bi_idx, 1)
            if replace:
                score[bi][keep[bi] == 0] = 1e-5
        pred["nms_keep"] = keep
        return keep

    def forward(self,
                pred,
                label,
                data,
                convert_task_sturct=False,
                task_name='vehicle_heatmap_3d_detection'):

        # 根据data重新构建label       add by zmj
        label = dict(
            calibration=data['calib'],
            image_transform={"original_size": [
                data['ori_img_shape'][:,1],
                data['ori_img_shape'][:,0]
            ]},
            dist_coffes=data['distCoeffs']
        )

        image_transform = label["image_transform"]
        calibration = label["calibration"]
        # max_pooling nms
        heatmap = pred["hm"].sigmoid()
        batch_size, _, height, width = heatmap.shape
        kernel = self.max_pooling_kernel_size
        pad = (kernel - 1) // 2
        maxp = F.max_pool2d(heatmap, (kernel, kernel),
                            stride=1,
                            padding=(pad, pad))
        keep = (maxp == heatmap).float()
        pred["hm"] = heatmap * keep
        pred["dep"] = 1.0 / (pred["dep"].sigmoid() + 1e-6) - 1.0

        for key, out in pred.items():
            channel_size = out.shape[1]
            pred[key] = out.permute(0, 2, 3, 1).contiguous()
            pred[key] = pred[key].view(batch_size, -1, channel_size)

        heatmap = pred.pop("hm")
        scores, classes = heatmap.max(dim=-1)
        scores, topk_indices = scores.topk(k=self.topk, dim=1)
        pred["category_id"] = classes

        for key, out in pred.items():
            pred[key] = torch.cat(
                [_out[_inds][None] for _out, _inds in zip(out, topk_indices)],
                dim=0,
            )
        pred["score"] = scores
        u, v = topk_indices % width, topk_indices // width
        center = torch.cat([u[:, :, None], v[:, :, None]], axis=-1)

        ori_width, ori_height = image_transform["original_size"]
        image_original_size = torch.cat(
            [ori_width[:, None], ori_height[:, None]], dim=1)
        inv_Ms = [
            get_affine_transform(
                trans_size.cpu().numpy(),
                0,
                [width, height],
                inverse=True,
                center_shift=self.center_shift,
                pre_resize_scale=self.pre_resize_scale,
            ) for trans_size in image_original_size
        ]

        inv_Ms = torch.cat(
            [
                torch.from_numpy(inv_M).to(center.device)[None]
                for inv_M in inv_Ms
            ],
            dim=0,
        )

        if "wh" in pred:
            wh = pred.pop("wh")
            bbox_x1y1 = center - wh / 2
            bbox_x2y2 = center + wh / 2
            bbox_x1y1 = _affine_transform_torch(bbox_x1y1, inv_Ms)
            bbox_x2y2 = _affine_transform_torch(bbox_x2y2, inv_Ms)

            bbox = torch.cat([bbox_x1y1, bbox_x2y2], dim=2)
            pred["bbox"] = bbox

        center = _affine_transform_torch(center, inv_Ms)
        pred["center"] = center

        dep = pred.pop("dep")
        calibration = calibration.to(dep.device)
        norm_focal = (calibration[:, 0, 0] + calibration[:, 1, 1]) / 2
        camera_focal_length = norm_focal[:, None, None]
        focal_length_scaler = camera_focal_length / self.focal_length_default
        dep *= focal_length_scaler
        pred["dep"] = dep.squeeze(dim=-1)

        if self.undistort:
            dist_coeffs = to_numpy(label["dist_coeffs"])
            center_undistort = center.new_zeros(center.shape)
            for i in range(len(center)):
                ct_np = to_numpy(center[i]).reshape(-1, 1, 2)
                camera = to_numpy(calibration[i, :3, :3])
                dist_coeffs_i = dist_coeffs[i]
                if self.fisheye:
                    ct_undistort = cv2.fisheye.undistortPoints(
                        ct_np, camera, dist_coeffs_i, None, camera).squeeze()
                else:
                    ct_undistort = cv2.undistortPoints(ct_np, camera,
                                                       dist_coeffs_i, None,
                                                       camera).squeeze()
                center_undistort[i] = torch.from_numpy(ct_undistort)
            center = center_undistort

        z = dep - calibration[:, 2, 3][:, None, None]
        cx = calibration[:, 0, 2][:, None, None]
        cy = calibration[:, 1, 2][:, None, None]
        fx = calibration[:, 0, 0][:, None, None]
        fy = calibration[:, 1, 1][:, None, None]
        tx = calibration[:, 0, 3][:, None, None]
        ty = calibration[:, 1, 3][:, None, None]
        x = (center[:, :, [0]] * dep - tx - cx * z) / fx
        y = (center[:, :, [1]] * dep - ty - cy * z) / fy
        loc = torch.cat([x, y, z], dim=-1)
        dim = pred["dim"]
        loc[:, :,
            1] += dim[:, :, 0] / 2  # loc从3dbox的几何中心点变换到底面中心点     note by zmj
        pred["location"] = loc

        if "loc_offset" in pred:
            loc_offset = pred.pop("loc_offset")
            loc_offset *= focal_length_scaler
            loc[:, :, :2] += loc_offset

        alpha = pred.pop("rot")
        if self.use_multibin:
            rot_y, alpha_x, alpha_z, theta = get_orientation(
                alpha, loc, self.multibin_centers)
            alpha = alpha_x
        else:
            alpha = torch.atan2(alpha[:, :, 0], alpha[:, :, 1]) - np.pi / 2
            rot_y = alpha + torch.atan2(loc[:, :, 0], loc[:, :, 2])
            rot_y[rot_y > np.pi] -= 2 * np.pi
            rot_y[rot_y < -np.pi] += 2 * np.pi
        pred["alpha"] = pred['real_alpha_x'] if self.use_gt else alpha
        pred["rotation_y"] = pred['real_rot_y'] if self.use_gt else rot_y

        # TODO nms要改为3dnms或者bevnms
        if self.nms_kwargs is not None:
            self.nms(pred, **self.nms_kwargs)

        if convert_task_sturct:
            results = self.convert2task_struct(pred, task_name)
            return results

        return pred

    def convert2task_struct(self, pred, task_name):

        lengths = [len(res) for res in pred.values()]
        assert max(lengths) == min(lengths)
        results = []
        for b in range(max(lengths)):
            # score过滤,另外nms里的replace=true时会将nms过滤的score赋0，这里一起过滤掉
            keep = pred['score'][b] > self.score_thres
            pred_tmp = {k: v[b][keep] for k, v in pred.items()}
            detboxes3d = DetBoxes3D(
                cls_idxs=pred_tmp["category_id"],
                scores=pred_tmp["score"],
                h=pred_tmp["dim"][:, 0],
                w=pred_tmp["dim"][:, 1],
                l=pred_tmp["dim"][:, 2],
                x=pred_tmp["location"][:, 0],
                y=pred_tmp["location"][:, 1],
                z=pred_tmp["location"][:, 2],
                yaw=pred_tmp["rotation_y"],
                bbox=pred_tmp["bbox"],
            )
            res_tmp = {task_name: detboxes3d}
            _, TASK_STRUCT = build_task_struct(
                "DetObject",
                "DetObjects",
                [(task_name, DetBoxes3D)],
                bases=(DetObject, DetObjects),
            )

            res_struct = TASK_STRUCT(**res_tmp)
            results.append(res_struct)
        return results
