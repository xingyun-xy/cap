# Copyright (c) Changan Auto. All rights reserved.

# Implements some commonly used bounding box utilities. Applies to both 2d box
# and 3d box.
from typing import Optional, Tuple, Union

import numpy as np
import torch

from cap.core.point_geometry import coor_transformation

__all__ = [
    "box_center_to_corner",
    "box_corner_to_center",
    "bbox_overlaps",
    "get_bev_bbox",
    "zoom_boxes",
]


def box_center_to_corner(
    bboxes: torch.Tensor,
    split: Optional[bool] = False,
    legacy_bbox: Optional[bool] = False,
):  # noqa: D205,D400
    """
    Convert bounding box from center format (xcenter, ycenter,
    width, height) to corner format (x_low, y_low, x_high, y_high)

    Args:
        bboxes (torch.Tensor): Shape is (..., 4) represents bounding boxes.
        split: (:obj:`bool`, optional): Whether to split the final output to
            for (..., 1) tensors, or keep the (..., 4) original output.
            Default to False.
        legacy_bbox: (:obj:`bool`, optional): Whether the boxes are decoded
            in legacy manner (should add one to bottom or right coordinate
            before using) or not. Default to False.
    """

    border = int(legacy_bbox)
    cx, cy, w, h = torch.split(bboxes, 1, dim=-1)
    x1 = cx - (w - border) * 0.5
    y1 = cy - (h - border) * 0.5
    x2 = x1 + w - border
    y2 = y1 + h - border

    if split:
        return x1, y1, x2, y2
    else:
        return torch.cat([x1, y1, x2, y2], dim=-1)


def box_corner_to_center(
    bboxes: torch.Tensor,
    split: Optional[bool] = False,
    legacy_bbox: Optional[bool] = False,
):  # noqa: D205,D400
    """
    Convert bounding box from corner format (x_low, y_low, x_high, y_high)
    to center format (xcenter, ycenter, width, height)

    Args:
        bboxes (torch.Tensor): Shape is (..., 4) represents bounding boxes.
        split: (:obj:`bool`, optional): Whether to split the final output to
            for (..., 1) tensors, or keep the (..., 4) original output.
            Default to False.
        legacy_bbox: (:obj:`bool`, optional): Whether the boxes are decoded
            in legacy manner (should add one to bottom or right coordinate
            before using) or not. Default to False.
    """

    border = int(legacy_bbox)
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
    width = x2 - x1 + border
    height = y2 - y1 + border
    cx = x1 + (width - border) * 0.5
    cy = y1 + (height - border) * 0.5

    if split:
        return cx, cy, width, height
    else:
        return torch.cat([cx, cy, width, height], dim=-1)


def bbox_overlaps(
    bboxes1: Union[torch.tensor, np.ndarray],
    bboxes2: Union[torch.tensor, np.ndarray],
    mode: Optional[str] = "iou",
    is_aligned: Optional[bool] = False,
    eps: Optional[float] = 1e-6,
):
    """
    Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (Tensor or np.ndarray):
            shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor or np.ndarray):
            shape (n, 4) in <x1, y1, x2, y2> format or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """

    assert isinstance(bboxes1, type(bboxes2))
    is_ndarray = False
    if isinstance(bboxes1, np.ndarray):
        is_ndarray = True
        bboxes1 = torch.from_numpy(bboxes1)
        bboxes2 = torch.from_numpy(bboxes2)

    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new_zeros((rows,))
        else:
            return bboxes1.new_zeros((rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1]
    )
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1]
    )

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(
            bboxes1[:, None, :2], bboxes2[None, :, :2]
        )  # [rows, cols, 2]
        rb = torch.min(
            bboxes1[:, None, 2:], bboxes2[None, :, 2:]
        )  # [rows, cols, 2]
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[:, None, :2], bboxes2[None, :, :2])
            enclosed_rb = torch.max(bboxes1[:, None, 2:], bboxes2[None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious if not is_ndarray else ious.numpy()
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious if not is_ndarray else gious.numpy()


# =============================================================================
# The following methods are mostly used in lidar 3d box processing.
# =============================================================================


def corners_nd(
    dims: np.ndarray, origin: Union[Tuple[float, ...], float] = 0.5
):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray): [N, ndim] tensor. Box size in each dimension.
        origin ([Union[Tuple[float, ...], float]):
            origin point relative to the smallest point. Defaults to 0.5.

    Returns:
        corners (np.ndarray): [N, 2**ndim, ndim] sized tensor of corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim]
    )
    return corners


def rotation_2d(points: np.ndarray, angles: float):
    """Rotate 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(
    centers: np.ndarray,
    dims: np.ndarray,
    angles: Optional[np.ndarray] = None,
    origin: float = 0.5,
):
    """Convert Kitti-style locations, dimensions and angles to corners.

    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        np.ndarray: corner representation of boxes.
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def zoom_boxes(boxes: torch.Tensor, roi_wh_zoom_scale: Tuple[float, float]):
    """Zoom boxes.

    Args:
        boxes: shape (m, 4) in <x1, y1, x2, y2> format.
        roi_wh_zoom_scale: (w_scale, h_scale).

    Returns:
        torch.Tensor: zoomed bboxes.
    """
    boxes = boxes[..., :4]
    boxes_w = boxes[..., 2] - boxes[..., 0]
    boxes_h = boxes[..., 3] - boxes[..., 1]

    w_bias = 0.5 * (roi_wh_zoom_scale[0] - 1) * boxes_w
    h_bias = 0.5 * (roi_wh_zoom_scale[1] - 1) * boxes_h

    return torch.stack(
        [
            boxes[..., 0] - w_bias,
            boxes[..., 1] - h_bias,
            boxes[..., 2] + w_bias,
            boxes[..., 3] + h_bias,
        ],
        dim=-1,
    )


def minmax_to_corner_2d(minmax_box: np.ndarray):
    """Convert min-max representation of a box into corner representation.

    Args:
        minmax_box (np.ndarray): [N, 2*ndim] box. ndim indicates whether it is
            a 2-d box or a 3-d box.

    Returns:
        np.ndarray: corner representation of a boxes.
    """
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def get_bev_bbox(coordinate, size, yaw):

    size = np.clip(size, a_min=1, a_max=None)
    if len(coordinate) == 0:
        return np.zeros([0, 4, 2])

    corners = size / 2
    corners = np.stack(
        [
            corners,
            corners * np.array([1, -1]),
            corners * np.array([-1, -1]),
            corners * np.array([-1, 1]),
        ],
        axis=-2,
    )
    bev_bbox = coor_transformation(corners, yaw[:, None], coordinate[:, None])

    return bev_bbox
