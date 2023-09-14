from typing import Tuple

import cv2
import numpy as np

__all__ = [
    "get_affine_transform",
    "affine_transform",
    "point_affine_transform",
    "point_2D_affine_transform",
    "bbox_affine_transform",
]


def get_affine_transform(
    size: Tuple[float],
    rotation: float,
    out_size: Tuple[float],
    inverse: bool = False,
    center_shift: Tuple[float] = (0, 0),
    pre_resize_scale: float = -1.0,
):
    """Calculate affine transform matrix based input arguments.

    Args:
        size (list or tuple): Width and height of the input image
        rotation (float): rotation degree in angle degree
        out_size (list or tuple): Width and height of the output image
        inverse (bool): If True, the matrix is from output image to input image
        center_shift (list or tuple): coord shift(x, y) of center point in
            input image, x/y could be positive or negative, in pixel
        pre_resize_scale (float): Default is -1.0.
            If `pre_resize_scale` > 0, it will first rescale `size` by
            pre_resize_scale, and then crop or padding.

    Returns (ndarray): The transform matrix.

    """
    in_width, in_height = size
    in_center = np.array([in_width / 2.0, in_height / 2.0], dtype=np.float32)
    center_shift = np.array(center_shift, dtype=np.float32)
    in_center += center_shift
    out_width, out_height = out_size
    out_center = np.array(
        [out_width / 2.0, out_height / 2.0], dtype=np.float32
    )
    rot_rad = np.pi * rotation / 180.0

    # get direction
    sin, cos = np.sin(rot_rad), np.cos(rot_rad)
    R = np.array([[cos, -sin], [sin, cos]])
    in_direction = np.dot(np.array([0, in_width * -0.5]), R.T)
    if pre_resize_scale > 0:
        out_direction = np.array([0, in_width * pre_resize_scale * -0.5])
    else:
        out_direction = np.array([0, out_width * -0.5])

    # get 3 points
    in_pt1 = in_center
    in_pt2 = in_center + in_direction
    delta = in_pt1 - in_pt2
    in_pt3 = in_pt2 + [-delta[1], delta[0]]
    in_pts = np.concatenate(
        [
            in_pt1[None],
            in_pt2[None],
            in_pt3[None],
        ],
        axis=0,
    ).astype(np.float32)

    out_pt1 = out_center
    out_pt2 = out_center + out_direction
    delta = out_pt1 - out_pt2
    out_pt3 = out_pt2 + [-delta[1], delta[0]]
    out_pts = np.concatenate(
        [
            out_pt1[None],
            out_pt2[None],
            out_pt3[None],
        ],
        axis=0,
    ).astype(np.float32)

    if inverse:
        M = cv2.getAffineTransform(out_pts, in_pts)
    else:
        M = cv2.getAffineTransform(in_pts, out_pts)

    return M


def affine_transform(pt, t):
    pt = np.array(pt, dtype=np.float32)
    t = np.array(t, dtype=np.float32)
    assert pt.ndim == 2 and pt.shape[1] == 2
    assert t.ndim == 2 and t.shape[1] == 3
    new_pt = np.concatenate([pt, np.ones([pt.shape[0], 1])], axis=1).T
    new_pt = np.dot(t, new_pt).T
    return new_pt[:, :2]


def point_affine_transform(point, matrix):
    point_exd = np.array([point[0], point[1], 1.0])
    new_point = np.matmul(matrix, point_exd)

    return new_point[:2]


def point_2D_affine_transform(data, affine_mat):
    """Transform 2D points by affine matrix.

    Note that affine matrix should be in the shape of (2, 3),
    and the last dimension of 2D points data should be 2.

    Args:
        data: np.ndarray.
            2D Points to apply transformation.
            The last dimension should be 2.
        affine_mat: np.ndarray. The affine matrix.
            It should be in the shape of (2, 3).

    Returns:
        new_data: np.ndarray
            The 2D points after transformation.
    """
    assert isinstance(data, np.ndarray)
    if affine_mat.shape == (3, 3):
        affine_mat = affine_mat[0:2]
    assert affine_mat.shape == (2, 3)
    assert data.shape[-1] == 2, data.shape
    ori_shape = data.shape
    data = data.reshape((-1, 2))
    new_data = np.ones((data.shape[0], data.shape[1] + 1), dtype=data.dtype)
    new_data[:, 0:2] = data
    new_data = np.dot(new_data, np.transpose(affine_mat))
    return new_data.reshape(ori_shape)


def _refine_bbox(bbox):
    assert bbox.shape[-1] >= 4
    old_shape = bbox.shape
    bbox = bbox.reshape((-1, bbox.shape[-1]))
    x_min = np.min(bbox[:, (0, 2)], axis=1)
    y_min = np.min(bbox[:, (1, 3)], axis=1)
    x_max = np.max(bbox[:, (0, 2)], axis=1)
    y_max = np.max(bbox[:, (1, 3)], axis=1)
    bbox[:, 0] = x_min
    bbox[:, 1] = y_min
    bbox[:, 2] = x_max
    bbox[:, 3] = y_max
    return bbox.reshape(old_shape)


def bbox_affine_transform(data, affine_mat):
    """Transform bounding boxes by affine matrix.

    Args:
        data: np.ndarray. Bounding boxes to apply transformation.
            It should be in the shape of (N, 4+),
            where 4 represents (x1, y1, x2, y2).
        affine_mat: np.ndarray. The affine matrix.
            It should be in the shape of (2, 3).

    Returns:
        new_bbox: np.ndarray.
            The bounding boxes after transformation with shape (N, 4+)
    """
    assert isinstance(data, np.ndarray)
    if affine_mat.shape == (3, 3):
        affine_mat = affine_mat[0:2]
    assert affine_mat.shape == (2, 3)
    assert data.shape[-1] >= 4
    if len(data.shape) == 1:
        data = np.reshape(data, (1, data.shape[0]))
    else:
        assert len(data.shape) == 2
    new_data = np.empty(data.shape)
    new_data[:, :4] = point_2D_affine_transform(
        data[:, :4].reshape((-1, 2)), affine_mat
    ).reshape((-1, 4))
    new_data[:, 4:] = data[:, 4:]
    return _refine_bbox(new_data)
