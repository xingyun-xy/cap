# Copyright (c) Changan Auto. All rights reserved.
import copy
from typing import Dict

import cv2
import numpy as np

from cap.core.data_struct.base_struct import DetBoxes3D
from .bbox2d import draw_bbox

__all__ = [
    "draw_bbox3d",
]


def compute_box_3d(dim, location, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
        3, 1
    )
    return corners_3d.transpose(1, 0)


def project_to_image(pts_3d, P, dist_coeff=None, fisheye=False):
    P = np.array(P)
    if dist_coeff is not None:
        if not fisheye:
            rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)
            dist_coeff = np.array(dist_coeff)
            image_pts = cv2.projectPoints(
                pts_3d[:, :3], np.array(rvec), tvec, P[:, :3], dist_coeff
            )[0]
            pts_2d = np.squeeze(image_pts)
        else:
            pts_3d = pts_3d[:, :3]
            pts_3d = copy.deepcopy(pts_3d)
            pts_3d[pts_3d[:, 2] < 0, 2] = 0.001
            pts_3d = np.expand_dims(pts_3d, 0)
            rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)
            dist_coeff = np.array(dist_coeff)
            fx, fy = P[0, 0], P[1, 1]
            u, v = P[0, 2], P[1, 2]
            k_ = np.mat([[fx, 0.0, u], [0.0, fy, v], [0.0, 0.0, 1.0]])
            d_ = np.mat(dist_coeff[:4].T)
            image_pts = cv2.fisheye.projectPoints(
                pts_3d, np.array(rvec), tvec, k_, d_
            )[0]
            pts_2d = np.squeeze(image_pts)
    else:
        pts_3d_homo = np.concatenate(
            [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1
        )
        pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d


def cal_truncation(box_2d, im_hw):
    bbox2 = np.array(
        [
            box_2d[:, 0].min(),
            box_2d[:, 1].min(),
            box_2d[:, 0].max(),
            box_2d[:, 1].max(),
        ]
    )
    bbox1 = np.array([0, 0, im_hw[1], im_hw[0]])

    # area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    inter = max(
        min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1, 0
    ) * max(min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + 1, 0)

    truncation = 1.0 - inter / area2
    return truncation


def _draw_box_3d(image, corners, c=(0, 0, 255), show_arrow=True, thickness=1):
    face_idx = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(
                image,
                (corners[f[j], 0], corners[f[j], 1]),
                (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]),
                c,
                thickness,
                lineType=cv2.LINE_AA,
            )

        if not show_arrow:
            if ind_f == 0:
                cv2.line(
                    image,
                    (corners[f[0], 0], corners[f[0], 1]),
                    (corners[f[2], 0], corners[f[2], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    image,
                    (corners[f[1], 0], corners[f[1], 1]),
                    (corners[f[3], 0], corners[f[3], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        # show an arrow to indicate 3D orientation of the object
        if show_arrow:
            # 4,5,6,7
            p1 = (
                corners[0, :] + corners[1, :] + corners[2, :] + corners[3, :]
            ) / 4
            p2 = (corners[0, :] + corners[1, :]) / 2
            p3 = p2 + (p2 - p1) * 0.5

            p1 = p1.astype(np.int32)
            p2 = p2.astype(np.int32)
            p3 = p3.astype(np.int32)

            cv2.line(
                image,
                (p1[0], p1[1]),
                (p3[0], p3[1]),
                c,
                thickness,
                lineType=cv2.LINE_AA,
            )
    return image


def project_3d_to_bird(pt, out_size, world_size):
    pt[:, 0] += world_size / 2
    pt[:, 1] = world_size - pt[:, 1]
    pt = pt * out_size / world_size
    return pt.astype(np.int32)


def draw_bev(
    location,
    dimension,
    yaw,
    color,
    canvas=None,
    out_size=192,
    world_size=32,
):
    if canvas is None:
        canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    pts_3d = compute_box_3d(dimension, location, yaw)
    rect = pts_3d[:4, [0, 2]]
    rect = project_3d_to_bird(rect, out_size, world_size)
    cv2.polylines(
        canvas,
        [rect.reshape(-1, 1, 2).astype(np.int32)],
        True,
        color,
        1,
        lineType=cv2.LINE_AA,
    )
    p1 = np.mean(rect, axis=0)
    p2 = (rect[0] + rect[1]) / 2
    p3 = p2 + (p2 - p1) / 2
    p1, p3 = p1.astype(np.int32), p3.astype(np.int32)

    cv2.line(
        canvas,
        (p1[0], p1[1]),
        (p3[0], p3[1]),
        color,
        1,
        lineType=cv2.LINE_AA,
    )

    return canvas


def draw_bbox3d(
    img,
    location,
    dimension,
    yaw,
    color,
    thickness,
    calib,
    distCoeffs,
    fisheye=False,
    truncation_thresh=0.3,
    min_depth_dist=0.8,
    bbox=None,
    draw_2d=False,
):
    pts_3d = compute_box_3d(dimension, location, yaw)
    pts_2d = project_to_image(
        pts_3d, calib, dist_coeff=distCoeffs, fisheye=fisheye
    )
    truncation = cal_truncation(pts_2d, img.shape[:2])

    if (
        truncation > truncation_thresh
        or (pts_3d[:, 2] <= min_depth_dist).any()
    ) and bbox is not None:
        img = draw_bbox(img, bbox, color, thickness)
    else:
        pts_2d = pts_2d.astype(np.int32)
        _draw_box_3d(img, pts_2d, color, thickness=thickness)
        if draw_2d:
            bbox2d = [
                pts_2d[:, 0].min(),
                pts_2d[:, 1].min(),
                pts_2d[:, 0].max(),
                pts_2d[:, 1].max(),
            ]
            img = draw_bbox(img, bbox2d, color, thickness)

    return img


def blend_top_right(img, blend, blend_factor=0.8):
    height, width = blend.shape[:2]
    img[:height, -width:] = (
        img[:height, -width:] * (1 - blend_factor) + blend * blend_factor
    )
    return img.astype(np.uint8)


def vis_det_boxes_3d(
    vis_image: np.ndarray,
    calib: np.ndarray,
    distCoeffs: np.ndarray,
    det_boxes_3d: DetBoxes3D,
    vis_configs: Dict,
):
    color = vis_configs["color"]
    thickness = vis_configs["thickness"]

    for det_box_3d in iter(det_boxes_3d):
        vis_image = draw_bbox3d(
            vis_image,
            det_box_3d.location.numpy(),
            det_box_3d.dimension.numpy(),
            det_box_3d.yaw.item(),
            color,
            thickness,
            calib,
            distCoeffs,
        )
    return vis_image


def vis_det_boxes_3d_bev(
    det_boxes_3d: DetBoxes3D,
    vis_configs: Dict,
):
    canvas = None
    for det_box_3d in iter(det_boxes_3d):
        canvas = draw_bev(
            det_box_3d.location.numpy(),
            det_box_3d.dimension.numpy(),
            det_box_3d.yaw.item(),
            vis_configs.get("color", vis_configs["color"]),
            canvas=canvas,
        )

    return canvas
