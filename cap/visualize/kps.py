# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict

import cv2
import numpy as np

from cap.core.data_struct.base import BaseDataList
from cap.core.data_struct.base_struct import Lines2D

__all__ = [
    "draw_kps",
]


POINT_COLOR_SET = (
    (0, 128, 255),
    (128, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (61, 61, 255),
    (0, 255, 128),
    (255, 0, 0),
    (122, 122, 255),
    (0, 255, 0),
    (255, 0, 128),
    (255, 255, 61),
    (128, 255, 0),
    (255, 0, 255),
    (255, 255, 122),
    (79, 62, 213),
    (188, 172, 113),
    (255, 255, 0),
    (67, 109, 244),
    (165, 194, 102),
    (97, 174, 253),
    (152, 245, 230),
    (139, 224, 254),
    (191, 255, 255),
    (164, 221, 171),
    (170, 221, 136),
    (155, 182, 55),
    (189, 136, 50),
)


def draw_line(img, points, color, radius, thickness):
    assert len(points) == 2
    for i, point in enumerate(points):
        img = draw_kp(
            img,
            point,
            radius,
            thickness,
            POINT_COLOR_SET[i % len(POINT_COLOR_SET)],
            with_score=False,
        )

    cv2.line(
        img,
        tuple(points[0]),
        tuple(points[1]),
        color=color,
        thickness=thickness,
    )

    return img


def draw_kp(
    img, point, radius, thickness, color, score=None, with_score=False
):
    assert len(point) == 2
    x, y = [int(p) for p in point]

    cv2.circle(
        img,
        center=(x, y),
        radius=radius,
        color=color,
        thickness=int(thickness),
    )

    if with_score:
        assert score is not None
        cv2.putText(
            img,
            str("%.2f" % score),
            (max(x - 20, 0), max(y - 20, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return img


def draw_kps(img, points, radius, thickness, scores=None):
    for i, point in enumerate(points):
        img = draw_kp(
            img,
            point,
            radius,
            thickness,
            POINT_COLOR_SET[i % len(POINT_COLOR_SET)],
            score=None if scores is None else scores[i],
        )
    return img


def vis_n_points_2d(
    vis_image: np.ndarray,
    n_points_2d: BaseDataList,
    vis_configs: Dict,
):
    for n_point_2d in iter(n_points_2d):

        coords, scores = [], []
        for i in range(n_point_2d.num_points):
            point_i = getattr(n_point_2d, f"point{i}")
            coords.append(point_i.point.numpy().astype(int))
            scores.append(point_i.score.item())

        vis_image = draw_kps(
            vis_image,
            coords,
            radius=vis_configs.get("radius", 4),
            thickness=vis_configs.get("thickness", 2),
            scores=scores if vis_configs.get("with_score", True) else None,
        )

    return vis_image


def vis_lines_2d(
    vis_image: np.ndarray,
    lines_2d: Lines2D,
    vis_configs,
):
    color = vis_configs["color"]
    thickness = vis_configs["thickness"]
    for line_2d in iter(lines_2d):
        cv2.line(
            vis_image,
            tuple(line_2d.point0.point.numpy().astype(int)),
            tuple(line_2d.point1.point.numpy().astype(int)),
            color=color,
            thickness=thickness,
        )
    return vis_image
