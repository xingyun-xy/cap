# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, List

import cv2
import numpy as np

__all__ = [
    "draw_grid",
    "draw_label",
    "draw_reg",
    "draw_online_mapping",
    "get_bev_pts",
]


category_color_dict = {
    "background": (0, 0, 0),
    "solid_lanes": (255, 255, 255),
    "roadedges": (0, 0, 255),
    "crosswalks": (0, 255, 0),
    "TP_GT": (0, 255, 0),
    "TP_PRED": (255, 0, 0),
    "FN_GT": (0, 255, 255),
    "FP_PRED": (0, 0, 255),
}

instance_colors = (np.random.rand(32, 3) * 256).astype(np.uint8)


def put_text_on_image(
    img,
    info: List[Dict[str, float]],
    start_x=10,
    start_y=20,
    vertical_margin=20,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.5,
    thickness=1,
    colors=None,
):
    """Put text on image."""
    if not colors:
        colors = [(255, 255, 255) for _ in range(len(info))]
    for idx, line in enumerate(info):
        _str = ""
        for k, v in line.items():
            if isinstance(v, float):
                _str += f" {k}: {v:.3f}"
            else:
                _str += f" {k}: {v}"
        pos = (int(start_x), int(start_y + vertical_margin * idx))
        cv2.putText(
            img,
            _str,
            pos,
            fontFace=fontFace,
            fontScale=fontScale,
            color=colors[idx],
            thickness=thickness,
        )


def draw_grid(
    img,
    pxstep_x=8,
    pxstep_y=8,
    line_color=(0, 255, 0),
    thickness=1,
    type_=cv2.LINE_AA,
):
    """Draw grid on image."""
    x = pxstep_x - 1
    y = pxstep_y - 1
    while x < img.shape[1]:
        cv2.line(
            img,
            (x, 0),
            (x, img.shape[0]),
            color=line_color,
            lineType=type_,
            thickness=thickness,
        )
        x += pxstep_x

    while y < img.shape[0]:
        cv2.line(
            img,
            (0, y),
            (img.shape[1], y),
            color=line_color,
            lineType=type_,
            thickness=thickness,
        )
        y += pxstep_y
    return img


def draw_label(
    label,
    prob=None,
    text=None,
    out_width=512,
    out_height=512,
    color_by_category=False,
    target_categorys=None,
):
    """Draw label on image."""
    if prob is None:
        prob = np.ones_like(label)
    if color_by_category:
        assert target_categorys is not None
        idx2category = {v: k for k, v in target_categorys.items()}
        idx2category.update({0: "background"})
    h, w = label.shape
    img = np.zeros((h, w, 3))
    ids = np.unique(label)
    for idx in ids:
        color = (
            category_color_dict[idx2category[idx]]
            if color_by_category
            else list(np.random.random(size=3) * 256)
        )
        mask = label == idx
        prob_ = prob[mask]
        for i in range(3):
            img[mask, i] = prob_ * color[i]
    img = img.astype(np.uint8)
    img = cv2.resize(img, (out_width, out_height))
    if text:
        cv2.putText(
            img,
            text,
            (50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 255, 255),
            thickness=2,
        )
    return img


def draw_reg(
    reg, direction=False, prob=None, text=None, out_width=512, out_height=512
):
    """Draw reg."""
    if prob is None:
        prob = np.ones_like(reg)
    colormap = cv2.COLORMAP_HSV if direction else cv2.COLORMAP_RAINBOW
    img = cv2.applyColorMap((reg * 255).astype(np.uint8), colormap)
    img = img * np.expand_dims(prob, axis=2)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (out_width, out_height))
    if text:
        cv2.putText(
            img,
            text,
            (50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 255, 255),
            thickness=2,
        )
    return img


def draw_direction(
    sin,
    cos,
    prob=None,
    text=None,
    out_width=512,
    out_height=512,
    normalize=False,
):
    """Draw direction map.

    Args:
        sin (_type_): sin value map
        cos (_type_): cos value map
        prob (_type_, optional): equal view mask. Defaults to None.
        text (_type_, optional): descriptive text. Defaults to None.
        out_width (int, optional): Defaults to 512.
        out_height (int, optional): Defaults to 512.
        normalize (bool, optional): true, angle alpha and alpha+180 are equal.

    Returns:
       direction view color img
    """
    if prob is None:
        prob = np.ones_like(sin)
    init_h, init_w = sin.shape
    ang = np.degrees(np.arctan2(sin, cos)) + 180.0
    if normalize:
        ang = ang % 180 * 2
    hsv = np.zeros((init_h, init_w, 3))
    hsv[..., 0] = ang
    hsv[..., 1] = 255.0
    hsv[..., 2] = 255.0
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img * np.expand_dims(prob, axis=2)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (out_width, out_height))
    if text:
        cv2.putText(
            img,
            text,
            (50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 255, 255),
            thickness=2,
        )
    return img


def draw_online_mapping(
    online_mapping,
    bev_h,
    bev_w,
    mat_vcs2bev,
    text=None,
    crosswalk_rec=False,
    draw_polyline_crosswalk=False,
    draw_polyline=False,
    draw_pt_crosswalk=False,
    draw_pt=False,
    draw_line_crosswalk=False,
    draw_line=False,
    draw_segment=False,
    draw_idx=False,
    draw_start_end_pt=False,
    thickness=1,
    pt_thickness=None,
    img_bev=None,
    color_by_category=False,
):
    """Draw online mapping."""
    meter_per_out_pixel = 1.6
    stride = 8
    delta_t = meter_per_out_pixel / 4.0
    dist_threshold = meter_per_out_pixel * stride * 2

    pt_thickness = thickness * 2 if pt_thickness is None else pt_thickness
    if img_bev is None:
        img_bev = np.zeros((bev_h, bev_w, 3))
    else:
        h, w, _ = img_bev.shape
        assert h == bev_h and w == bev_w
    if text:
        cv2.putText(
            img_bev,
            text,
            (50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 255, 255),
            thickness=2,
        )

    bev_ego_pts = (mat_vcs2bev @ np.array([[0], [0], [1]])).transpose()
    bev_ego_pts[:, 0] = bev_ego_pts[:, 0] / bev_ego_pts[:, 2]
    bev_ego_pts[:, 1] = bev_ego_pts[:, 1] / bev_ego_pts[:, 2]
    bev_ego_pts = bev_ego_pts[0, :2]
    cv2.rectangle(
        img_bev,
        (int(bev_ego_pts[0]) - 5, int(bev_ego_pts[1]) - 10),
        (int(bev_ego_pts[0]) + 5, int(bev_ego_pts[1]) + 10),
        color=(0, 255, 0),
        thickness=thickness,
    )
    for (category, lanes) in online_mapping.items():
        if crosswalk_rec and category == "crosswalks":
            for lane in lanes:
                if len(lane) <= 10:
                    continue
                color = (
                    category_color_dict[category]
                    if color_by_category
                    else list(np.random.random(size=3) * 256)
                )
                bev_pts = get_bev_pts(lane, mat_vcs2bev)
                rect = cv2.minAreaRect(np.int32([bev_pts]))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img_bev, [box], 0, color, 2)
            continue
        for idx, lane in enumerate(lanes):
            color = (
                category_color_dict[category]
                if color_by_category
                else list(np.random.random(size=3) * 256)
            )
            if isinstance(lane, tuple):
                color = instance_colors[lane[1] % 32, :].tolist()
                lane = lane[0]
            bev_pts = get_bev_pts(lane[:, :2], mat_vcs2bev)
            if draw_idx:
                img_bev = cv2.putText(
                    img_bev,
                    str(idx),
                    (int(bev_pts[0, 0]), int(bev_pts[0, 1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=color,
                    thickness=1,
                )
                img_bev = cv2.putText(
                    img_bev,
                    str(idx),
                    (int(bev_pts[-1, 0]), int(bev_pts[-1, 1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=color,
                    thickness=1,
                )
            if draw_polyline and (
                category != "crosswalks" or draw_polyline_crosswalk
            ):
                img_bev = cv2.polylines(
                    img_bev,
                    np.int32([bev_pts]),
                    isClosed=False,
                    color=color,
                    thickness=thickness,
                )
            if draw_line and (category != "crosswalks" or draw_line_crosswalk):
                bev_pts_last = bev_pts[:-1]
                bev_pts_next = bev_pts[1:]
                for bev_pt_last, bev_pt_next in zip(
                    bev_pts_last, bev_pts_next
                ):
                    if (
                        np.linalg.norm(bev_pt_last - bev_pt_next)
                        > dist_threshold
                    ):
                        continue
                    img_bev = cv2.line(
                        img_bev,
                        (int(bev_pt_last[0]), int(bev_pt_last[1])),
                        (int(bev_pt_next[0]), int(bev_pt_next[1])),
                        color,
                        thickness=thickness * 2,
                    )
            if draw_pt or (category == "crosswalks" and draw_pt_crosswalk):
                for pt in bev_pts:
                    cv2.circle(
                        img_bev,
                        (int(round(pt[0])), int(round(pt[1]))),
                        radius=1,
                        color=color,
                        thickness=pt_thickness,
                    )
            if draw_start_end_pt:
                cv2.circle(
                    img_bev,
                    (int(round(bev_pts[0][0])), int(round(bev_pts[0][1]))),
                    radius=3,
                    color=(0, 255, 255),
                    thickness=thickness * 2,
                )  # start point yellow
                cv2.circle(
                    img_bev,
                    (int(round(bev_pts[-1][0])), int(round(bev_pts[-1][1]))),
                    radius=3,
                    color=(255, 0, 0),
                    thickness=thickness * 2,
                )  # end point blue
            if draw_segment:
                vcs_pts_plus = np.hstack(
                    [
                        lane[:, 0:1] - delta_t * lane[:, 4:5],
                        lane[:, 1:2] + delta_t * lane[:, 5:6],
                    ]
                )
                vcs_pts_minus = np.hstack(
                    [
                        lane[:, 0:1] + delta_t * lane[:, 4:5],
                        lane[:, 1:2] - delta_t * lane[:, 5:6],
                    ]
                )
                bev_pts_plus = get_bev_pts(vcs_pts_plus, mat_vcs2bev)
                bev_pts_minus = get_bev_pts(vcs_pts_minus, mat_vcs2bev)
                for pt_plus, pt_minus in zip(bev_pts_plus, bev_pts_minus):
                    img_bev = cv2.line(
                        img_bev,
                        (int(pt_plus[0] + 0.5), int(pt_plus[1] + 0.5)),
                        (int(pt_minus[0] + 0.5), int(pt_minus[1] + 0.5)),
                        color,
                        thickness=2,
                    )
    return img_bev


def get_bev_pts(lane, mat_vcs2bev):
    """Get bev points."""
    lane = np.array([[pt[0], pt[1], 1] for pt in lane]).transpose()
    bev_pts = (mat_vcs2bev @ lane).transpose()
    bev_pts[:, 0] = bev_pts[:, 0] / bev_pts[:, 2]
    bev_pts[:, 1] = bev_pts[:, 1] / bev_pts[:, 2]
    bev_pts = bev_pts[:, :2]
    return bev_pts
