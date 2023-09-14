# Copyright (c) Changan Auto. All rights reserved.
import copy
import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from cap.core.data_struct.base_struct import DetBoxes2D, MultipleBoxes2D
from cap.core.data_struct.img_structures import ImgObjDet
from cap.utils.apply_func import convert_numpy

__all__ = [
    "draw_bbox2d",
    "draw_bbox",
    "draw_text",
]


def draw_bbox2d(
    det_object: List[ImgObjDet],
    scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
    score_thresh: float = 0.5,
    class_names: List[str] = None,
    colors: Optional[Dict[int, Tuple[int]]] = None,
    thickness: int = 2,
    scale: float = 1.0,
):
    """
    Visualize bounding boxes 2d.

    Args:
        det_object: List of det objects.
        scores: Scores of det objects.
        score_thresh: Score thresh for plotting.
        class_names: Names of classes.
        colors: Colors of classes.
        thickness: Thickness for cv2 plotting.
        scale: Scale for cv2 plotting.
    """

    if plt is None:
        raise ModuleNotFoundError("matplotlib is required")

    img_bgr = convert_numpy(det_object.img)
    if det_object.layout == "chw":
        img_bgr = img_bgr.transpose((1, 2, 0))
    if det_object.color_space == "rgb":
        img_bgr = img_bgr[:, :, ::-1].copy()

    gt_bboxes = convert_numpy(det_object.gt_bboxes)
    labels = gt_bboxes[:, 4]
    bboxes = gt_bboxes[:, :4]
    if scores is not None:
        scores = convert_numpy(scores)
        assert len(scores) == len(bboxes)

    if len(bboxes) < 1:
        return img_bgr

    if colors is None:
        colors = {}
    else:
        colors = copy.copy(colors)

    for idx, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[idx] < score_thresh:
            continue
        if labels is not None and labels.flat[idx] < 0:
            continue
        if labels is not None:
            cls_id = int(labels.flat[idx])
        else:
            cls_id = -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap("hsv")(cls_id / len(class_names))
            else:
                colors[cls_id] = (
                    random.random(),
                    random.random(),
                    random.random(),
                )
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        bcolor = [x * 255 for x in colors[cls_id]]
        cv2.rectangle(
            img_bgr, (xmin, ymin), (xmax, ymax), bcolor, int(thickness)
        )
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ""
        score = "%.2f" % scores.flat[idx] if scores is not None else ""
        if class_name or score:
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(
                img_bgr,
                "{:s} {:s}".format(class_name, score),
                (xmin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                min(scale / 2, 2),
                bcolor,
                min(int(scale), 5),
                lineType=cv2.LINE_AA,
            )
    return img_bgr


def draw_bbox(img, bbox, color, thickness):
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, int(thickness))
    return img


def draw_text(img, text, coord, color, scale):
    cv2.putText(
        img,
        text,
        coord,
        cv2.FONT_HERSHEY_SIMPLEX,
        min(scale / 2, 2),
        color,
        min(int(scale), 5),
        lineType=cv2.LINE_AA,
    )
    return img


def vis_det_boxes_2d(
    vis_image: np.ndarray, det_boxes_2d: DetBoxes2D, vis_configs: Dict
):
    color = vis_configs["color"]
    thickness = vis_configs["thickness"]
    for det_box_2d in iter(det_boxes_2d):
        box = det_box_2d.box.numpy()
        vis_image = draw_bbox(
            vis_image,
            box,
            color,
            thickness,
        )
    return vis_image

def vis_det_boxes_2d_waic(
    vis_image: np.ndarray, det_boxes_2d: DetBoxes2D, vis_configs: Dict
):
    color = vis_configs["color"]
    thickness = vis_configs["thickness"]
    for det_box_2d in iter(det_boxes_2d):
        box = det_box_2d[:4]
        vis_image = draw_bbox(
            vis_image,
            box,
            color,
            thickness,
        )
    return vis_image


def vis_multiple_boxes_2d(
    vis_image: np.ndarray,
    multiple_boxes_2d: MultipleBoxes2D,
    vis_configs: Dict,
):
    for multiple_box_2d in iter(multiple_boxes_2d):
        vis_image = vis_det_boxes_2d(vis_image, multiple_box_2d, vis_configs)
    return vis_image
