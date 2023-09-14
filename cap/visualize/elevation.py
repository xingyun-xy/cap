# Copyright (c) Changan Auto. All rights reserved.

import os
from typing import Union

import cv2
import numpy as np
import torch

__all__ = [
    "vis_parallax",
]

GAMMA_RANGE = (0, 0.03)
DEPTH_RANGE = (0, 50)
HEIGHT_RANGE = (0, 0.5)


def vis_parallax(
    pred: Union[torch.Tensor, np.ndarray],
    img: Union[torch.Tensor, np.ndarray],
    save: bool = False,
    path: str = "",
    timestamp: str = "",
    vis_type: str = "gamma",
) -> np.ndarray:
    """Visulalize prediction result.

    Args:
        pred: pred of depth, height or gamma.
        img: color img.
        save: whether save the vis image.
        path: save path of vis img.
        timestamp: img timestamp.
        vis_type:
            vis type, the options are 'depth', 'height' and 'gamma'.
    """
    VIS_RANGE = {
        "gamma": GAMMA_RANGE,
        "height": HEIGHT_RANGE,
        "depth": DEPTH_RANGE,
    }
    VIS_MIN, VIS_MAX = VIS_RANGE[vis_type]
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).clone().detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(pred, torch.Tensor):
        pred = np.abs(
            pred.squeeze(0).clone().detach().cpu().numpy().transpose(1, 2, 0)
        )
        scale = 255
    else:
        pred = pred.squeeze(0).transpose(1, 2, 0)
        scale = 1

    h, w, _ = img.shape
    pred = cv2.resize(pred, (w, h))[:, :, np.newaxis]
    pred = pred[h // 2 :, :, :]
    pred = pred.clip(VIS_MIN, VIS_MAX)
    color_gt = (pred) / (VIS_MAX) * 255
    color_gt = color_gt.astype("uint8")
    color_gt = cv2.applyColorMap(color_gt, cv2.COLORMAP_JET)

    zeros = np.zeros((h, w, 3))
    zeros[h // 2 :, :, :] = color_gt
    mask = zeros
    alpha, beta, gamma = 1.0, 0.5, 0
    img = cv2.cvtColor(img * scale, cv2.COLOR_RGB2BGR)

    img = np.asarray(img, np.float64)
    mask = np.asarray(mask, np.float64)
    vis_img = cv2.addWeighted(img, alpha, mask, beta, gamma)
    vis_img = cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5)
    if save:
        cv2.imwrite(
            os.path.join(path, timestamp + "_" + vis_type + ".jpg"), vis_img
        )
    return vis_img
