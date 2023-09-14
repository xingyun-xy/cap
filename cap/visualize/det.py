# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Union

import numpy
import torch
from matplotlib import pyplot as plt

from cap.registry import OBJECT_REGISTRY
from .utils import plot_bbox

__all__ = ["DetViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DetViz(object):
    """
    The visiualize method of det result.

    Args:
        score_thresh (float): Score thresh for filtering box in plot.
        absolute_coord (bool): Coordinates of pred_box is absolute or not.
        is_plot (bool): Whether to plot image.
    """

    def __init__(
        self,
        score_thresh: float = 0.5,
        absolute_coord: bool = True,
        is_plot: bool = True,
    ):
        self.score_thresh = score_thresh
        self.absolute_coord = absolute_coord
        self.is_plot = is_plot

    def __call__(
        self,
        image: Union[torch.Tensor, numpy.ndarray],
        detections: Union[torch.Tensor, numpy.ndarray],
    ):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()

        det = detections.cpu().numpy()
        pred_label = det[:, -1]
        pred_score = det[:, -2]
        pred_bbox = det[:, 0:4]

        logger.debug(
            f"pred_labels: {pred_label}; \n"
            + f"pred_scores: {pred_score}; \n"
            + f"pred_bboxes: {pred_bbox}; \n"
        )

        plot_bbox(
            img=image,
            bboxes=pred_bbox,
            labels=pred_label,
            scores=pred_score,
            thresh=self.score_thresh,
            absolute_coordinates=self.absolute_coord,
        )

        if self.is_plot:
            plt.show()
