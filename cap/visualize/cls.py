# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Union

import numpy
import torch
from matplotlib import pyplot as plt

from cap.registry import OBJECT_REGISTRY

__all__ = ["ClsViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ClsViz(object):
    """
    The viz method of classification result.

    Args:
        is_plot (bool): Whether to plot image.
    """

    def __init__(self, is_plot: bool = False):
        self.is_plot = is_plot

    def __call__(
        self,
        image: Union[torch.Tensor, numpy.ndarray],
        preds: torch.Tensor,
    ):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()

        if preds.shape[-1] > 1:
            preds = torch.argmax(preds, -1)
        logger.info(f"The result is: {preds}")

        if self.is_plot:
            plt.imshow(image)
            plt.show()
