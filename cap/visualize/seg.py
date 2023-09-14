# Copyright (c) Changan Auto. All rights reserved.
from typing import Callable, Union

import cv2
import numpy
import torch
from torch import Tensor

from cap.registry import OBJECT_REGISTRY
from .utils import colormap, constructed_show, show_images

__all__ = ["SegViz"]

_cityscapes_colormap = torch.tensor(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ]
)


def _default_image_process(image: Tensor, prefix):
    show_images(image, prefix)


def _seg_target_process(target: Tensor, prefix):
    if target.size(1) > 1:
        one_hot = torch.argmax(target, dim=1, keepdim=True)
    else:
        one_hot = target
    one_hot = colormap(one_hot, _cityscapes_colormap)
    show_images(one_hot, prefix, "hwc")


@OBJECT_REGISTRY.register
class SegViz(object):
    """
    The visualize method of segmentation result.

    Args:
        image_process (Callable): Process of image.
        label_process (Callable): Process of label.
    """

    def __init__(
        self,
        image_process: Callable = _default_image_process,
        label_process: Callable = _seg_target_process,
    ):
        self.image_process = image_process
        self.label_process = label_process

    def __call__(
        self,
        output: Tensor,
        image: Union[numpy.ndarray, Tensor],
    ):
        constructed_show(image, "image", self.image_process)
        constructed_show(output, "labels", self.label_process)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            return
