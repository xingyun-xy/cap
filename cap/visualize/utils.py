# Copyright (c) Changan Auto. All rights reserved.
import copy
import random
from typing import List, Sequence, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor


def colormap(
    gray_image: Union[Tensor, np.ndarray, Image.Image],
    colormap: Union[Tensor, int],
    scale=1,
):
    """Append colormap on gray image.

    Args:
        gray_image: Input image for appending color map.
        colormap: Color map for image.
        scale: Scale of image.
    Returns:
        image: Image after colormap.
    """

    if isinstance(gray_image, Image.Image):
        gray_image = copy.deepcopy(gray_image)
        gray_image.flags.writeable = True
        gray_image = np.asarray(gray_image)
    if isinstance(gray_image, Tensor):
        assert isinstance(colormap, Tensor)
        return colormap.to(device=gray_image.device)[
            (gray_image * scale).to(dtype=torch.int64)
        ].squeeze()
    elif isinstance(gray_image, np.ndarray):
        assert isinstance(colormap, int)
        return cv2.applyColorMap(gray_image * scale, colormap)
    else:
        raise ValueError("Unsupported input type: %s" + str(type(gray_image)))


def show_images(images: Tensor, prefix: str, layout="chw", reverse_rgb=False):
    """
    Show the image from Tensor.

    Args:
        images: Images for showing.
        prefix: Prefix for showing window.
        layout: Layout of images.
        reverse_rgb: Whether to reverse channel of rgb.
    """

    if images.ndim == 4:
        if "chw" in layout:
            images = images.permute(0, 2, 3, 1)
        if reverse_rgb:
            images[:, :, :, (0, 1, 2)] = images[:, :, :, (2, 1, 0)]
        images = images.split(1)
        for i, image in enumerate(images):
            cv2.imshow(
                prefix + "_%d" % i,
                image.squeeze(0).detach().cpu().numpy().astype(np.uint8),
            )
    else:
        if "chw" in layout:
            images = images.permute(1, 2, 0)
        if reverse_rgb:
            images[:, :, (0, 1, 2)] = images[:, :, (2, 1, 0)]
        cv2.imshow(prefix, images.detach().cpu().numpy().astype(np.uint8))


def constructed_show(data, prefix, process):
    """
    Show constructed images.

    Args:
        data: Constructed images.
        prefix: Prefix for showing window.
        process: Process of images before showing.
    """

    if isinstance(data, dict):
        for k, v in data.items():
            constructed_show(v, prefix + "_" + str(k), process)
    elif isinstance(data, Sequence):
        for i, v in enumerate(data):
            constructed_show(v, prefix + "_" + str(i), process)
    elif isinstance(data, Tensor):
        process(data, prefix)
    else:
        raise TypeError("Visualization only accept dict/Sequence of Tensors")


def plot_image(img: np.array, ax=None, reverse_rgb=False):
    """Visualize image.

    Args:
        img: Image with shape `H, W, 3`.
        ax: You can reuse previous axes if provided.
        reverse_rgb: Reverse RGB<->BGR orders if `True`.
    Returns:
        The ploted axes.

    Examples:
        from matplotlib import pyplot as plt
        ax = plot_image(img)
        plt.show()
    """

    assert isinstance(img, np.ndarray)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax


def plot_bbox(
    img: np.array,
    bboxes: np.array,
    scores: np.array = None,
    labels: np.array = None,
    thresh: float = 0.5,
    class_names: List[str] = None,
    colors=None,
    ax=None,
    reverse_rgb=False,
    absolute_coordinates=True,
):
    """Visualize bounding boxes.

    Args:
        img: Image with shape `H, W, 3`.
        bboxes: Bounding boxes with shape `N, 4`.
            Where `N` is the number of boxes.
        scores: Confidence scores of the provided
            `bboxes` with shape `N`.
        labels: Class labels of the provided `bboxes` with shape `N`.
        thresh: Display threshold if `scores` is provided.
            Scores with less than `thresh` will be ignored
            in display, this is visually more elegant if you
            have a large number of bounding boxes with very small scores.
        class_names: Description of parameter `class_names`.
        colors: You can provide desired colors as
            {0: (255, 0, 0), 1:(0, 255, 0), ...},
            otherwise random colors will be substituted.
        ax: You can reuse previous axes if provided.
        reverse_rgb: Reverse RGB<->BGR orders if `True`.
        absolute_coordinates: If `True`, absolute coordinates
            will be considered, otherwise coordinates are
            interpreted as in range(0, 1).

    Returns:
        The ploted axes.
    """

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError(
            "The length of labels and bboxes mismatch, {} vs {}".format(
                len(labels), len(bboxes)
            )
        )
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError(
            "The length of scores and bboxes mismatch, {} vs {}".format(
                len(scores), len(bboxes)
            )
        )

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= height
        bboxes[:, (1, 3)] *= width

    # use random colors if None is provided
    if colors is None:
        colors = {}
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
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
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5,
        )
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ""
        score = "{:.3f}".format(scores.flat[i]) if scores is not None else ""
        if class_name or score:
            ax.text(
                xmin,
                ymin - 2,
                "{:s} {:s}".format(class_name, score),
                bbox={"facecolor": colors[cls_id], "alpha": 0.5},
                fontsize=12,
                color="white",
            )
    return ax
