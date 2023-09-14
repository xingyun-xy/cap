# Copyright (c) Changan Auto. All rights reserved.
import warnings
from typing import Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from torch import Tensor

from cap.registry import OBJECT_REGISTRY
from cap.visualize.utils import plot_image

__all__ = ["FlowViz"]


def flow_to_img(
    flow, normalize=True, info=None, flow_mag_max=None
):  # noqa: D205,D400
    """
    Convert flow to viewable image, using color hue to encode
    flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial
    on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of
    the saturation plane, as we do here.

    - OpenCV 3.0.0-dev documentation » OpenCV-Python
    Tutorials » Video Analysis »
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video\
    /py_lucas_kanade/py_lucas_kanade.html.

    Args:
        flow: optical flow.
        normalize: Normalize flow to 0..255.
        info: Text to superimpose on image (typically,\
        the epe for the predicted flow).
        flow_mag_max: Max flow to map to 255.
    Returns:
        img: viewable representation of the dense optical flow in RGB format.
        flow_avg: optionally, also return average flow magnitude.

    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(
        flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)
    )

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.0

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(
                flow_magnitude, None, 0, 255, cv2.NORM_MINMAX
            )
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return img


def flow_write(flow, dst_file):
    """Write optical flow to a .flo file.

    Args:
        flow: optical flow.
        dst_file: Path where to write optical flow.
    """
    TAG_FLOAT = 202021.25
    # Save optical flow to disk
    with open(dst_file, "wb") as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)


def flow_write_as_png(flow, dst_file, info=None, flow_mag_max=None):
    """Write optical flow to a .PNG file.

    Args:
        flow: optical flow.
        dst_file: Path where to write optical flow as a .PNG file.
        info: Text to superimpose on image (typically, the epe
        for the predicted flow).
        flow_mag_max: Max flow to map to 255.
    """
    # Convert the optical flow field to RGB
    img = flow_to_img(flow, flow_mag_max=flow_mag_max)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Save RGB version of optical flow to disk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(dst_file, img)


@OBJECT_REGISTRY.register
class FlowViz(object):
    """
    The visualize method of Flow_pred result.

    Args:
        is_plot: Whether to plot image.
    """

    def __init__(
        self,
        is_plot: bool = True,
    ):
        self.is_plot = is_plot

    def __call__(
        self,
        image: Union[np.ndarray, Tensor],
        output: Union[np.ndarray, Tensor],
    ):

        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        if isinstance(output, torch.Tensor):
            output = output.squeeze().cpu().numpy()

        flow_img = flow_to_img(output)
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("image1")
        ax = plot_image(image[..., :3], ax=ax)
        ax = fig.add_subplot(1, 3, 2)
        ax.set_title("image2")
        ax = plot_image(image[..., 3:], ax=ax)
        ax = fig.add_subplot(1, 3, 3)
        ax.set_title("opticalflow")
        ax = plot_image(flow_img, ax=ax)
        if self.is_plot:
            plt.show()
