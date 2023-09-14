# Copyright (c) Changan Auto. All rights reserved.

from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

__all__ = ["imresize", "random_flip", "image_pad", "image_normalize"]

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def imresize(
    img: np.ndarray,
    w,
    h,
    layout,
    divisor=1,
    keep_ratio=False,
    return_scale=False,
    interpolation="bilinear",
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Resize image with OpenCV.

    If keep_ratio=False, the image will be scaled to the maximum size that
    does not exceed wh and is divisible by divisor, otherwise resize shorter
    side to min(w, h) if the long side does not exceed max(w, h), otherwise
    resize the long side to max(w, h).

    Args:
        w (int): Width of resized image.
        h (int): Height of resized image.
        layout (str): Layout of img, `hwc` or `chw` or `hw`.
        divisor (int): Width and height are rounded to multiples of
            `divisor`, usually used in FPN-like structure.
        keep_ratio (bool): If True, resize img to target size while keeping w:h
            ratio.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method of image scaling, candidate
            value is ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'].

    Returns:
        tuple: resized_img + (w_scale, h_scale)
    """
    assert layout in ["hwc", "chw", "hw"]
    if layout == "hwc":
        img_h, img_w, _ = img.shape
    elif layout == "chw":
        _, img_h, img_w = img.shape
    else:
        img_h, img_w = img.shape

    if layout == "chw":
        # opencv only supports hwc layout
        img = np.ascontiguousarray(img.transpose((1, 2, 0)))  # chw > hwc
    if keep_ratio:
        short = min(w, h)
        max_size = max(w, h)
        im_size_min, im_size_max = (
            (img_h, img_w) if img_w > img_h else (img_w, img_h)
        )  # noqa
        scale = float(short) / float(im_size_min)
        if np.floor(scale * im_size_max / divisor) * divisor > max_size:
            # fit in max_size
            scale = (
                float(np.floor(max_size / divisor) * divisor) / im_size_max
            )  # noqa
        new_w, new_h = (
            int(np.floor(img_w * scale / divisor) * divisor),
            int(np.floor(img_h * scale / divisor) * divisor),
        )
    else:
        new_w, new_h = (
            int(np.floor(w / divisor) * divisor),
            int(np.floor(h / divisor) * divisor),
        )

    resized_img = cv2.resize(
        img, (new_w, new_h), interpolation=cv2_interp_codes[interpolation]
    )
    if layout == "chw":
        # change to the original layout
        resized_img = np.ascontiguousarray(resized_img.transpose((2, 0, 1)))
    w_scale = float(new_w / img_w)
    h_scale = float(new_h / img_h)
    if return_scale:
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def random_flip(
    img: Union[np.ndarray, torch.Tensor], layout, px=0, py=0
) -> Tuple[Union[np.ndarray, torch.Tensor], Tuple[bool, bool]]:
    """Randomly flip image along Horizontal and vertical with probabilities.

    Args:
        layout (str): Layout of img, `hwc` or `chw`.
        px (float): Horizontal flip probability, range between [0, 1].
        py (float): Vertical flip probability, range between [0, 1].

    Returns:
        tuple: flipped image + (flip_x, flip_y)

    """
    assert layout in ["hwc", "chw", "hw"]
    assert isinstance(img, (torch.Tensor, np.ndarray))
    h_index = layout.index("h")
    w_index = layout.index("w")
    flip_x = np.random.choice([False, True], p=[1 - px, px])
    flip_y = np.random.choice([False, True], p=[1 - py, py])

    if isinstance(img, np.ndarray):
        if flip_x:
            img = np.flip(img, axis=w_index)
        if flip_y:
            img = np.flip(img, axis=h_index)
    else:
        if flip_x:
            img = img.flip(w_index)
        if flip_y:
            img = img.flip(h_index)

    return img, (flip_x, flip_y)


def image_pad(
    img: Union[np.ndarray, torch.Tensor],
    layout,
    shape=None,
    divisor=1,
    pad_val=0,
) -> Union[np.ndarray, torch.Tensor]:
    """Pad image to a certain shape.

    Args:
        layout (str): Layout of img, `hwc` or `chw` or `hw`.
        shape (tuple): Expected padding shape, meaning of dimension is the
            same as img, if layout of img is `hwc`, shape must be (pad_h,
            pad_w) or (pad_h, pad_w, c).
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Union[float, Sequence[float]]): Values to be filled in padding
            areas, single value or a list of values with len c.
            E.g. : pad_val = 10, or pad_val = [10, 20, 30].

    Returns:
        ndarray or torch.Tensor: padded image.
    """
    assert layout in ["hwc", "chw", "hw"]
    if isinstance(pad_val, Sequence):
        assert layout in ["hwc", "chw"]
        c_index = layout.index("c")
        assert len(pad_val) == img.shape[c_index]
        pad_val = torch.tensor(pad_val)
        if layout == "hwc":
            pad_val = pad_val.unsqueeze(0).unsqueeze(0)
        elif layout == "chw":
            pad_val = pad_val.unsqueeze(-1).unsqueeze(-1)
        if isinstance(img, np.ndarray):
            pad_val = pad_val.numpy()

    # calculate pad_h and pad_h
    if shape is None:
        shape = img.shape
        if divisor == 1:
            return img
    assert len(shape) in [2, 3]
    if layout == "chw":
        if len(shape) == 2:
            h = max(img.shape[1], shape[0])
            w = max(img.shape[2], shape[1])
        else:
            h = max(img.shape[1], shape[1])
            w = max(img.shape[2], shape[2])
    else:
        h = max(img.shape[0], shape[0])
        w = max(img.shape[1], shape[1])
    pad_h = int(np.ceil(h / divisor)) * divisor
    pad_w = int(np.ceil(w / divisor)) * divisor

    if len(shape) == 3:
        if layout == "hwc":
            shape = (pad_h, pad_w, shape[-1])
        elif layout == "chw":
            shape = (shape[0], pad_h, pad_w)
    else:
        shape = (pad_h, pad_w)

    if len(shape) < len(img.shape):
        if layout == "hwc":
            shape = tuple(shape) + (img.shape[-1],)
        elif layout == "chw":
            shape = (img.shape[0],) + tuple(shape)
    assert len(shape) == len(img.shape)
    for i in range(len(shape)):
        assert shape[i] >= img.shape[i], (
            "padded shape must greater than " "the src shape of img"
        )

    if isinstance(img, Tensor):
        pad = torch.zeros(shape, dtype=img.dtype, device=img.device)
    elif isinstance(img, np.ndarray):
        pad = np.zeros(shape, dtype=img.dtype)
    else:
        raise TypeError
    pad[...] = pad_val

    if len(img.shape) == 2:
        pad[: img.shape[0], : img.shape[1]] = img
    elif layout == "hwc":
        pad[: img.shape[0], : img.shape[1], ...] = img
    else:
        pad[:, : img.shape[1], : img.shape[2]] = img
    return pad


def image_normalize(img: Union[np.ndarray, Tensor], mean, std, layout):
    """Normalize the image with mean and std.

    Args:
        mean (Union[float, Sequence[float]]): Shared mean or sequence of means
            for each channel.
        std (Union[float, Sequence[float]]): Shared std or sequence of stds for
            each channel.
        layout (str): Layout of img, `hwc` or `chw`.

    Returns:
        np.ndarray or torch.Tensor: Normalized image.

    """
    assert layout in ["hwc", "chw"]
    c_index = layout.index("c")

    return_ndarray = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.astype(np.float32))
        return_ndarray = True
    elif isinstance(img, Tensor):
        img = img.float()
    else:
        raise TypeError

    if isinstance(mean, Sequence):
        assert len(mean) == img.shape[c_index]
    else:
        mean = [mean] * img.shape[c_index]

    if isinstance(std, Sequence):
        assert len(std) == img.shape[c_index]
    else:
        std = [std] * img.shape[c_index]

    dtype = img.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
    std = torch.as_tensor(std, dtype=dtype, device=img.device)
    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, "
            "leading to division by zero.".format(dtype)
        )
    if c_index == 0:
        mean = mean[:, None, None]
        std = std[:, None, None]
    else:
        mean = mean[None, None, :]
        std = std[None, None, :]
    img.sub_(mean).div_(std)

    if return_ndarray:
        img = img.numpy()

    return img
