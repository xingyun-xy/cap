# -*- coding: utf-8 -*-
# Copyright (c) Changan Auto. All rights reserved.

import copy
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

from cap.registry import OBJECT_REGISTRY
from .classification import BgrToYuv444

__all__ = [
    "ResizeElevation",
    "CropElevation",
    "ToTensorElevation",
    "NormalizeElevation",
    "PrepareDataElevation",
]

PIL_INTERP_CODES = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
}


def get_front_mask(seg):
    front_mask = (seg != 20).float()
    return front_mask


def get_ground_mask(seg):
    ground_mask = (
        (seg == 0)
        | (seg == 7)
        | (seg == 8)
        | (seg == 24)
        | (seg == 25)
        | (seg == 26)
        | (seg == 27)
    ).float()
    return ground_mask


def get_obj_mask(seg):
    obj_mask = (seg <= 17) * (seg >= 9).float()
    return obj_mask


def get_mask(seg: torch.Tensor, maks_type: str = None):
    """Get segmentation mask.

    Args:
        seg: tensor of segmentation map.
        maks_type: the mask type, Default is 'None', other choices
            contain 'ground', 'obj', 'fornt' .
    """
    if maks_type is None:
        """seg mask:
        0 is sky
        2 is obj
        1 is others
        """
        seg[(seg >= 9) & (seg <= 17)] = 40  # obj
        seg[seg == 20] = 41  # sky
        # seg[seg==2] = 41 # tree

        seg[seg < 40] = 1  # others
        seg[seg == 41] = 0  # sky
        seg[seg == 40] = 2  # obj
    elif maks_type == "obj":
        seg = get_obj_mask(seg)
    elif maks_type == "front":
        seg = get_front_mask(seg)
    elif maks_type == "ground":
        seg = get_ground_mask(seg)
    return seg


@OBJECT_REGISTRY.register
class ResizeElevation(object):
    """Resize PIL Images to the given size and modify intrinsics.

    Args:
        size: Desired output size. If size is a
            sequence like (h, w), output size will be matched to this.
        interpolation: Desired interpolation. Default is 'nearest'.
    """

    def __init__(
        self,
        size: Optional[Sequence] = None,
        interpolation: str = "nearest",
    ):
        self.size = size if isinstance(size[0], Sequence) else [size]
        assert interpolation in PIL_INTERP_CODES
        self.interpolation = interpolation

    def _resize(self, data: Union[Image.Image, Sequence], size, interpolation):
        if isinstance(data, Sequence):
            assert len(data) == len(size)
            return [
                self._resize(data_i, size, interpolation)
                for data_i, size in zip(data, size)
            ]
        else:
            return F.resize(data, size, interpolation)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'
        if len(data["pil_imgs"][0]) != len(self.size):
            duplicate = len(data["pil_imgs"][0]) / len(self.size)
            self.size = self.size * int(duplicate)

        data["pil_imgs"] = [
            self._resize(
                pil_img, self.size, PIL_INTERP_CODES[self.interpolation]
            )
            for pil_img in data["pil_imgs"]
        ]
        data["size"] = copy.deepcopy(self.size)
        image_height = self.size[0][0]
        image_width = self.size[0][1]

        if "gt_depth" in data:
            data["gt_depth"] = [
                cv2.resize(
                    depth.squeeze(),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                for depth in data["gt_depth"]
            ]

        if "gt_height" in data:
            data["gt_height"] = [
                cv2.resize(
                    height.squeeze(),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                for height in data["gt_height"]
            ]

        if "gt_gamma" in data:
            data["gt_gamma"] = [
                cv2.resize(
                    gamma.squeeze(),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                for gamma in data["gt_gamma"]
            ]

        if "mask" in data:
            data["mask"] = [
                self._resize(mask, self.size, Image.NEAREST)
                for mask in data["mask"]
            ]

        if "intrinsics" in data:
            # scale intrinsics matrix
            data["intrinsics"][0, :] *= self.size[0][1]
            data["intrinsics"][1, :] *= self.size[0][0]

        return data

    def __repr__(self):
        return "ResizeElevation"


@OBJECT_REGISTRY.register
class CropElevation(object):
    """Crop the images and modify intrinsics/homography matrix.

    The image can be a PIL Image or a Tensor,
    in which case it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        height: Height of the crop box.
        width: Width of the crop box.
        top: Vertical component of the top left corner of the
            crop box. Setting None means random crop.
        left: Horizontal component of the top left corner of
            the crop box. Setting None means random crop.
        resized: whether resize gt before crop, default is 'True'
    """

    def __init__(
        self,
        height: Union[int, Sequence],
        width: Union[int, Sequence],
        top: Optional[int] = None,
        left: Optional[int] = None,
        resized: bool = True,
    ):
        self.top = top if isinstance(top, Sequence) or top is None else [top]
        self.left = (
            left if isinstance(left, Sequence) or left is None else [left]
        )
        self.height = height
        self.width = width
        self.resized = resized

    def _crop(
        self, data: Union[Image.Image, Sequence], top, left, height, width
    ):
        if isinstance(data, Sequence):
            assert len(data) == len(top) and len(data) == len(left)
            return [
                self._crop(_, top, left, height, width)
                for _, top, left in zip(data, top, left)
            ]
        else:
            return F.crop(data, top, left, height, width)

    def __call__(self, data: Mapping):

        for size in data["size"]:
            assert size[0] >= self.height
            assert size[1] >= self.width

        # generate a random integer if self.top is None
        top = (
            [
                np.random.randint(size[0] - self.height + 1)
                for size in data["size"]
            ]
            if self.top is None
            else self.top
        )

        # generate a random integer if self.left is None
        left = (
            [
                np.random.randint(size[1] - self.width + 1)
                for size in data["size"]
            ]
            if self.left is None
            else self.left
        )

        assert "pil_imgs" in data, 'input data must has "pil_imgs"'

        if len(data["pil_imgs"][0]) != len(top):
            duplicate = len(data["pil_imgs"][0]) / len(top)
            top = top * int(duplicate)

        if len(data["pil_imgs"][0]) != len(left):
            duplicate = len(data["pil_imgs"][0]) / len(left)
            left = left * int(duplicate)

        data["pil_imgs"] = [
            self._crop(pil_img, top, left, self.height, self.width)
            for pil_img in data["pil_imgs"]
        ]

        if "mask" in data:
            data["mask"] = [
                self._crop(mask, top, left, self.height, self.width)
                for mask in data["mask"]
            ]  # [[m1],[m2],[m3]]

        if self.resized and "gt_depth" in data:
            scale = int(
                data["gt_depth"][0].shape[0] / data["pil_imgs"][0][0].size[1]
            )
        else:
            scale = 1
        top_gt = top[0] * scale
        left_gt = left[0] * scale
        h_gt = self.height * scale
        w_gt = self.width * scale

        if "gt_depth" in data:
            data["gt_depth"] = [
                depth[top_gt : top_gt + h_gt, left_gt : left_gt + w_gt]
                for depth in data["gt_depth"]
            ]

        if "gt_height" in data:
            data["gt_height"] = [
                height[top_gt : top_gt + h_gt, left_gt : left_gt + w_gt]
                for height in data["gt_height"]
            ]

        if "gt_gamma" in data:
            data["gt_gamma"] = [
                gamma[top_gt : top_gt + h_gt, left_gt : left_gt + w_gt]
                for gamma in data["gt_gamma"]
            ]

        if "intrinsics" in data:
            # modify intrinsics matrix
            data["intrinsics"][0, 2] -= left[0]
            data["intrinsics"][1, 2] -= top[0]

        return data

    def __repr__(self):
        return "CropElevation"


@OBJECT_REGISTRY.register
class ToTensorElevation(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to torch.tensor.

    Args:
        to_yuv:
            wheather convert to yuv444 format durning quanti training.
        with_color_imgs: whether use color imgs.
    """

    def __init__(self, to_yuv: bool = False, with_color_imgs: bool = True):
        self.to_yuv = to_yuv
        self.with_color_imgs = with_color_imgs
        self.pil2yuv = torchvision.transforms.Compose(
            [torchvision.transforms.PILToTensor(), BgrToYuv444(rgb_input=True)]
        )

    def _to_tensor(self, data: Union[Image.Image, Sequence], as_numpy=False):
        # convert a pillow img to tensor(normlizae to [0,1])
        # F.to_tensor will normlize a pillow img to [0,1] defaultly,
        # depth and seg do not need normlization so wo convert depth and seg
        # data to numpy first.
        if isinstance(data, Sequence):
            return [self._to_tensor(_) for _ in data]
        else:
            return F.to_tensor(np.array(data) if as_numpy else data)

    def _to_yuv_tensor(self, data):
        # step1.convert a pil img to tensor(do not normlization)
        # step2.convart a rgb tensor to yuv tensor

        if isinstance(data, Sequence):
            return [self._to_yuv_tensor(_) for _ in data]
        else:
            return self.pil2yuv(data)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'
        data["color_imgs"] = self._to_tensor(data["pil_imgs"])

        if self.to_yuv:
            data["imgs"] = self._to_yuv_tensor(data["pil_imgs"])
            # generate input yuv img for quanti training
        else:
            data["imgs"] = self._to_tensor(data["pil_imgs"])
            # generate input rgb img normalized to [0,1] for float training

        if "mask" in data:
            data["mask"] = self._to_tensor(data["mask"], as_numpy=True)

        if "gt_depth" in data:
            data["gt_depth"] = self._to_tensor(data["gt_depth"], as_numpy=True)

        if "gt_height" in data:
            data["gt_height"] = self._to_tensor(
                data["gt_height"], as_numpy=True
            )

        if "gt_gamma" in data:
            data["gt_gamma"] = self._to_tensor(data["gt_gamma"], as_numpy=True)

        if "ground_norm" in data:
            data["ground_norm"] = [
                torch.from_numpy(i) for i in data["ground_norm"]
            ]

        if "camera_high" in data:
            data["camera_high"] = [
                torch.from_numpy(i) for i in data["camera_high"]
            ]

        if "ground_homo" in data:
            data["ground_homo"] = [
                torch.from_numpy(i) for i in data["ground_homo"]
            ]

        if "rotation" in data:
            data["rotation"] = [torch.from_numpy(i) for i in data["rotation"]]

        if "transition" in data:
            data["transition"] = [
                torch.from_numpy(i) for i in data["transition"]
            ]

        if "intrinsics" in data:
            data["intrinsics"] = torch.from_numpy(data["intrinsics"])

        if "timestamp" in data:
            data["timestamp"] = torch.from_numpy(data["timestamp"])

        if not self.with_color_imgs:
            data.pop("color_imgs")
        data.pop("pil_imgs")
        return data

    def __repr__(self):
        return "ToTensorElevation"


@OBJECT_REGISTRY.register
class NormalizeElevation(object):
    """Normalize a tensor image and scale gt.

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of std for each channel.
        gamma_scale: coefficient to scale gamma.
    """

    def __init__(
        self,
        mean: float = 128.0,
        std: float = 128.0,
        gamma_scale: float = 1000.0,
    ):
        self.mean = mean
        self.std = std
        self.gamma_scale = gamma_scale

    def _normalize(self, data: Union[Image.Image, Sequence], mean, std):
        if isinstance(data, Sequence):
            return [self._normalize(_, mean, std) for _ in data]
        else:
            return F.normalize(data, mean, std)

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["imgs"] = self._normalize(data["imgs"], self.mean, self.std)

        if "gt_gamma" in data:
            data["gt_gamma"] = [
                gamma * self.gamma_scale for gamma in data["gt_gamma"]
            ]
        return data

    def __repr__(self):
        return "NormalizeElevation"


@OBJECT_REGISTRY.register
class PrepareDataElevation(object):
    """Build input data for elevation task.

    Args:
        organize_data_type: specify which method to organize data.
    """

    def __init__(self, organize_data_type: str):
        assert organize_data_type in [
            "elevation_train",
            "elevation_val",
            "inference",
        ], f"do not support type of {organize_data_type}"
        self.organize_data_type = organize_data_type

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'

        cat_data = [torch.stack(_) for _ in data["imgs"]]
        if self.organize_data_type == "elevation_train":  # t,t-1,t+1
            data["img"] = [cat_data[1], cat_data[0]]  # t-1,t
            data["extra_img"] = [cat_data[0], cat_data[2]]  # t,t+1
        elif self.organize_data_type == "elevation_val":
            data["img"] = [cat_data[1], cat_data[0]]  # t-1,t
        elif self.organize_data_type == "inference":
            data["img"] = [cat_data[1], cat_data[0]]  # t-1,t

        data.pop("imgs")

        if "color_imgs" in data:
            cat_data = []
            for color_img_i in data["color_imgs"]:
                cat_data.append(torch.stack(color_img_i))
            data["color_imgs"] = cat_data
            data["color_img_cur"] = cat_data[:1]

        if "obj_mask" in data:
            data["obj_mask"] = [
                get_mask(seg[0], maks_type="obj") for seg in data["mask"]
            ]
        # class idx between 9 and 17 means dynamic class
        # (auto 33class parsing model). e.g. bus, car, person and so on.

        if "ground_mask" in data:
            data["ground_mask"] = [
                get_mask(seg[0], maks_type="ground") for seg in data["mask"]
            ]

        if "mask" in data:
            data["mask"] = [get_mask(seg[0]) for seg in data["mask"]]

        if "size" in data:
            data.pop("size")

        return data

    def __repr__(self):
        return "PrepareDataElevation"
