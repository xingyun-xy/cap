# Copyright (c) Changan Auto. All rights reserved.

import random
from numbers import Real
from typing import Dict, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.nn import functional as torchFunctional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as visionFunctional

try:
    from torchvision.transforms.functional import get_image_size
except ImportError:
    from torchvision.transforms.functional import (
        _get_image_size as get_image_size,
    )

from cap.registry import OBJECT_REGISTRY
from .functional_target import one_hot

__all__ = [
    "SegRandomCrop",
    "SegReWeightByArea",
    "LabelRemap",
    "SegOneHot",
    "SegResize",
    "SegRandomAffine",
    "Scale",
    "FlowRandomAffineScale",
]


@OBJECT_REGISTRY.register
class SegRandomCrop(object):  # noqa: D205,D400
    """Random crop on data with gt_seg label, can only be used for segmentation
     task.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout', 'gt_seg'.

    Args:
        size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float, optional): The maximum ratio that single category
            could occupy.
        ignore_index (int, optional): When considering the cat_max_ratio
            condition, the area corresponding to ignore_index will be ignored.
    """

    def __init__(self, size, cat_max_ratio=1.0, ignore_index=255):
        assert size[0] > 0 and size[1] > 0
        self.size = size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, data):
        """Randomly get a crop bounding box."""
        assert data["layout"] in ["hwc", "chw", "hw"]
        if data["layout"] == "chw":
            h, w = data["img"].shape[1:]
        else:
            h, w = data["img"].shape[:2]

        margin_h = max(h - self.size[0], 0)
        margin_w = max(w - self.size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        crop_y1, crop_y2 = offset_h, offset_h + self.size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox, layout):
        assert layout in ["hwc", "chw", "hw"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if layout in ["hwc", "hw"]:
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        else:
            img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def __call__(self, data):
        # find the right crop_bbox
        crop_bbox = self.get_crop_bbox(data)
        if self.cat_max_ratio < 1.0:
            # repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(data["gt_seg"], crop_bbox, "hw")
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (
                    len(cnt) > 1
                    and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio
                ):
                    break
                crop_bbox = self.get_crop_bbox(data)

        # crop the image
        img = self.crop(data["img"], crop_bbox, data["layout"])
        data["img"] = img
        data["img_shape"] = img.shape
        data["pad_shape"] = img.shape
        # crop semantic seg
        if "gt_seg" in data:
            data["gt_seg"] = self.crop(data["gt_seg"], crop_bbox, "hw")

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"cat_max_ratio={self.cat_max_ratio}, "
        repr_str += f"ignore_index={self.ignore_index}"
        return repr_str


@OBJECT_REGISTRY.register
class SegReWeightByArea(object):  # noqa: D205,D400
    """Calculate the weight of each category according to the area of each
    category.

    For each category, the calculation formula of weight is as follows:
    weight = max(1.0 - seg_area / total_area, lower_bound)

    .. note::
        Affected keys: 'gt_seg', 'gt_seg_weight'.

    Args:
        seg_num_classes (int): Number of segmentation categories.
        lower_bound (float): Lower bound of weight.
        ignore_index (int): Index of ignore class.
    """

    def __init__(
        self,
        seg_num_classes,
        lower_bound: int = 0.5,
        ignore_index: int = 255,
    ):
        self.seg_num_classes = seg_num_classes
        self.lower_bound = lower_bound
        self.ignore_index = ignore_index

    def _reweight_by_area(self, gt_seg):
        """Private function to generate weights based on area of semantic."""
        H, W = gt_seg.shape[0], gt_seg.shape[1]
        gt_seg_weight = np.zeros((H, W), dtype=np.float32)
        total_area = (gt_seg != self.ignore_index).sum()
        for ind in range(self.seg_num_classes):
            seg_area = (gt_seg == ind).sum()
            if seg_area > 0:
                gt_seg_weight[gt_seg == ind] = max(
                    1.0 - seg_area / total_area, self.lower_bound
                )
        return gt_seg_weight

    def __call__(self, data):
        if "gt_seg" in data:
            gt_seg_weight = self._reweight_by_area(data["gt_seg"])
            data["gt_seg_weight"] = gt_seg_weight
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"seg_num_classes={self.seg_num_classes}, "
        repr_str += f"lower_bound={self.lower_bound}"
        return repr_str


@OBJECT_REGISTRY.register
class LabelRemap(object):
    r"""
    Remap labels.

    .. note::
        Affected keys: 'gt_seg'.

    Args:
        mapping (Sequence): Mapping from input to output.
    """

    def __init__(self, mapping: Sequence):
        super(LabelRemap, self).__init__()
        if not isinstance(mapping, Sequence):
            raise TypeError(
                "mapping should be a sequence. Got {}".format(type(mapping))
            )
        self.mapping = mapping

    def __call__(self, data: Tensor):
        label = data["gt_seg"]
        mapping = torch.tensor(
            self.mapping, dtype=label.dtype, device=label.device
        )
        data["gt_seg"] = mapping[label.to(dtype=torch.long)]
        return data


@OBJECT_REGISTRY.register
class SegOneHot(object):
    r"""
    OneHot is used for convert layer to one-hot format.

    .. note::
        Affected keys: 'gt_seg'.

    Args:
        num_classes (int): Num classes.
    """

    def __init__(self, num_classes: int):
        super(SegOneHot, self).__init__()
        self.num_classes = num_classes

    def __call__(self, data):
        ndim = data["gt_seg"].ndim
        if ndim == 3 or ndim == 2:
            data["gt_seg"] = torch.unsqueeze(data["gt_seg"], 0)
        data["gt_seg"] = one_hot(data["gt_seg"], self.num_classes)
        if ndim == 3 or ndim == 2:
            data["gt_seg"] = data["gt_seg"][0]
        return data


@OBJECT_REGISTRY.register
class SegResize(torchvision.transforms.Resize):
    """
    Apply resize for both image and label.

    .. note::
        Affected keys: 'img', 'gt_seg'.

    Args:
        size: target size of resize.
        interpolation: interpolation method of resize.

    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super(SegResize, self).__init__(size, interpolation)

    def forward(self, data):
        data["img"] = super(SegResize, self).forward(data["img"])
        data["gt_seg"] = super(SegResize, self).forward(data["gt_seg"])

        return data


@OBJECT_REGISTRY.register
class SegRandomAffine(torchvision.transforms.RandomAffine):
    """
    Apply random for both image and label.

    Please refer to :class:`~torchvision.transforms.RandomAffine` for details.

    .. note::
        Affected keys: 'img', 'gt_flow', 'gt_seg'.

    Args:
        label_fill_value (tuple or int, optional): Fill value for label.
            Defaults to -1.
        translate_p: Translate flip probability, range between [0, 1].
        scale_p: Scale flip probability, range between [0, 1].
    """

    def __init__(
        self,
        degrees: Union[Sequence, float] = 0,
        translate: Tuple = None,
        scale: Tuple = None,
        shear: Union[Sequence, float] = None,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[tuple, int] = 0,
        label_fill_value: Union[tuple, int] = -1,
        translate_p: float = 1.0,
        scale_p: float = 1.0,
    ):
        super(SegRandomAffine, self).__init__(
            degrees, translate, scale, shear, interpolation, fill
        )

        self.label_fill_value = label_fill_value
        self.translate_p = translate_p
        self.scale_p = scale_p

    def __call__(self, data: Dict[str, Tensor]):
        img = data["img"]
        img_size = get_image_size(img)

        translate_flag = np.random.choice(
            [False, True], p=[1 - self.translate_p, self.translate_p]
        )
        scale_flag = np.random.choice(
            [False, True], p=[1 - self.scale_p, self.scale_p]
        )
        params = [
            self.degrees,
            self.translate,
            self.scale,
            self.shear,
            img_size,
        ]
        if not translate_flag:
            params[1] = (0.0, 0.0)
        if not scale_flag:
            params[2] = (1.0, 1.0)
        ret = self.get_params(*params)

        if "gt_flow" in data:
            if translate_flag:
                params[2] = (1.0, 1.0)
                ret = self.get_params(*params)
                data["img"][3:] = visionFunctional.affine(
                    img[3:],
                    *ret,
                    interpolation=self.interpolation,
                    fill=self.fill,
                )
                data["gt_flow"][0, ...] += ret[1][0]
                data["gt_flow"][1, ...] += ret[1][1]
            if scale_flag:
                params[1] = (0.0, 0.0)
                params[2] = self.scale
                ret = self.get_params(*params)

                data["img"] = visionFunctional.affine(
                    data["img"],
                    *ret,
                    interpolation=self.interpolation,
                    fill=self.fill,
                )
                data["gt_flow"] = visionFunctional.affine(
                    data["gt_flow"],
                    *ret,
                    interpolation=self.interpolation,
                    fill=self.label_fill_value,
                )
                data["gt_flow"] *= ret[2]
        else:
            data["img"] = visionFunctional.affine(
                img,
                *ret,
                interpolation=self.interpolation,
                fill=self.fill,
            )

            if "gt_seg" in data:
                data["gt_seg"] = visionFunctional.affine(
                    data["gt_seg"],
                    *ret,
                    interpolation=self.interpolation,
                    fill=self.label_fill_value,
                )

        return data

    def __repr__(self):
        s = super(SegRandomAffine, self).__repr__()[:-1]
        if self.label_fill_value != 0:
            s += ", label_fill_value={label_fill_value}"
        s += ")"
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


@OBJECT_REGISTRY.register
class Scale(object):
    r"""
    Scale input according to a scale list.

    .. note::
        Affected keys: 'img', 'gt_flow', 'gt_ori_flow', 'gt_seg'.

    Args:
        scales (Union[Real, Sequence]): The scales to apply on input.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'`` | ``'area'``. Default: ``'nearest'``
        mul_scale(bool): Whether to multiply the scale coefficient.
    """

    def __init__(
        self,
        scales: Union[Real, Sequence],
        mode: str = "nearest",
        mul_scale: bool = False,
    ):
        super(Scale, self).__init__()
        if isinstance(scales, Real):
            self.scales = [scales]
        elif isinstance(scales, Sequence):
            self.scales = scales
        else:
            raise TypeError(
                "scales should be number or sequence. Got {}".format(
                    type(scales)
                )
            )
        self.mode = mode
        self.mul_scale = mul_scale

    def _scale(self, data: Tensor):
        scaled_data = []
        for scale in self.scales:
            scaled_tmp_data = torchFunctional.interpolate(
                data.to(dtype=torch.float),
                scale_factor=scale,
                mode=self.mode,
                recompute_scale_factor=True,
            ).to(dtype=data.dtype)
            scaled_data.append(
                scaled_tmp_data * scale if self.mul_scale else scaled_tmp_data
            )
        return scaled_data

    def __call__(self, data: dict):
        if "gt_seg" in data:
            data["gt_seg"] = self._scale(data["gt_seg"])
        if "gt_flow" in data:
            data["gt_ori_flow"] = data["gt_flow"]
            data["gt_flow"] = self._scale(data["gt_flow"])
        return data


@OBJECT_REGISTRY.register
class FlowRandomAffineScale(object):
    def __init__(
        self,
        scale_p: float = 0.5,
        scale_r: float = 0.05,
    ):  # noqa: D205,D400,D401,D403
        """
        RandomAffineScale using Opencv, the results are slightly different from
        ~torchvision.transforms.RandomAffine with scale.

        .. note::
            Affected keys: 'img', 'gt_flow'.

        Args:
        scale_p: Scale flip probability, range between [0, 1].
        scale_r: The scale transformation range is (1-scale_r, 1 + scale_r).

        """
        self.scale_p = scale_p
        self.scale_r = scale_r

    def cvscale(self, img, zoom_factor):  # noqa: D205,D400,D401
        """
        Center zoom in/out of the given image and returning
        an enlarged/shrinked view of the image without changing dimensions

        - Scipy rotate and zoom an image without changing its dimensions
        https://stackoverflow.com/a/48097478
        Written by Mohamed Ezz
        License: MIT License

        Args:
            img : Image array
            zoom_factor : amount of zoom as a ratio (0 to Inf)
        """
        height, width = img.shape[:2]  # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(
            width * zoom_factor
        )

        # Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized \
        # (larger/smaller) image coordinates
        y1, x1 = (
            max(0, new_height - height) // 2,
            max(0, new_width - width) // 2,
        )
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1, x1, y2, x2])

        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(
            new_width, width
        )
        pad_height1, pad_width1 = (height - resize_height) // 2, (
            width - resize_width
        ) // 2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (
            width - resize_width
        ) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [
            (0, 0)
        ] * (img.ndim - 2)

        result = cv2.resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode="constant")

        assert result.shape[0] == height and result.shape[1] == width
        return result

    def __call__(self, data: Dict):
        assert data["img"].size()[0] == 6
        assert data["img"].ndim == 3
        assert "gt_flow" in data
        image1 = np.copy(data["img"].permute((1, 2, 0)).numpy()[..., :3])
        image2 = np.copy(data["img"].permute((1, 2, 0)).numpy()[..., 3:])
        flow = np.copy(data["gt_flow"].permute((1, 2, 0)).numpy())
        if self.scale_p > 0.0:
            rand = random.random()
            if rand < self.scale_p:
                ratio = random.uniform(1.0 - self.scale_r, 1.0 + self.scale_r)
                image1 = self.cvscale(image1, ratio)
                image2 = self.cvscale(image2, ratio)
                flow = self.cvscale(flow, ratio)
                flow *= ratio
        imgs = np.concatenate((image1, image2), axis=2)
        imgs_chw_np = np.ascontiguousarray(imgs.transpose((2, 0, 1)))
        flow_chw_np = np.ascontiguousarray(flow.transpose((2, 0, 1)))
        imgs_chw_tensor = torch.from_numpy(imgs_chw_np.copy())
        flow_chw_tensor = torch.from_numpy(flow_chw_np.copy())
        data["img"] = imgs_chw_tensor
        data["gt_flow"] = flow_chw_tensor
        return data
