# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection, gluoncv, gluonnas.

import copy
import logging
import random
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision

from cap.data.transforms.functional_bbox import (
    bbox_overlaps,
    is_center_of_bboxes_in_roi,
)
from cap.data.transforms.functional_img import (
    image_normalize,
    image_pad,
    imresize,
    random_flip,
)
from cap.registry import OBJECT_REGISTRY
from .affine import (
    AffineAugMat,
    AffineMat2DGenerator,
    AffineMatFromROIBoxGenerator,
    AlphaImagePyramid,
    ImageAffineTransform,
    LabelAffineTransform,
    _pad_array,
    resize_affine_mat,
)
from .bbox import (
    clip_bbox,
    filter_bbox,
    remap_bbox_label_by_area,
    remap_bbox_label_by_clip_area_ratio,
)
from .classification import BgrToYuv444
from .common import Cast

logger = logging.getLogger(__name__)

__all__ = [
    "Resize",
    "RandomFlip",
    "Pad",
    "Normalize",
    "RandomCrop",
    "ToTensor",
    "Batchify",
    "FixedCrop",
    "ColorJitter",
    "RandomExpand",
    "MinIoURandomCrop",
    "AugmentHSV",
    "get_dynamic_roi_from_camera",
    "IterableDetRoITransform",
]


def _transform_bboxes(
    gt_boxes,
    ig_regions,
    img_roi,
    affine_aug_param,
    clip=True,
    min_valid_area=8,
    min_valid_clip_area_ratio=0.5,
    min_edge_size=2,
):
    bbox_ts = LabelAffineTransform(label_type="box")

    ts_gt_boxes = bbox_ts(
        gt_boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
    )
    if clip:
        clip_gt_boxes = clip_bbox(ts_gt_boxes, img_roi)
    else:
        clip_gt_boxes = ts_gt_boxes
    clip_gt_boxes = remap_bbox_label_by_area(clip_gt_boxes, min_valid_area)
    clip_gt_boxes = remap_bbox_label_by_clip_area_ratio(
        ts_gt_boxes, clip_gt_boxes, min_valid_clip_area_ratio
    )
    clip_gt_boxes = filter_bbox(
        clip_gt_boxes,
        img_roi,
        allow_outside_center=True,
        min_edge_size=min_edge_size,
    )

    if ig_regions is not None:
        ts_ig_regions = bbox_ts(
            ig_regions, affine_aug_param.mat, flip=affine_aug_param.flipped
        )
        if clip:
            clip_ig_regions = clip_bbox(ts_ig_regions, img_roi)
        else:
            clip_ig_regions = ts_ig_regions
    else:
        clip_ig_regions = None

    return clip_gt_boxes, clip_ig_regions


@OBJECT_REGISTRY.register
class Resize(object):
    """Resize image & bbox & mask & seg.

    .. note::
        Affected keys: 'img', 'ori_img', 'img_shape', 'pad_shape',
        'resized_shape', 'pad_shape', 'scale_factor', 'gt_bboxes', 'gt_seg'.

    Args:
        img_scale: See above.
        max_scale: The max size of image. If the image's shape > max_scale,
            The image is resized to max_scale
        multiscale_mode (str): Value must be one of "range" or "value".
            This transform resizes the input image and bbox to same scale
            factor.
            There are 3 multiscale modes:
            'ratio_range' is not None: randomly sample a ratio from the ratio
            range and multiply with the image scale.
            e.g. Resize(img_scale=(400, 500)), multiscale_mode='range',
            ratio_range=(0.5, 2.0)
            'ratio_range' is None and 'multiscale_mode' == "range": randomly
            sample a scale from a range, the length of img_scale[tuple] must be
            2, which represent small img_scale and large img_scale.
            e.g. Resize(img_scale=((100, 200), (400,500)),
            multiscale_mode='range')
            'ratio_range' is None and 'multiscale_mode' == "value": randomly
            sample a scale from multiple scales.
            e.g. Resize(img_scale=((100, 200), (300, 400), (400, 500)),
            multiscale_mode='value')))
        ratio_range (tuple[float]): Value represent (min_ratio, max_ratio), \
            scale factor range.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the \
            image.
    """

    def __init__(
        self,
        img_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        max_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
    ):
        if img_scale is None:
            self.img_scale = img_scale
        else:
            if isinstance(img_scale, (list, tuple)):
                if isinstance(img_scale[0], (tuple, list)):
                    self.img_scale = img_scale
                else:
                    self.img_scale = [img_scale]
            else:
                self.img_scale = [img_scale]
            for value in self.img_scale:
                assert isinstance(value, (tuple, list)), (
                    "you should set img_scale like a tupe/list or a list of "
                    "tuple/list"
                )
        self.max_scale = max_scale
        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range", "max_size"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long), max(img_scale_long) + 1
        )
        short_edge = np.random.randint(
            min(img_scale_short), max(img_scale_short) + 1
        )
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, (tuple, list)) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, 0

    @staticmethod
    def max_size(max_scale, origin_shape):
        if max(origin_shape) > max(max_scale):
            resize_scale = max_scale
        else:
            resize_scale = origin_shape

        return resize_scale, 0

    def _random_scale(self, data):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif self.multiscale_mode == "max_size":
            scale, scale_idx = self.max_size(
                self.max_scale, data["img_shape"][:2]
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        data["scale"] = scale
        data["scale_idx"] = scale_idx

    def _resize_img(self, data):
        h = data["scale"][0]
        w = data["scale"][1]
        img = data["img"]
        resized_img, w_scale, h_scale = imresize(
            img,
            w,
            h,
            data["layout"],
            keep_ratio=self.keep_ratio,
            return_scale=True,
        )
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
        )

        if "ori_img" in data:
            ori_img = data["ori_img"]
            resized_ori_img, _, _ = imresize(
                ori_img,
                w,
                h,
                data["layout"],
                keep_ratio=self.keep_ratio,
                return_scale=True,
            )
            # No direct replacement to prevent ori_img used.
            data["resized_ori_img"] = resized_ori_img

        data["img"] = resized_img
        data["img_shape"] = resized_img.shape
        data["resized_shape"] = resized_img.shape
        data[
            "pad_shape"
        ] = resized_img.shape  # in case that there is no padding  # noqa
        data["scale_factor"] = scale_factor
        data["keep_ratio"] = self.keep_ratio

    def _resize_bbox(self, data):
        if data["gt_bboxes"].any():
            if data["layout"] == "hwc":
                h, w = data["img_shape"][:2]
            else:
                h, w = data["img_shape"][1:]
            bboxes = data["gt_bboxes"] * data["scale_factor"]
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h - 1)
            data["gt_bboxes"] = bboxes

    def _resize_seg(self, data):
        """Resize semantic segmentation map with ``data['scale']``."""
        h = data["scale"][0]
        w = data["scale"][1]
        resized_seg = imresize(
            data["gt_seg"],
            w,
            h,
            "hw",
            keep_ratio=self.keep_ratio,
            return_scale=False,
            interpolation="nearest",
        )
        data["gt_seg"] = resized_seg

    def _resize_mask(self, data):
        raise NotImplementedError

    def inverse_transform(self, inputs, task_type, inverse_info):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array|Tensor): Prediction.
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            scale_factor = inverse_info["scale_factor"]
            if not isinstance(scale_factor, torch.Tensor):
                scale_factor = inputs.new_tensor(scale_factor)
            inputs[:, :4] = inputs[:, :4] / scale_factor
            return inputs
        elif task_type == "segmentation":
            scale_factor = inverse_info["scale_factor"][:2]
            if isinstance(scale_factor, torch.Tensor):
                scale_factor = scale_factor.detach().cpu().numpy()
            elif isinstance(scale_factor, (tuple, list)):
                scale_factor = np.array(scale_factor)
            else:
                assert isinstance(scale_factor, np.ndarray)
            before_resize_shape = inputs.shape / scale_factor
            out_height, out_width = before_resize_shape
            out_img = cv2.resize(
                inputs,
                (int(out_width), int(out_height)),
                interpolation=cv2.INTER_NEAREST,
            )
            return out_img
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        self._random_scale(data)
        self._resize_img(data)
        if "gt_bboxes" in data:
            self._resize_bbox(data)
        if "gt_seg" in data:
            self._resize_seg(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio}"
        return repr_str


@OBJECT_REGISTRY.register
class RandomFlip(object):
    """Flip image & bbox & mask & seg & flow.

    .. note::
        Affected keys: 'img', 'ori_img', 'img_shape', 'pad_shape', 'gt_bboxes',
        'gt_seg', 'gt_flow', 'gt_mask', 'gt_ldmk', 'ldmk_pairs'.

    Args:
        px: Horizontal flip probability, range between [0, 1].
        py: Vertical flip probability, range between [0, 1].
    """

    def __init__(self, px: Optional[float] = 0.5, py: Optional[float] = 0):
        assert px >= 0 and px <= 1, "px must range between [0, 1]"
        assert py >= 0 and py <= 1, "py must range between [0, 1]"
        self.px = px
        self.py = py

    def _flip_img(self, data):
        data["img"], (flip_x, flip_y) = random_flip(
            data["img"], data["layout"], self.px, self.py
        )
        return flip_x, flip_y

    def _flip_bbox(self, data):
        if "img_shape" not in data:
            img_shape = (data["img_height"], data["img_width"])
        else:
            img_shape = data["img_shape"]
            if "pad_shape" in data:
                img_shape = data["pad_shape"]
        bboxes = data["gt_bboxes"]  # shape is Nx4, format is (x1, y1, x2, y2)
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape[:2]
        if self.flip_x:
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        if self.flip_y:
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        data["gt_bboxes"] = flipped

    def _flip_seg(self, data):
        flipped_seg, _ = random_flip(
            data["gt_seg"], "hw", self.flip_x, self.flip_y
        )
        data["gt_seg"] = flipped_seg

    def _flip_flow(self, data):
        flipped_flow, _ = random_flip(
            data["gt_flow"], "hwc", self.flip_x, self.flip_y
        )
        if isinstance(flipped_flow, np.ndarray):
            flipped_flow_clone = np.copy(flipped_flow)
        else:
            flipped_flow_clone = flipped_flow.clone()
        if self.flip_x:
            flipped_flow_clone[:, :, 0] = flipped_flow_clone[:, :, 0] * -1
        if self.flip_y:
            flipped_flow_clone[:, :, 1] = flipped_flow_clone[:, :, 1] * -1
        data["gt_flow"] = flipped_flow_clone

    def _flip_mask(self, data):
        flipped_mask, _ = random_flip(
            data["gt_mask"], data["layout"], self.flip_x, self.flip_y
        )
        data["gt_mask"] = flipped_mask

    def _flip_target_img(self, data):
        """Gt image for reconstruction loss.

        data["gt_img"] is used for superviseing reconstructed image.
        It may be different from data["img] in image size or color space.
        """
        flipped_gt_img, _ = random_flip(
            data["gt_img"], data["layout"], self.flip_x, self.flip_y
        )
        data["gt_img"] = flipped_gt_img

    def _flip_ldmk(self, data):
        if "img_shape" not in data:
            height, width = (data["img_height"], data["img_width"])
        else:
            height, width, _ = data["img_shape"]
        ldmk = data["gt_ldmk"].copy()
        pairs = data["ldmk_pairs"]

        if self.flip_x:
            if pairs is not None:
                for pair in pairs:
                    temp = ldmk[pair[0]].copy()
                    ldmk[pair[0]] = ldmk[pair[1]]
                    ldmk[pair[1]] = temp
            ldmk[:, 0] = width - ldmk[:, 0] - 1
        if self.flip_y:
            ldmk[:, 1] = height - ldmk[:, 1] - 1
        data["gt_ldmk"] = ldmk

    def __call__(self, data):
        flip_x, flip_y = self._flip_img(data)
        # bbox & mask & seg do the same flip operation as img
        self.flip_x = flip_x
        self.flip_y = flip_y
        if "gt_bboxes" in data:
            self._flip_bbox(data)
        if "gt_seg" in data:
            self._flip_seg(data)
        if "gt_flow" in data:
            self._flip_flow(data)
        if "gt_ldmk" in data:
            self._flip_ldmk(data)
        if "gt_mask" in data:
            self._flip_mask(data)
        if "gt_img" in data:
            self._flip_target_img(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"px={self.px}, py={self.py}"
        return repr_str


@OBJECT_REGISTRY.register
class Pad(object):
    def __init__(self, size=None, divisor=1, pad_val=0, seg_pad_val=255):
        """Pad image & mask & seg.

        .. note::
            Affected keys: 'img', 'layout', 'pad_shape', 'gt_seg'.

        Args:
            size (Optional[tuple]): Expected padding size, meaning of dimension
                is the same as img, if layout of img is `hwc`, shape must be
                (pad_h, pad_w) or (pad_h, pad_w, c).
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (Union[float, Sequence[float]]): Values to be filled in
                padding areas for img, single value or a list of values with
                len c. E.g. : pad_val = 10, or pad_val = [10, 20, 30].
            seg_pad_val (Optional[float]): Value to be filled in padding areas
                for gt_seg.
        """
        self.size = size
        self.divisor = divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, data):
        padded_img = image_pad(
            data["img"], data["layout"], self.size, self.divisor, self.pad_val
        )
        data["img"] = padded_img
        data["pad_shape"] = padded_img.shape

    def _pad_seg(self, data):
        if data["layout"] == "chw":
            size = data["pad_shape"][1:]
        else:
            size = data["pad_shape"][:2]
        padded_seg = image_pad(data["gt_seg"], "hw", size, 1, self.seg_pad_val)
        data["gt_seg"] = padded_seg

    def __call__(self, data):
        self._pad_img(data)
        if "gt_seg" in data:
            self._pad_seg(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"divisor={self.divisor}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"seg_pad_val={self.seg_pad_val}"
        return repr_str


@OBJECT_REGISTRY.register
class Normalize(object):
    """
    Normalize image.

    .. note::
        Affected keys: 'img', 'layout'.

    Args:
        mean: mean of normalize.
        std: std of normalize.
    """

    def __init__(
        self,
        mean: Union[float, Sequence[float]],
        std: Union[float, Sequence[float]],
    ):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        layout = data.get("layout", "chw")
        data["img"] = image_normalize(data["img"], self.mean, self.std, layout)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"mean={self.mean}, std={self.std}"
        return repr_str


@OBJECT_REGISTRY.register
class RandomCrop(object):
    def __init__(
        self,
        size,
        min_area=-1,
        min_iou=-1,
        center_crop_prob=0.0,
        center_shake=None,
        truncate_gt=True,
    ):  # noqa: D205,D400
        """

        .. note::
            Affected keys: 'img', 'img_shape', 'pad_shape', 'layout',
            'gt_bboxes', 'gt_classes', 'gt_seg', 'gt_flow'.

        Args:
            size (Tuple): Expected size after cropping, (h, w).
            min_area (Optional[int]): If min_area > 0, boxes whose areas are
                less than min_area will be ignored.
            min_iou (Optional[float]): If min_iou > 0, boxes whose iou between
                before and after truncation < min_iou will be ignored.
            center_crop_prob (Optional[float]): The center_crop_prob is the
                center crop probability
            center_shake (Optional[tuple]): The list is the center shake's top,
                bottom, left, right range
            truncate_gt (Optional[bool]): if True, truncate the gt_boxes when
                the gt_boxes exceed the crop area. Default True.
        """
        self.size = size
        self.min_area = min_area
        self.min_iou = min_iou
        assert 0 <= center_crop_prob <= 1
        self.center_crop_prob = center_crop_prob
        self.center_shake = center_shake
        self.truncate_gt = truncate_gt

    def _crop_img(self, data):
        img = data["img"]
        layout = data["layout"]
        assert layout in ["hwc", "chw"], (
            "layout of img must be `chw` or " "`hwc`"
        )
        do_center_crop = np.random.rand() < self.center_crop_prob

        if layout == "hwc":
            margin_h = max(img.shape[0] - self.size[0], 0)
            margin_w = max(img.shape[1] - self.size[1], 0)
        else:
            margin_h = max(img.shape[1] - self.size[0], 0)
            margin_w = max(img.shape[2] - self.size[1], 0)
        # randomly generate upper left corner coordinates
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        if do_center_crop:
            if self.center_shake is not None:
                center_shake = copy.deepcopy(list(self.center_shake))
                if "scale_factor" in data.keys():
                    scale_factor_h = data["scale_factor"][1]
                    scale_factor_w = data["scale_factor"][0]
                    center_shake[0] = int(center_shake[0] * scale_factor_h)
                    center_shake[1] = int(center_shake[1] * scale_factor_h)
                    center_shake[2] = int(center_shake[2] * scale_factor_w)
                    center_shake[3] = int(center_shake[3] * scale_factor_w)
                center_h = np.floor(margin_h / 2)
                center_w = np.floor(margin_w / 2)
                offset_h = np.random.randint(
                    max(center_h - center_shake[0], 0),
                    min(center_h + center_shake[1] + 1, margin_h + 1),
                )
                offset_w = np.random.randint(
                    max(center_w - center_shake[2], 0),
                    min(center_w + center_shake[3] + 1, margin_w + 1),
                )
            else:
                offset_h = np.floor(margin_h / 2)
                offset_w = np.floor(margin_w / 2)
        crop_y1, crop_y2 = offset_h, min(offset_h + self.size[0], img.shape[0])
        crop_x1, crop_x2 = offset_w, min(offset_w + self.size[1], img.shape[1])
        # crop the image
        crop_img = img[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2), :
        ]
        img_shape = crop_img.shape
        data["img"] = crop_img
        data["img_shape"] = img_shape
        data["pad_shape"] = img_shape
        data["crop_offset"] = [offset_w, offset_h, offset_w, offset_h]
        return offset_h, offset_w, (crop_y1, crop_y2, crop_x1, crop_x2)

    def _crop_bbox(self, data, offset_h, offset_w):
        # crop bboxes and clip to the image boundary
        img_shape = data["img_shape"]
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] - bbox_offset
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            area_ignore_index = (area < 1e-3) & (classes >= 0)
            classes[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                classes[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        classes[i] = -1

            # filter out the gt bboxes that are completely cropped
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1]
            )
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            # keep dtype consistency with origin bboxes and classes
            if not np.any(valid_inds):
                boxes_real = bboxes = np.zeros(
                    (0, 4), dtype=data["gt_bboxes"].dtype
                )
                classes = np.zeros((0,), dtype=data["gt_classes"].dtype)
            data["gt_bboxes"] = bboxes if self.truncate_gt else boxes_real
            data["gt_classes"] = classes

    def _crop_seg(self, data, crop_bbox):
        gt_seg = data["gt_seg"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_seg = gt_seg[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_seg"] = crop_gt_seg

    def _crop_flow(self, data, crop_bbox):
        gt_flow = data["gt_flow"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_flow = gt_flow[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_flow"] = crop_gt_flow

    def inverse_transform(self, inputs, task_type, inverse_info):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array): Prediction
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            crop_offset = inverse_info["crop_offset"]
            if not isinstance(crop_offset, torch.Tensor):
                crop_offset = inputs.new_tensor(crop_offset)
            crop_offset = crop_offset.reshape((1, 4))
            inputs[:, :4] = inputs[:, :4] + crop_offset
            return inputs
        elif task_type == "segmentation":
            before_crop_shape = inverse_info["before_crop_shape"][1:]
            crop_offset_x, crop_offset_y = inverse_info["crop_offset"][:2]
            crop_h, crop_w = inputs.shape
            out_img = np.full(before_crop_shape, 255)
            out_img[
                crop_offset_y : crop_offset_y + crop_h,
                crop_offset_x : crop_offset_x + crop_w,
            ] = inputs.cpu().numpy()
            return out_img
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        offset_h, offset_w, crop_bbox = self._crop_img(data)
        if "gt_bboxes" in data:
            self._crop_bbox(data, offset_h, offset_w)
        elif "gt_seg" in data:
            self._crop_seg(data, crop_bbox)
        elif "gt_flow" in data:
            self._crop_flow(data, crop_bbox)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"min_area={self.min_area}, "
        repr_str += f"min_iou={self.min_iou}"
        return repr_str


@OBJECT_REGISTRY.register
class FixedCrop(RandomCrop):
    """Crop image with fixed position and size.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout',
        'before_crop_shape', 'crop_offset', 'gt_bboxes', 'gt_classes'.

    """

    def __init__(
        self,
        size=None,
        min_area=-1,
        min_iou=-1,
        dynamic_roi_params=None,
    ):  # noqa: D205,D400
        """

        Args:
            size (Tuple): Expected size after cropping, (w, h) or
                (x1, y1, w, h).
            min_area (Optional[int]): If min_area > 0, boxes whose areas are
                less than min_area will be ignored.
            min_iou (Optional[float]): If min_iou > 0, boxes whose iou between
                before and after truncation < min_iou will be ignored.
            dynamic_roi_param (Dict): Dynamic ROI parameters contains keys
                {'w', 'h', 'fp_x', 'fp_y'}

        """
        if not dynamic_roi_params:
            assert size is not None
            if len(size) == 2:
                size = (0, 0, size[0], size[1])
            else:
                assert len(size) == 4
        self.size = size
        self.min_area = min_area
        self.min_iou = min_iou
        self.dynamic_roi_params = dynamic_roi_params

    def _crop_img(self, data):
        img = data["img"]
        if self.dynamic_roi_params:
            crop_roi, crop_roi_on_orig = get_dynamic_roi_from_camera(
                camera_info=data["camera_info"],
                dynamic_roi_params=self.dynamic_roi_params,
                img_hw=data["img_shape"][:2],
                infer_model_type=data["infer_model_type"],
            )
            data["crop_roi"] = crop_roi_on_orig
            x1, y1, x2, y2 = crop_roi
        else:
            x1, y1, w, h = self.size
            x2, y2 = x1 + w, y1 + h
            assert w > 0 and h > 0

        assert x2 <= img.shape[1] and y2 <= img.shape[0]
        offset_w, offset_h = x1, y1

        # crop the image
        crop_img = img[y1:y2, x1:x2, :]
        data["img"] = crop_img
        data["img_shape"] = crop_img.shape
        data["before_crop_shape"] = img.shape
        data["crop_offset"] = [x1, y1, x1, y1]
        return offset_h, offset_w

    def _crop_bbox(self, data, offset_h, offset_w):
        # crop bboxes and clip to the image boundary
        img_shape = data["img_shape"]
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] - bbox_offset
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            area_ignore_index = (area < 1e-3) & (classes >= 0)
            classes[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                classes[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        classes[i] = -1

            # filter out the gt bboxes that are completely cropped
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1]
            )
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            if not np.any(valid_inds):
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int64)
            data["gt_bboxes"] = bboxes
            data["gt_classes"] = classes

    def inverse_transform(self, inputs, task_type, inverse_info):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array): Prediction
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            crop_offset = inverse_info["crop_offset"]
            if not isinstance(crop_offset, torch.Tensor):
                crop_offset = inputs.new_tensor(crop_offset)
            crop_offset = crop_offset.reshape((1, 4))
            inputs[:, :4] = inputs[:, :4] + crop_offset
            return inputs
        elif task_type == "segmentation":
            crop_offset_x, crop_offset_y = inverse_info["crop_offset"][:2]
            crop_h, crop_w = inputs.shape
            before_crop_shape = inverse_info["before_crop_shape"][1:]
            if before_crop_shape:
                out_img = np.full(before_crop_shape, 255)
                out_img[
                    crop_offset_y : crop_offset_y + crop_h,
                    crop_offset_x : crop_offset_x + crop_w,
                ] = inputs.cpu().numpy()
            else:
                out_img = inputs.cpu().numpy()
            return out_img
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        offset_h, offset_w = self._crop_img(data)
        if "gt_bboxes" in data:
            self._crop_bbox(data, offset_h, offset_w)
        return data


@OBJECT_REGISTRY.register
class ToTensor(object):  # noqa: D205,D400
    """Convert objects of various python types to torch.Tensor and convert the
    img to yuv444 format if to_yuv is True.

    Supported types are: numpy.ndarray, torch.Tensor, Sequence, int, float.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout', 'gt_bboxes',
        'gt_seg', 'gt_seg_weights', 'gt_flow', 'color_space'.

    Args:
        to_yuv (bool): If true, convert the img to yuv444 format.

    """

    def __init__(self, to_yuv=False):
        self.to_yuv = to_yuv

    @staticmethod
    def _convert_layout(img: Union[torch.Tensor, np.ndarray], layout: str):
        # convert the layout from hwc to chw
        assert layout in ["hwc", "chw"]
        if layout == "chw":
            return img, layout

        if isinstance(img, torch.Tensor):
            img = img.permute((2, 0, 1))  # HWC > CHW
        elif isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC > CHW
        else:
            raise TypeError
        return img, "chw"

    @staticmethod
    def _to_tensor(
        data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
    ):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy())
        elif isinstance(data, Sequence) and not isinstance(data, str):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(
                f"type {type(data)} cannot be converted to tensor."
            )

    @staticmethod
    def _convert_hw_to_even(data):  # noqa: D205,D400
        """Convert hw of img and img-like labels to even because BgrToYuv444
        requires that the h and w of img are even numbers.
        """
        if data["layout"] == "hwc":
            h, w, c = data["img"].shape
        else:
            c, h, w = data["img"].shape
            if "before_crop_shape" in data:
                data["before_crop_shape"] = (
                    data["before_crop_shape"][2],
                    data["before_crop_shape"][0],
                    data["before_crop_shape"][1],
                )

        crop_h = h if (h % 2) == 0 else h - 1
        crop_w = w if (w % 2) == 0 else w - 1

        if data["layout"] == "hwc":
            data["img"] = data["img"][:crop_h, :crop_w, :]
        else:
            data["img"] = data["img"][:, :crop_h, :crop_w]

        if "gt_seg" in data:
            data["gt_seg"] = data["gt_seg"][:crop_h, :crop_w]
        if "gt_seg_weights" in data:
            data["gt_seg_weights"] = data["gt_seg_weights"][:crop_h, :crop_w]

        if "gt_flow" in data:
            data["gt_flow"] = data["gt_flow"][:, :crop_h, :crop_w]
        # update img_shape
        data["img_shape"] = np.array(data["img"].shape)
        # update pad_shape
        data["pad_shape"] = np.array(data["img"].shape)
        return data

    def __call__(self, data):
        # step1: convert the layout from hwc to chw
        data_layout = data["layout"]
        data["img"], data["layout"] = self._convert_layout(
            data["img"], data["layout"]
        )
        # update img_shape
        data["img_shape"] = np.array(data["img"].shape)
        # update pad_shape
        data["pad_shape"] = np.array(data["img"].shape)
        # step2: convert to tensor
        data["img"] = self._to_tensor(data["img"])
        if "gt_bboxes" in data:
            data["gt_bboxes"] = self._to_tensor(data["gt_bboxes"])
        if "gt_classes" in data:
            data["gt_classes"] = self._to_tensor(data["gt_classes"])
        if "gt_seg" in data:
            data["gt_seg"] = self._to_tensor(data["gt_seg"])
        if "gt_seg_weights" in data:
            data["gt_seg_weights"] = self._to_tensor(data["gt_seg_weights"])
        if "gt_flow" in data:
            data["gt_flow"], _ = self._convert_layout(
                data["gt_flow"], data_layout
            )
            data["gt_flow"] = self._to_tensor(data["gt_flow"])
        # step3: convert to yuv color_space, if necessary
        if self.to_yuv:
            data = self._convert_hw_to_even(data)
            color_space = data.get("color_space", None)
            if color_space is None:
                color_space = "bgr"
                logger.warning(
                    "current color_space is unknown, treat as bgr "
                    "by default"
                )
            rgb_input = True if color_space.lower() == "rgb" else False
            data["img"] = BgrToYuv444(rgb_input=rgb_input)(data["img"])
            data["color_space"] = "yuv"

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"to_yuv={self.to_yuv}"
        return repr_str


@OBJECT_REGISTRY.register
class Batchify(object):
    def __init__(
        self, size: Sequence, divisor=1, pad_val=0, seg_pad_val=255, repeat=1
    ):
        """Collate the image-like data to the expected size.

        .. note::
            Affected keys: 'img', 'img_shape', 'layout', 'gt_seg'.

        Args:
            size (Tuple): The expected size of collated images, (h, w).
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (Union[float, Sequence[float]]): Values to be filled in
                padding areas for img, single value or a list of values with
                len c. E.g. : pad_val = 10, or pad_val = [10, 20, 30].
            seg_pad_val (Optional[float]): Value to be filled in padding areas
                for gt_seg.
            repeat (int): The returned imgs will consist of repeat img.

        """
        size = list(size)
        size[0] = int(np.ceil(size[0] / divisor)) * divisor
        size[1] = int(np.ceil(size[1] / divisor)) * divisor
        self.size = size
        self.divisor = divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.repeat = repeat

    def __call__(self, data):
        # get max-shape
        short, long = min(self.size), max(self.size)
        if data["layout"] == "hwc":
            h, w = data["img"].shape[:2]
        else:
            h, w = data["img"].shape[1:]

        if w > h:
            max_shape = (short, long)
        else:
            max_shape = (long, short)
        # pad img
        padded_img = image_pad(
            data["img"], data["layout"], max_shape, pad_val=self.pad_val
        )
        data["imgs"] = [padded_img for _ in range(self.repeat)]
        data["pad_shape"] = np.array(padded_img.shape)
        # pad seg
        if "gt_seg" in data:
            padded_seg = image_pad(
                data["gt_seg"], "hw", max_shape, pad_val=self.seg_pad_val
            )
            data["gt_seg"] = padded_seg
        # delete useless key-value
        del data["img"]

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"divisor={self.divisor}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"seg_pad_val={self.seg_pad_val}, "
        repr_str += f"repeat={self.repeat}"
        return repr_str


@OBJECT_REGISTRY.register
class ColorJitter(torchvision.transforms.ColorJitter):  # noqa: D205,D400
    """Randomly change the brightness, contrast, saturation and
    hue of an image.

    For det and dict input are the main differences
    with ColorJitter in torchvision and the default settings have been
    changed to the most common settings.

    .. note::
        Affected keys: 'img'.

    Args:
        brightness (float or tuple of float (min, max)):
            How much to jitter brightness.
        contrast (float or tuple of float (min, max)):
            How much to jitter contrast.
        saturation (float or tuple of float (min, max)):
            How much to jitter saturation.
        hue (float or tuple of float (min, max)):
            How much to jitter hue.
    """

    def __init__(
        self,
        brightness=0.5,
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5),
        hue=0.1,
    ):
        super(ColorJitter, self).__init__(
            brightness, contrast, saturation, hue
        )

    def __call__(self, data):
        assert "img" in data.keys()
        img = data["img"]
        img = super().__call__(img)
        data["img"] = img
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"brightness={self.brightness}, "
        repr_str += f"contrast={self.contrast}, "
        repr_str += f"saturation={self.saturation}, "
        repr_str += f"hue={self.hue}. "
        return repr_str


@OBJECT_REGISTRY.register
class RandomExpand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    .. note::
        Affected keys: 'img', 'gt_bboxes'.

    Args:
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self, mean=(0, 0, 0), ratio_range=(1, 4), prob=0.5):
        self.mean = mean
        self.ratio_range = ratio_range
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob

    def __call__(self, data):
        """Call function to expand images, bounding boxes.

        Args:
            data (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return data

        img = data["img"]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_img = np.empty(
                (int(h * ratio), int(w * ratio), c), img.dtype
            )
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full(
                (int(h * ratio), int(w * ratio), c), self.mean, dtype=img.dtype
            )
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top : top + h, left : left + w] = img

        data["img"] = expand_img

        # expand bboxes
        data["gt_bboxes"] = data["gt_bboxes"] + np.tile((left, top), 2).astype(
            data["gt_bboxes"].dtype
        )

        # TODO(zhigang.yang, 0.5): expand segs
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        return repr_str


@OBJECT_REGISTRY.register
class MinIoURandomCrop(object):  # noqa: D205,D400
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    .. note::
        Affected keys: 'img', 'gt_bboxes', 'gt_classes', 'gt_difficult'.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool): Whether clip the objects outside
            the border of the image. Defaults to True.
        repeat_num (float): Max repeat num for finding avaiable bbox.

    """

    def __init__(
        self,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3,
        bbox_clip_border=True,
        repeat_num=50,
    ):
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.repeat_num = repeat_num

    def __call__(self, data):
        img = data["img"]
        assert "gt_bboxes" in data
        boxes = data["gt_bboxes"]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return data

            min_iou = mode
            for _i in range(self.repeat_num):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h))
                )
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)
                ).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    mask = is_center_of_bboxes_in_roi(boxes, patch)
                    if not mask.any():
                        continue

                    boxes = data["gt_bboxes"].copy()
                    mask = is_center_of_bboxes_in_roi(boxes, patch)
                    boxes = boxes[mask]
                    if self.bbox_clip_border:
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)

                    data["gt_bboxes"] = boxes
                    # labels
                    if "gt_classes" in data:
                        data["gt_classes"] = data["gt_classes"][mask]
                    if "gt_difficult" in data:
                        data["gt_difficult"] = data["gt_difficult"][mask]

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1] : patch[3], patch[0] : patch[2]]
                data["img"] = img
                data["img_shape"] = img.shape
                data["img_height"] = img.shape[0]
                data["img_width"] = img.shape[1]

                # TODO(zhigang.yang, 0.5): add seg mask
                return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(min_ious={self.min_ious}, "
        repr_str += f"min_crop_size={self.min_crop_size}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


@OBJECT_REGISTRY.register
class AugmentHSV(object):
    """Random add color disturbance.

    Convert RGB img to HSV, and then randomly change the hue,
    saturation and value.

    .. note::
        Affected keys: 'img'.

    Args:
        hgain (float): Gain of hue.
        sgain (float): Gain of saturation.
        vgain (float): Gain of value.
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, data):
        img = data["img"]
        r = (
            np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
            + 1
        )
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge(
            (
                cv2.LUT(hue, lut_hue),
                cv2.LUT(sat, lut_sat),
                cv2.LUT(val, lut_val),
            )
        ).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        data["img"] = img
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hgain={self.hgain}, "
        repr_str += f"sgain={self.sgain}, "
        repr_str += f"vgain={self.vgain})"
        return repr_str


def get_dynamic_roi_from_camera(
    camera_info,
    dynamic_roi_params,
    img_hw,
    infer_model_type,
):
    """
    Get dynamic roi from camera info.

    Args:
        camera_info (Dict): Camera info.
        dynamic_roi_params (Dict): Must contains keys
            {'w', 'h', 'fp_x', 'fp_y'}
        img_hw (List|Tuple):  of 2 int height and width of the image.

    Returns:
        dynamic_roi (List|Tuple):  dynamic ROI coordinate [x1, y1, x2, y2]
            of the image.
    """
    from crop_roi.merge.utils import get_camera, get_fanishing_point

    assert isinstance(dynamic_roi_params, dict)
    assert (
        "w" in dynamic_roi_params
        and "h" in dynamic_roi_params
        and "fp_x" in dynamic_roi_params
        and "fp_y" in dynamic_roi_params
    ), f"{dynamic_roi_params}"
    cam_info = None if camera_info == 0 else camera_info
    fp_x = dynamic_roi_params["fp_x"]
    fp_y = dynamic_roi_params["fp_y"]
    w = dynamic_roi_params["w"]
    h = dynamic_roi_params["h"]
    if cam_info:
        cam = get_camera(cam_info)
        fanishing_point = get_fanishing_point(cam.gnd2img)
        if infer_model_type == "crop_with_resize_quarter":
            fanishing_point = [loc / 2 for loc in fanishing_point]
        x1, y1 = (
            fanishing_point[0] - fp_x,
            fanishing_point[1] - fp_y,
        )
        x2, y2 = x1 + w, y1 + h
        dynamic_roi = [x1, y1, x2, y2]
    else:
        fanishing_point = (img_hw[1] / 2, img_hw[0] / 2)
        x1, y1 = (
            fanishing_point[0] - fp_x,
            fanishing_point[1] - fp_y,
        )
        x2, y2 = (
            x1 + w,
            y1 + h,
        )
        dynamic_roi = [x1, y1, x2, y2]
    if infer_model_type == "crop_with_resize_quarter":
        dynamic_roi_on_orig = [coord * 2 for coord in dynamic_roi]
    else:
        dynamic_roi_on_orig = dynamic_roi
    dynamic_roi = [int(i) for i in dynamic_roi]
    dynamic_roi_on_orig = [int(i) for i in dynamic_roi_on_orig]
    return dynamic_roi, dynamic_roi_on_orig


def pad_detection_data(
    img,
    gt_boxes,
    ig_regions=None,
):
    """
    Pad detection data.

    Parameters
    ----------
    img : array
        With shape (H, W, 1) or (H, W, 3), required that
        H <= target_wh[1] and W <= target_wh[0]
    gt_boxes : array
        With shape (B, K), required that B <= max_gt_boxes_num
    ig_regions : array, optional
        With shape (IB, IK), required that IB <= max_ig_regions_num,
        by default None
    """
    im_hw = np.array(img.shape[:2]).reshape((2,))

    cast = Cast(np.float32)

    if ig_regions is None:
        return {
            "img": img,
            "im_hw": cast(im_hw),
            "gt_boxes": cast(gt_boxes),
        }

    return {
        "img": img,
        "im_hw": cast(im_hw),
        "gt_boxes": cast(gt_boxes),
        "ig_regions": cast(ig_regions),
    }


@OBJECT_REGISTRY.register
class IterableDetRoITransform:
    """
    Iterable transformer base on rois for object detection.

    Parameters
    ----------
    resize_wh : list/tuple of 2 int, optional
        Resize input image to target size, by default None
    **kwargs :
        Please see :py:class:`AffineMatFromROIBoxGenerator` and
        :py:class:`ImageAffineTransform`
    """

    # TODO(alan): No need to use resize_wh.

    def __init__(
        self,
        target_wh,
        flip_prob,
        img_scale_range=(0.5, 2.0),
        roi_scale_range=(0.8, 1.0 / 0.8),
        min_sample_num=1,
        max_sample_num=1,
        center_aligned=True,
        inter_method=10,
        use_pyramid=True,
        pyramid_min_step=0.7,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
        min_valid_area=8,
        min_valid_clip_area_ratio=0.5,
        min_edge_size=2,
        rand_translation_ratio=0,
        rand_aspect_ratio=0,
        rand_rotation_angle=0,
        reselect_ratio=0,
        clip_bbox=True,
        rand_sampling_bbox=True,
        resize_wh=None,
        keep_aspect_ratio=False,
    ):
        self._roi_ts = AffineMatFromROIBoxGenerator(
            target_wh=target_wh,
            scale_range=roi_scale_range,
            min_sample_num=min_sample_num,
            max_sample_num=max_sample_num,
            min_valid_edge=min_edge_size,
            min_valid_area=min_valid_area,
            center_aligned=center_aligned,
            rand_scale_range=img_scale_range,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=rand_aspect_ratio,
            rand_rotation_angle=rand_rotation_angle,
            flip_prob=flip_prob,
            rand_sampling_bbox=rand_sampling_bbox,
            reselect_ratio=reselect_ratio,
        )
        self._img_ts = ImageAffineTransform(
            dst_wh=target_wh,
            inter_method=inter_method,
            border_value=0,
            use_pyramid=use_pyramid,
            pyramid_min_step=pyramid_min_step,
            pyramid_max_step=pyramid_max_step,
            pixel_center_aligned=pixel_center_aligned,
        )
        self._bbox_ts_kwargs = {
            "clip": clip_bbox,
            "min_valid_area": min_valid_area,
            "min_valid_clip_area_ratio": min_valid_clip_area_ratio,
            "min_edge_size": min_edge_size,
        }
        self._resize_wh = resize_wh
        self._use_pyramid = use_pyramid
        self._pyramid_min_step = pyramid_min_step
        self._pyramid_max_step = pyramid_max_step
        self._keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, data):
        assert isinstance(data, (dict))
        assert "img" in data.keys()
        assert "gt_boxes" in data.keys()
        img = data.get("img")
        gt_boxes = data.get("gt_boxes")
        ig_regions = data.get("ig_regions", None)

        if self._keep_aspect_ratio and self._resize_wh:
            origin_wh = img.shape[:2][::-1]
            resize_wh_ratio = float(self._resize_wh[0]) / float(
                self._resize_wh[1]
            )  # noqa
            origin_wh_ratio = float(origin_wh[0]) / float(origin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

            if resize_wh_ratio > origin_wh_ratio:
                new_wh = (
                    int(origin_wh[1] * resize_wh_ratio),
                    origin_wh[1],
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < origin_wh_ratio:
                new_wh = (
                    origin_wh[0],
                    int(origin_wh[0] / resize_wh_ratio),
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
        else:
            if self._use_pyramid:
                img = AlphaImagePyramid(
                    img,
                    scale_step=np.random.uniform(
                        self._pyramid_min_step, self._pyramid_max_step
                    ),
                )

        roi = gt_boxes.copy()

        if self._resize_wh is None:
            img_wh = img.shape[:2][::-1]
            affine_mat = AffineMat2DGenerator.identity()
        else:
            img_wh = self._resize_wh
            affine_mat = resize_affine_mat(
                img.shape[:2][::-1], self._resize_wh
            )
            roi = LabelAffineTransform(label_type="box")(
                roi, affine_mat, flip=False
            )

        for affine_aug_param in self._roi_ts(roi, img_wh):  # noqa

            new_affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=new_affine_mat, flipped=affine_aug_param.flipped
            )
            ts_img = self._img_ts(img, affine_aug_param.mat)

            ts_img_wh = ts_img.shape[:2][::-1]
            ts_gt_boxes, ts_ig_regions = _transform_bboxes(
                gt_boxes,
                ig_regions,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                affine_aug_param,
                **self._bbox_ts_kwargs,
            )

            data = pad_detection_data(
                ts_img,
                ts_gt_boxes,
                ts_ig_regions,
            )
            data["img"] = data["img"].transpose(2, 0, 1)

            return data


@OBJECT_REGISTRY.register
class PadDetData(object):
    def __init__(self, max_gt_boxes_num=100, max_ig_regions_num=100):
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num

    def __call__(self, data):

        pad_shape = list(data["gt_boxes"].shape)
        pad_shape[0] = self.max_gt_boxes_num
        data["gt_boxes_num"] = np.array(data["gt_boxes"].shape[0]).reshape(
            (1,)
        )
        data["gt_boxes"] = _pad_array(data["gt_boxes"], pad_shape, "gt_boxes")

        if "ig_regions" in data:
            pad_shape = list(data["ig_regions"].shape)
            pad_shape[0] = self.max_ig_regions_num
            data["ig_regions_num"] = np.array(
                data["ig_regions"].shape[0]
            ).reshape((1,))
            data["ig_regions"] = _pad_array(
                data["ig_regions"], pad_shape, "ig_regions"
            )

        return data


@OBJECT_REGISTRY.register
class DetInputPadding(object):
    def __init__(self, input_padding: Tuple[int]):
        assert len(input_padding) == 4
        self.input_padding = input_padding

    def __call__(self, data):
        im_hw = data["im_hw"]
        im_hw[0] += self.input_padding[2] + self.input_padding[3]
        im_hw[1] += self.input_padding[0] + self.input_padding[1]

        data["gt_boxes"][..., :4:2] += self.input_padding[0]
        data["gt_boxes"][..., 1:4:2] += self.input_padding[2]

        if "ig_regions" in data:
            data["ig_regions"][..., :4:2] += self.input_padding[0]
            data["ig_regions"][..., 1:4:2] += self.input_padding[2]

        return data
