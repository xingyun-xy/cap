# Copyright (c) Changan Auto. All rights reserved.
# import copy

import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from capbc.utils import deprecated
from pycocotools import mask as coco_mask
from torchvision.transforms.functional import normalize

from cap.core.affine import affine_transform, get_affine_transform
from cap.registry import OBJECT_REGISTRY
from .classification import BgrToYuv444, ConvertLayout

__all__ = [
    "ImageTransform",
    "ImageToTensor",
    "ImageBgrToYuv444",
    "ImageConvertLayout",
    "ImageNormalize",
    "Real3dTargetGenerator",
    "RepeatImage",
]


def format_angle(angle: float) -> float:
    # Convert 0~2*pi to -pi~pi
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle


def roty2alpha_z(roty: float, loc: Sequence[float]) -> float:
    """Calculate alpha_z by roty and location.

    Args:
        roty (float): rotation y
        loc (Sequence[float]): location XYZ in camera coordinate

    Returns:
        float: alpha_z
    """
    theta = format_angle(-np.arctan2(loc[0] + 1e-7, loc[2] + 1e-7))
    alpha_x = format_angle(roty + theta)
    alpha_z = format_angle(alpha_x + np.pi / 2.0)
    return alpha_z


def angle2multibin(
    angle: float, bin_centers: Sequence[float], margin: Optional[float] = 1 / 6
) -> Tuple[np.ndarray]:
    """Convert an angle to multiple bin class id and offset.

    Args:
        angle (float): angle
        bin_centers (Sequence[float]): multiple bin centers
        margin (float, optional): intersection between two centers
            Defaults to 1/6.

    Returns:
        Tuple[np.ndarray]: bin class id and offset
    """
    angle = np.array(angle, np.float32)
    bin_centers = np.array(bin_centers, np.float32)

    num_bin = len(bin_centers)
    bin_cls = np.zeros(num_bin, np.float32)
    bin_offset = np.zeros(num_bin, np.float32)
    bin_size = 2 * np.pi / num_bin
    margin_size = bin_size * margin
    range_size = bin_size / 2 + margin_size

    offsets = angle - bin_centers
    offsets[offsets > np.pi] -= 2 * np.pi
    offsets[offsets < -np.pi] += 2 * np.pi

    for i in range(num_bin):
        offset = offsets[i]
        if abs(offset) < range_size:
            bin_cls[i] = 1
            bin_offset[i] = offset

    return bin_cls, bin_offset


@OBJECT_REGISTRY.register
class ImageTransform(object):
    """Apply image affine transformation.

    Args:
        size (list or tuple): Image size, using ``(width, height)`` format.
        interpolation (int, optional): Interpolation method.
            Default: ``cv2.INTER_LINEAR``
        center_shift(list or tuple): coord shift(x, y) of center point in
            input image, x/y could be positive or negative, in pixel
        pre_resize_scale (float): Default is -1.0.
            If `pre_resize_scale` > 0, it will rescale `size` by
            pre_resize_scale, and then crop or padding.
    """

    def __init__(
        self,
        size,
        interpolation=cv2.INTER_LINEAR,
        center_shift=(0, 0),
        pre_resize_scale=-1.0,
    ):
        self.size = size
        self.interpolation = interpolation
        self.center_shift = center_shift
        self.pre_resize_scale = pre_resize_scale

    def __call__(self, data):
        img = data["img"]
        img_size = img.shape[:2][::-1]
        M = get_affine_transform(
            img_size,
            0,
            self.size,
            center_shift=self.center_shift,
            pre_resize_scale=self.pre_resize_scale,
        )
        if data.get("track_mode", 0):
            for i, img in enumerate(data["imgs"]):
                img = cv2.warpAffine(
                    img, M, tuple(self.size), flags=self.interpolation
                )
                data["imgs"][i] = img
        else:
            img = cv2.warpAffine(
                img, M, tuple(self.size), flags=self.interpolation
            )
            data["img"] = img
        data["image_transform"] = {
            "M": M,
            "original_size": img_size,
            "input_size": self.size,
        }
        return data


# TODO(runzhou.ge, 0.5): Similar to cap/data/transforms/detection.py ToTensor
@OBJECT_REGISTRY.register
class ImageToTensor(object):
    def __init__(self, from_numpy=True):
        self.from_numpy = from_numpy

    def __call__(self, data):
        if data.get("track_mode", 0):
            for i, img in enumerate(data["imgs"]):
                if self.from_numpy:
                    img = torch.from_numpy(img)
                else:
                    img = torch.Tensor(img)
                data["imgs"][i] = img
        else:
            if self.from_numpy:
                data["img"] = torch.from_numpy(data["img"])
            else:
                data["img"] = torch.Tensor(data["img"])
        return data


# TODO(runzhou.ge, 0.5): Similar to cap/data/transforms/detection.py ToTensor
@OBJECT_REGISTRY.register
class ImageBgrToYuv444(BgrToYuv444):
    def __init__(self, rgb_input=False):
        super(ImageBgrToYuv444, self).__init__(rgb_input)

    def __call__(self, data):
        if data.get("track_mode", 0):
            for i, img in enumerate(data["imgs"]):
                data["imgs"][i] = super(ImageBgrToYuv444, self).__call__(img)
        else:
            data["img"] = super(ImageBgrToYuv444, self).__call__(data["img"])
        return data


@OBJECT_REGISTRY.register
class ImageConvertLayout(ConvertLayout):
    @deprecated("Please use `ConvertLayout` instead")
    def __init__(self, hwc2chw=True):
        super(ImageConvertLayout, self).__init__(hwc2chw)

    def __call__(self, data):
        data = super(ImageConvertLayout, self).__call__(data)
        return data


# TODO(runzhou.ge, 0.5): Similar to cap/data/transforms/detection.py Normalize
@OBJECT_REGISTRY.register
class ImageNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data):
        if data.get("track_mode", 0):
            for i, img in enumerate(data["imgs"]):
                data["imgs"][i] = normalize(
                    img, self.mean, self.std, self.inplace
                )
        else:
            data["img"] = normalize(
                data["img"], self.mean, self.std, self.inplace
            )
        return data


def get_dense_loc_offset(
    wh, center, loc, dim, calib, M, dist_coeffs=None, fisheye=False
):
    w, h = wh
    radius = (w // 2, h // 2)
    n, m = radius
    x, y = int(center[0]), int(center[1])

    focal_x = calib[0][0]
    focal_y = calib[1][1]
    cx = calib[0][2]
    cy = calib[1][2]

    x_scale_up = M[0, 0]
    y_scale_up = M[1, 1]
    x_offset = M[0, 2]
    y_offset = M[1, 2]
    x_grid = np.arange(x - n, x + n + 1) / x_scale_up - x_offset / x_scale_up
    y_grid = np.arange(y - m, y + m + 1) / y_scale_up - y_offset / y_scale_up

    xy_grid = np.array(
        np.meshgrid(x_grid, y_grid), dtype=np.float32
    ).transpose(1, 2, 0)
    xy_grid_shape = xy_grid.shape

    camera = calib[:3, :3]
    if dist_coeffs is not None:
        if fisheye:
            xy_grid = cv2.fisheye.undistortPoints(
                xy_grid.reshape(-1, 1, 2), camera, dist_coeffs, None, camera
            ).squeeze(1)
        else:
            xy_grid = cv2.undistortPoints(
                xy_grid.reshape(-1, 1, 2), camera, dist_coeffs, None, camera
            ).squeeze(1)

    xy_grid = xy_grid.reshape(xy_grid_shape)

    xy_grid[:, :, 0] = loc[0] - (xy_grid[:, :, 0] - cx) / focal_x * loc[2]
    xy_grid[:, :, 1] = (
        loc[1] - (xy_grid[:, :, 1] - cy) / focal_y * loc[2] - dim[0] / 2.0
    )

    return xy_grid


def get_gaussian2D(wh, alpha=0.54, eps=1e-6, sigma=None):
    radius = (wh / 2 * alpha).astype("int32")
    if sigma is None:
        sigma = (radius * 2 + 1) / 6
    else:
        sigma = np.array(sigma)

    n, m = radius
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    sx2, sy2 = np.array(sigma) ** 2
    heatmap = np.exp(-(y * y) / (2 * sy2 + eps) - (x * x) / (2 * sx2 + eps))
    heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
    return heatmap


def get_reg_map(wh, value):
    w, h = wh
    if isinstance(value, (list, tuple, np.ndarray)):
        reg_map = np.tile(value, (h, w, 1))
    else:
        reg_map = np.tile(value, (h, w))
    return reg_map


def draw_heatmap(
    heatmap, insert_hm, cxy, reg_map_list=(), insert_reg_map_list=(), op="max"
):
    """Draw object heatmaps to corresponding ground truth heatmaps.

    Args:
        heatmap (np.ndarray): Ground truth heatmap.
        insert_hm (np.ndarray): Object heatmap.
        cxy (tuple or np.ndarray or list): Center point coordinates of
            the object on ground truth heatmap.
        reg_map_list (list of np.ndarray): List of ground truth heatmaps.
        insert_reg_map_list (list of np.ndarray): List of object heatmaps.
            Object heatmaps in `insert_reg_map_list` will be drawn on ground
            truth heatmaps in `reg_map_list`, paired by its index in each list.
        op (str, 'max' or 'overwrite', default: `max`):
            If a pixel on ground truth heatmap is already be drawn, its value
            will be the maximum value between the new and old value if `op` is
            'max', or overwritten by new value if `op` is 'overwrite'

    """
    x, y = int(cxy[0]), int(cxy[1])
    ry, rx = (np.array(insert_hm.shape[:2]) - 1) // 2
    height, width = heatmap.shape[:2]

    left, right = min(x, rx), min(width - x, rx + 1)
    top, bottom = min(y, ry), min(height - y, ry + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_insert_hm = insert_hm[
        ry - top : ry + bottom, rx - left : rx + right
    ]
    if reg_map_list and insert_reg_map_list:
        masked_reg_map_list = [
            reg_map[y - top : y + bottom, x - left : x + right]
            for reg_map in reg_map_list
        ]
        masked_insert_reg_map_list = [
            insert_reg_map[ry - top : ry + bottom, rx - left : rx + right]
            for insert_reg_map in insert_reg_map_list
        ]

    if min(masked_insert_hm.shape) > 0 and min(masked_heatmap.shape) > 0:
        if op == "max":
            mask = masked_insert_hm > masked_heatmap
            masked_heatmap[mask] = masked_insert_hm[mask]
            if reg_map_list and insert_reg_map_list:
                for (masked_reg_map, masked_insert_reg_map) in zip(
                    masked_reg_map_list, masked_insert_reg_map_list
                ):
                    masked_reg_map[mask] = masked_insert_reg_map[mask]
        elif op == "overwrite":
            masked_heatmap[:] = masked_insert_hm
            if reg_map_list and insert_reg_map_list:
                for (masked_reg_map, masked_insert_reg_map) in zip(
                    masked_reg_map_list, masked_insert_reg_map_list
                ):
                    masked_reg_map[:] = masked_insert_reg_map
        else:
            raise NotImplementedError


def draw_reg_map(heatmap, insert_hm, cxy, op="max"):
    x, y = int(cxy[0]), int(cxy[1])
    ry, rx = (np.array(insert_hm.shape[:2]) - 1) // 2
    height, width = heatmap.shape[:2]

    left, right = min(x, rx), min(width - x, rx + 1)
    top, bottom = min(y, ry), min(height - y, ry + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_insert_hm = insert_hm[
        ry - top : ry + bottom, rx - left : rx + right
    ]
    if min(masked_insert_hm.shape) > 0 and min(masked_heatmap.shape) > 0:
        if op == "max":
            mask = masked_insert_hm > masked_heatmap
            masked_heatmap[mask] = masked_insert_hm[mask]
        elif op == "overwrite":
            masked_heatmap[:] = masked_insert_hm
        else:
            raise NotImplementedError


def xywh_to_x1y1x2y2(bboxes):
    if isinstance(bboxes, (list, tuple)):
        bboxes = np.array(bboxes)

    bboxes[..., 2:] += bboxes[..., :2]
    return bboxes


def fill_mask_by_bbox(mask, bbox, value=1.0):
    bbox = list(map(int, bbox))
    mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = value
    return mask


# TODO(runzhou.ge, 0.5): move this to cap/models/heads/targets following
# fcos/target.py format
@OBJECT_REGISTRY.register
class Real3dTargetGenerator(object):
    """Generate gound truth labels for real3d.

    Args:
        num_classes (int): Number of classes
        focal_length_default (float): The default focal length
        input_size (tuple or list): The width and heigh of the input images
        category_id_dict (dict): A mapping from raw category_id to training
            category_id which starts from 0
        origin_image_shape (tuple): The shape of the origin images
        head_channels (dict): a dict like {'hm':3,'dep':1,} to config output
            device nums.
        down_stride (int): The downstride between input size to output
            result size
        max_objs (int): Maximum number of objects used in the training and
            inference. This number should be large enough
        center_shift (list or tuple): coord shift(x, y) of center point in
            input image, x/y could be positive or negative, in pixel
        pop_anno (bool): whether to pop annotations. In validation, we need to
            keep annotions but training not.
        undistort (bool): whether undistort image points.
        fisheye(bool): whether process fisheye data.
        pre_resize_scale (float): Default is -1.0.
            If `pre_resize_scale` > 0, it will rescale `size` by
            pre_resize_scale, and then crop or padding.
        min_wh (float): Default is 0.0.
            If the w or h of bbox in output scale was less than or equal to
            `min_wh`, this bbox will be ignored.
        multibin_centers (tuple of float): Default is [0, pi/2, pi, -pi/2].
            The domain is [-pi, +pi].
    """

    def __init__(
        self,
        num_classes,
        focal_length_default,
        input_size,
        category_id_dict,
        origin_image_shape=(2160, 3840),
        head_channels=None,
        max_depth=300.0,
        down_stride=4,
        max_objs=100,
        center_shift=(0, 0),
        pop_anno=True,
        undistort=False,
        fisheye=False,
        pre_resize_scale=-1.0,
        min_wh=0.0,
        multibin_centers=(0.0, math.pi / 2, math.pi, -math.pi / 2),
    ):
        self.num_classes = num_classes
        self.focal_length_default = focal_length_default
        self.down_stride = down_stride
        self.max_objs = max_objs
        self.input_size = input_size
        self.category_id_dict = category_id_dict
        self.origin_image_shape = origin_image_shape
        if head_channels is None:
            self.head_channels = OrderedDict(
                hm=num_classes, rot=2, dep=1, dim=3, loc_offset=2, wh=2
            )
        else:
            self.head_channels = head_channels
        self.max_depth = max_depth
        self.center_shift = center_shift
        self.undistort = undistort
        self.fisheye = fisheye
        width, height = self.input_size

        assert width % self.down_stride == 0 and height % self.down_stride == 0

        self.output_width = int(width // self.down_stride)
        self.output_height = int(height // self.down_stride)
        self.pop_anno = pop_anno
        self.pre_resize_scale = pre_resize_scale
        self.min_wh = min_wh
        self.multibin_centers = np.array(multibin_centers, np.float32)

    def _parse_annotations(self, annotations):
        if len(annotations) == 0:
            return annotations
        for obj in annotations:
            if "in_camera" in obj:
                obj.update(obj.pop("in_camera"))
        return annotations

    def __call__(self, data):
        """Generate labels.

        Args:
            data (dict): Type is ndarray
                The dict contains at leaset below keys:
                    annotations, calibration, image_transform, ignore_mask

        Returns (dict): Type is ndarray

        """
        category_id_dict = self.category_id_dict
        if self.pop_anno:
            annotations = data.pop("annotations")
        else:
            annotations = data["annotations"]
        annotations = self._parse_annotations(annotations)
        calibration = data["calibration"]
        focal_length = (calibration[0, 0] + calibration[1, 1]) / 2
        dist_coeffs = None
        if self.undistort:
            dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
        trans_size = data["image_transform"]["original_size"]
        output_width = self.output_width
        output_height = self.output_height

        M = get_affine_transform(
            trans_size,
            0,
            [output_width, output_height],
            center_shift=self.center_shift,
            pre_resize_scale=self.pre_resize_scale,
        )

        ignore_mask = coco_mask.decode(data["ignore_mask"]).astype(np.uint8)

        if ignore_mask.shape != self.origin_image_shape:
            data["valid"] = False
        ignore_mask = cv2.warpAffine(
            ignore_mask,
            M,
            (output_width, output_height),
            flags=cv2.INTER_NEAREST,
        )
        ignore_mask = ignore_mask.astype(np.float32)[:, :, np.newaxis]

        # build gt heatmaps
        target = {}
        for head_name, channel_num in self.head_channels.items():
            if channel_num > 1:
                target[head_name] = np.zeros(
                    (output_height, output_width, channel_num),
                    dtype=np.float32,
                )
            else:
                target[head_name] = np.zeros(
                    (output_height, output_width), dtype=np.float32
                )
        target["loc_offset"] = np.full(
            (output_height, output_width, 2), -1e7, dtype=np.float32
        )

        if "track_offset" in target:
            target["track_offset"] = np.full(
                (output_height, output_width, 2), -1e7, dtype=np.float32
            )

        weight_hm = np.zeros((output_height, output_width), dtype=np.float32)
        point_pos_mask = np.zeros(
            (output_height, output_width), dtype=np.float32
        )

        # pointwise rotation loss and corner loss
        rot_y_ = np.zeros((self.max_objs), dtype=np.float32)
        alpha_z_ = np.zeros((self.max_objs), dtype=np.float32)
        loc_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        dim_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        ind_ = np.zeros((self.max_objs), dtype=np.int64)
        ind_mask_ = np.zeros((self.max_objs), dtype=np.float32)
        bin_cls_ = np.zeros(
            (self.max_objs, len(self.multibin_centers)), dtype=np.float32
        )
        bin_offset_ = np.zeros(
            (self.max_objs, len(self.multibin_centers)), dtype=np.float32
        )

        annos_valid = {
            "cls_id": [],
            "ct_int": [],
            "wh": [],
            "depth": [],
            "annos": [],
            "bbox": [],
        }
        if "track_offset" in target:
            trk_id = {}
            annos_valid.update({"track_offset": []})
            # this case, len(annotations)==2
            for ann in annotations[0]:
                cls_id = int(ann["category_id"])
                if cls_id <= -99 or ann.get("ignore", False):
                    continue
                if len(annos_valid["annos"]) >= self.max_objs:
                    break
                if ann["bbox_2d"] is not None:
                    bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                else:
                    bbox = xywh_to_x1y1x2y2(ann["bbox"])

                bbox[:2] = affine_transform([bbox[:2]], M)
                bbox[2:] = affine_transform([bbox[2:]], M)

                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)
                _wh = bbox[2:] - bbox[:2]
                ct = (bbox[:2] + bbox[2:]) / 2
                if np.any(_wh <= self.min_wh) or min(ann["dim"]) < 0:
                    continue
                trk_id_key = (
                    str(ann["category_id"]) + "_" + str(ann["track_id"])
                )
                trk_id[trk_id_key] = ct
            for _ann_idx, ann in enumerate(annotations[1]):
                cls_id = int(category_id_dict[ann["category_id"]])
                if cls_id <= -99 or ann.get("ignore", False):
                    continue
                if len(annos_valid["annos"]) >= self.max_objs:
                    break
                if ann["bbox_2d"] is not None:
                    bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                else:
                    bbox = xywh_to_x1y1x2y2(ann["bbox"])

                bbox[:2] = affine_transform([bbox[:2]], M)
                bbox[2:] = affine_transform([bbox[2:]], M)

                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)
                _wh = bbox[2:] - bbox[:2]
                ct = (bbox[:2] + bbox[2:]) / 2
                ct_int = tuple(np.rint(ct).astype(np.int32).tolist())
                if np.any(_wh <= self.min_wh) or min(ann["dim"]) < 0:
                    fill_mask_by_bbox(ignore_mask, bbox, 1)
                    continue

                if ann["depth"] > self.max_depth:
                    fill_mask_by_bbox(ignore_mask, bbox, 1)
                    continue
                annos_valid["cls_id"].append(cls_id)
                annos_valid["ct_int"].append(ct_int)
                annos_valid["wh"].append(_wh)
                annos_valid["depth"].append(ann["depth"])
                annos_valid["annos"].append(ann)
                trk_id_key = (
                    str(ann["category_id"]) + "_" + str(ann["track_id"])
                )
                if trk_id_key in trk_id:
                    track_offset = (ct - trk_id[trk_id_key]).tolist()
                else:
                    track_offset = [
                        (np.random.random() - 0.5) * 1e-7,
                        (np.random.random() - 0.5) * 1e-7,
                    ]
                annos_valid["track_offset"].append(track_offset)
                ann["track_offset"] = track_offset
        else:
            for _ann_idx, ann in enumerate(annotations):
                cls_id = int(category_id_dict[ann["category_id"]])
                if cls_id <= -99 or ann.get("ignore", False):
                    continue
                if len(annos_valid["annos"]) >= self.max_objs:
                    break
                if ann["bbox_2d"] is not None:
                    bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                else:
                    bbox = xywh_to_x1y1x2y2(ann["bbox"])

                bbox[:2] = affine_transform([bbox[:2]], M)
                bbox[2:] = affine_transform([bbox[2:]], M)

                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)
                _wh = bbox[2:] - bbox[:2]
                ct = (bbox[:2] + bbox[2:]) / 2
                ct_int = tuple(np.rint(ct).astype(np.int32).tolist())
                if np.any(_wh <= self.min_wh) or min(ann["dim"]) < 0:
                    fill_mask_by_bbox(ignore_mask, bbox, 1)
                    continue

                if ann["depth"] > self.max_depth:
                    fill_mask_by_bbox(ignore_mask, bbox, 1)
                    continue
                annos_valid["cls_id"].append(cls_id)
                annos_valid["ct_int"].append(ct_int)
                annos_valid["wh"].append(_wh)
                annos_valid["depth"].append(ann["depth"])
                annos_valid["annos"].append(ann)
                annos_valid["bbox"].append(bbox)

        dep_index = np.argsort(annos_valid["depth"])[::-1]
        if len(dep_index) == 0:
            data["valid"] = False
        for count_id, j in enumerate(dep_index):
            cls_id = annos_valid["cls_id"][j]
            ct_int = annos_valid["ct_int"][j]
            _wh = annos_valid["wh"][j]
            ann = annos_valid["annos"][j]
            ind_[count_id] = ct_int[1] * output_width + ct_int[0]
            dim_[count_id] = ann["dim"]
            loc_[count_id] = ann["location"]
            ind_mask_[count_id] = 1
            rot_y_[count_id] = ann["rotation_y"]
            bbox = annos_valid["bbox"][j]
            fill_mask_by_bbox(ignore_mask, bbox, 0)

            alpha_z_i = roty2alpha_z(ann["rotation_y"], ann["location"])
            alpha_z_[count_id] = alpha_z_i
            _cls, _off = angle2multibin(alpha_z_i, self.multibin_centers)
            bin_cls_[count_id] = _cls
            bin_offset_[count_id] = _off

            insert_hm = get_gaussian2D(_wh)
            insert_hm_wh = insert_hm.shape[:2][::-1]
            draw_reg_map(
                target["hm"][:, :, cls_id], insert_hm, ct_int, op="max"
            )
            draw_reg_map(weight_hm, insert_hm, ct_int, op="max")
            reg_map_loc_offset = get_dense_loc_offset(
                insert_hm_wh,
                ct_int,
                ann["location"],
                ann["dim"],
                calibration,
                M,
                dist_coeffs=dist_coeffs,
                fisheye=self.fisheye,
            )
            reg_map_dim = get_reg_map(insert_hm_wh, ann["dim"])
            reg_map_wh = get_reg_map(insert_hm_wh, _wh)
            reg_map_depth = get_reg_map(insert_hm_wh, ann["depth"])
            draw_reg_map(
                target["loc_offset"], reg_map_loc_offset, ct_int, op="max"
            )
            draw_reg_map(target["dim"], reg_map_dim, ct_int, op="max")
            draw_reg_map(target["dep"], reg_map_depth, ct_int, op="max")
            draw_reg_map(target["wh"], reg_map_wh, ct_int, op="max")
            if "track_offset" in target:
                track_offset = annos_valid["track_offset"][j]
                reg_map_track_offset = get_reg_map((1, 1), track_offset)
                draw_reg_map(
                    target["track_offset"],
                    reg_map_track_offset,
                    ct_int,
                    op="max",
                )

            point_pos_mask[ct_int[1], ct_int[0]] = 1

        # normalize depth
        target["dep"] *= self.focal_length_default / focal_length
        target["loc_offset"] *= self.focal_length_default / focal_length

        target.update(
            {
                "dep": target["dep"][:, :, np.newaxis],
                "weight_hm": weight_hm[:, :, np.newaxis],
                "point_pos_mask": point_pos_mask[:, :, np.newaxis],
                "ignore_mask": ignore_mask,
            }
        )

        for k in target.keys():
            target[k] = target[k].transpose(2, 0, 1)

        target.update(
            {
                "ind_": ind_,
                "dim_": dim_,
                "loc_": loc_,
                "ind_mask_": ind_mask_,
                "rot_y_": rot_y_,
                "alpha_z_": alpha_z_,
                "bin_cls_": bin_cls_,
                "bin_offset_": bin_offset_,
            }
        )

        data["target"] = target

        return data


# TODO (runzhou.ge, 0.5): update this class to make it more meaningful
@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("ParseDataReal3DMultitask")
class RepeatImage(object):
    """Repeat a single image multiple times.

    Args:
        repeat_times (int): The repeat times of the single image

    """

    def __init__(
        self,
        times: int = 1,
        input_key: str = "img",
        output_key: str = "imgs",
        replace: bool = True,
        repeat_times: Optional[int] = None,
    ):
        if repeat_times is not None:
            times = repeat_times
            warnings.warn("repeat_times is deprecated, please use times.")
        self.times = times
        assert self.times > 0
        self.input_key = input_key
        self.output_key = output_key
        self.replace = replace

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Repeat a single image multiple times.

        Args:
            data (dict): the key we use dict is img

        Returns (dict): we pop out img and insert imgs into it

        """
        assert self.input_key in data
        if self.replace:
            value = data.pop(self.input_key)
        else:
            value = data[self.input_key]

        data[self.output_key] = [value for _ in range(self.times)]
        return data

    def __repr__(self):
        return "RepeatImage"
