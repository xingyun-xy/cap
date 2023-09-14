# Copyright (c) Changan Auto. All rights reserved.

from dataclasses import dataclass, fields
from typing import Any, List, Union

import numpy as np
import torch

from cap.utils.apply_func import convert_numpy
from .base import BaseData

__all__ = [
    "ImgBase",
    "ImgCls",
    "ImgObjDet",
    "ImgObjDet3D",
    "ImgSemSeg",
    "BEVObject",
]


@dataclass
class ImgBase:
    """The base structure of all image-based dataset item.

    Args:
        img: Image array, should be (H, W, C) or (C, H, W).
        ori_img: Original image, should be (H, W, C) or (C, H, W).
        img_id: Any hashable object (possibly int or str) that can be
            used to identify corresponding image.
        layout: Image layout, describing the layout of image
            array (chw or hwc).
        color_space: Image color space, describing the color space
            of image array (rgb, bgr or yuv).
        img_width: The width of image array.
        img_height: The height of image array.
        flip_x: Whether the image array has been flipped in x axis.
        flip_y: Whether the image array has been flipped in y axis.
        w_scale: Scale of image compared to original input in width.
        h_scale: Scale of image compared to original input in height.
        calib: Calibration matrix of camera.
        distCoeffs: DistCoeffs matrix of camera.

    """

    img: torch.Tensor
    ori_img: torch.Tensor = None
    img_id: Any = None
    layout: str = None
    color_space: str = None
    img_width: int = None
    img_height: int = None
    flip_x: bool = False
    flip_y: bool = False
    w_scale: float = 1.0
    h_scale: float = 1.0
    calib: torch.Tensor = None
    distCoeffs: torch.Tensor = None

    def __post_init__(self):
        assert self.img is not None
        assert self.img_id is not None
        assert self.layout is not None
        assert self.color_space is not None
        assert self.img_width is not None
        assert self.img_height is not None

    def visualize(self, **kwargs):
        bgr_img = convert_numpy(self.ori_img)[:, :, ::-1].copy()

        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, BaseData):
                val.visualize(bgr_img)

        return bgr_img


@dataclass
class Cls(object):
    """The base structure of all classification dataset item.

    Args:
        cls_label: Classification label of current item.
    """

    cls_label: Any = None

    def __post_init__(self):
        assert self.cls_label is not None


@dataclass
class ImgCls(Cls, ImgBase):
    """Image classification dataset structure."""

    pass


@dataclass
class ObjDet:
    """The base structure of all object detection dataset item.

    Args:
        gt_bboxes: Ground truth bboxes, should be (N, 5), where
            the last entry in each row represents the object
            class.
        ig_regions: Ignore regions, should be (N, 5), where
            the last entry in each row represents the object
            class.
        parent_gt_bboxes: Parent ground truth bboxes, should be
            (N, 5), where the last entry in each row represents
            the object class.
        parent_ig_regions: Parent ignore regions, should be
            (N, 5), where the last entry in each row represents
            the object class.
        gt_points: Ground truth points.
        gt_masks: Ground truth mask of objects.
        track_ids: Ground truth tracking id.
    """

    gt_bboxes: Union[np.ndarray, torch.Tensor] = None
    ig_regions: Union[np.ndarray, torch.Tensor] = None
    parent_gt_bboxes: Union[np.ndarray, torch.Tensor] = None
    parent_ig_regions: Union[np.ndarray, torch.Tensor] = None
    gt_points: Union[np.ndarray, torch.Tensor] = None
    gt_masks: Any = None
    track_ids: List[Any] = None


@dataclass
class ObjDet3D:
    """The base structure of all 3d object detection dataset item.

    Args:
        dim: Ground truth object dimensions, should be (N, 3). In each
            row, the 3 entries represent length, width and height,
            respectively.
        location: Ground truth object location, should be (N, 3). In
            each row, the 3 entries represent coordinates in x (right),
            y (down) and z (forward) axis, respectively.
        rotation_y: Ground truth rotation angle in y axis, should be (N, ).
            Range from [-pi, pi].
        distCoeffs: Ground truth distortion coefficient of each object,
            should be (N, 5).
        alpha: Observation angle of each object, should be (N, ).
        location_offset: Offset between reprojected box center and 3d box
            center (x, y, z order), should be (N, 3).
    """

    dim: Union[np.ndarray, torch.Tensor] = None
    location: Union[np.ndarray, torch.Tensor] = None
    rotation_y: Union[np.ndarray, torch.Tensor] = None
    distCoeffs: Union[np.ndarray, torch.Tensor] = None
    alpha: List[float] = None
    location_offset: List[float] = None

    def __post_init__(self):
        assert self.dim is not None
        assert self.location is not None
        assert self.rotation_y is not None
        assert self.distCoeffs is not None
        assert self.alpha is not None
        assert self.location_offset is not None


@dataclass
class ImgObjDet(ObjDet, ImgBase):
    """Image object detection dataset structure."""

    pass


@dataclass
class ImgObjDet3D(ObjDet3D, ImgObjDet):
    """Image 3D object detection dataset structure."""

    pass


@dataclass
class SemSeg:
    """The base structure of all semantic segmentation dataset item.

    Args:
        gt_seg: Ground truth semantic segmentation map.
        gt_seg_stride: Stride of gt_seg (versus image).
        ignore_label: Ignore label value.
    """

    gt_seg: Union[np.ndarray, torch.Tensor] = None
    gt_seg_stride: int = 1
    ignore_label: int = -1

    def __post_init__(self):
        assert self.gt_seg is not None


@dataclass
class ImgSemSeg(SemSeg, ImgBase):
    """Image semantic segmentation dataset structure."""

    def __post_init__(self):
        super().__post_init__()
        height, width = self.gt_seg.shape
        assert (
            self.img_height / height
            == self.img_width / width
            == self.gt_seg_stride
        )


@dataclass
class BEVObject:
    """The base structure of all BEV dataset item.

    Args:
        frames: Sequence data.
        homography: Homography matrix of V views (V, 3, 3).
        front_mask: Front view mask (mask out windshield).
        intrinsics: Camera intrinsic matrix of V views
            (V, 3, 3).
        distortcoef: Distort coefficient of front view (5,).
    """

    frames: List
    homography: Union[np.ndarray, torch.Tensor]
    front_mask: Union[np.ndarray, torch.Tensor]
    intrinsics: Union[np.ndarray, torch.Tensor]
    distortcoef: Union[np.ndarray, torch.Tensor]
