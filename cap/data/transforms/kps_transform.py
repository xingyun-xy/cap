from typing import Tuple

import cv2
import numpy as np
from capbc.utils import _as_list

from cap.registry import OBJECT_REGISTRY
from .affine import (
    AffineAugMat,
    AffineMat2DGenerator,
    AffineMatFromROIBoxGenerator,
    ImageAffineTransform,
    LabelAffineTransform,
    Point2DAffineTransform,
    resize_affine_mat,
)
from .bbox import clip_bbox
from .common import Cast
from .detection import DetInputPadding


@OBJECT_REGISTRY.register
class KPSIterableDetRoITransform:
    def __init__(
        self,
        target_wh,
        kps_num=2,
        img_scale_range=(0.5, 2.0),
        roi_scale_range=(0.8, 1.0 / 0.8),
        min_sample_num=1,
        max_sample_num=5,
        center_aligned=True,
        inter_method=10,
        use_pyramid=False,
        pyramid_min_step=0.45,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
        min_valid_area=8,
        min_valid_clip_area_ratio=0.5,
        min_edge_size=2,
        rand_translation_ratio=0,
        rand_aspect_ratio=0,
        rand_rotation_angle=0,
        flip_prob=0.5,
        clip_bbox=True,
        rand_sampling_bbox=False,
        resize_wh=None,
        min_kps_distance=4,
        keep_aspect_ratio=False,
    ):
        self.kps_num = kps_num
        self._roi_ts = AffineMatFromROIBoxGenerator(
            target_wh=target_wh,
            scale_range=img_scale_range,
            min_sample_num=min_sample_num,
            max_sample_num=max_sample_num,
            min_valid_edge=min_edge_size,
            min_valid_area=min_valid_area,
            center_aligned=center_aligned,
            rand_scale_range=roi_scale_range,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=rand_aspect_ratio,
            rand_rotation_angle=rand_rotation_angle,
            flip_prob=flip_prob,
            rand_sampling_bbox=rand_sampling_bbox,
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
            "min_kps_dis": min_kps_distance,
        }
        self._resize_wh = resize_wh
        self._keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, data):
        img = data["img"]
        ret = data["anno"]
        all_boxes = np.asarray(ret["boxes"], dtype=np.float32)
        all_kps = np.asarray(ret["keypoints"], dtype=np.float32)
        gt_classes = np.asarray(ret["gt_classes"], dtype=np.int32)

        keep = np.where(gt_classes == 1)[0]
        assert len(keep) > 0
        gt_boxes = all_boxes[keep]
        gt_kps = all_kps[keep]
        roi = gt_boxes.copy()
        kps = gt_kps.copy()
        instance_num = kps.shape[0]
        kps = kps.reshape((instance_num, self.kps_num, -1))
        kps_2dim = kps[..., :2].copy()

        cls_label = np.expand_dims(kps[:, :, 2].copy(), axis=-1)

        if self._keep_aspect_ratio and self._resize_wh:
            orgin_wh = img.shape[:2][::-1]
            resize_wh_ratio = float(self._resize_wh[0]) / float(
                self._resize_wh[1]
            )  # noqa
            orgin_wh_ratio = float(orgin_wh[0]) / float(orgin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

            if resize_wh_ratio > orgin_wh_ratio:
                new_wh = (
                    int(orgin_wh[1] * resize_wh_ratio),
                    orgin_wh[1],
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < orgin_wh_ratio:
                new_wh = (
                    orgin_wh[0],
                    int(orgin_wh[0] / resize_wh_ratio),
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)

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

            affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=affine_mat, flipped=affine_aug_param.flipped
            )

            ts_img = self._img_ts(img, affine_aug_param.mat)
            ts_img_wh = ts_img.shape[:2][::-1]

            ts_gt_boxes, ts_gt_kps, mask = transform_bboxes_and_kps(
                gt_boxes,
                kps_2dim,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                affine_aug_param,
                **self._bbox_ts_kwargs
            )

            cls_label = cls_label[mask]
            cls_label = cls_label.reshape(-1, 1)
            ts_gt_kps = ts_gt_kps.reshape(-1, 2)
            ts_gt_kps = np.column_stack([ts_gt_kps, cls_label])
            ts_gt_kps = ts_gt_kps.reshape(-1, 3 * self.kps_num)

            data = pad_kps(
                ts_img,
                ts_gt_boxes,
                ts_gt_kps,
            )
            data["img"] = data["img"].transpose(2, 0, 1)

            return data


def pad_kps(
    img,
    gt_boxes,
    kps_label,
):

    im_hw = np.array(img.shape[:2]).reshape((2,))

    gt_boxes = np.column_stack((gt_boxes, kps_label))

    cast = Cast(np.float32)

    return {
        "img": img,
        "im_hw": cast(im_hw),
        "gt_boxes": cast(gt_boxes),
    }


def transform_bboxes_and_kps(
    gt_boxes,
    gt_kps,
    img_roi,
    affine_aug_param,
    clip=True,
    min_valid_area=8,
    min_valid_clip_area_ratio=0.5,
    min_edge_size=2,
    min_kps_dis=4,
):

    bbox_ts = LabelAffineTransform(label_type="box")
    kps_ts = Point2DAffineTransform()

    ts_gt_boxes = bbox_ts(
        gt_boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
    )

    ts_gt_kps = [
        kps_ts(i, affine_aug_param.mat) if _is_valid(i) else i
        for i in _as_list(gt_kps)
    ]
    if len(ts_gt_kps) == 1:
        ts_gt_kps = ts_gt_kps[0]

    if clip:
        clip_gt_boxes = clip_bbox(ts_gt_boxes, img_roi)
    else:
        clip_gt_boxes = ts_gt_boxes

    clip_gt_boxes, ts_gt_kps, mask = filter_bbox(
        clip_gt_boxes,
        ts_gt_kps,
        img_roi,
        allow_outside_center=True,
        min_edge_size=min_edge_size,
        min_kps_dis=min_kps_dis,
    )

    return clip_gt_boxes, ts_gt_kps, mask


def _is_valid(obj):
    return obj is not None and obj.size > 0


def filter_bbox(
    bbox,
    kps,
    img_roi,
    allow_outside_center=True,
    min_edge_size=0.1,
    min_kps_dis=4,
):
    assert bbox.ndim == 2
    assert len(img_roi) == 4
    roi = np.array(img_roi)
    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(roi[:2] <= centers, centers < roi[2:]).all(
            axis=1
        )
    mask = np.logical_and(
        mask, (bbox[:, :2] + min_edge_size < bbox[:, 2:4]).all(axis=1)
    )
    kps_mask = np.logical_and(
        mask, (np.abs(kps[:, 1, 0] - kps[:, 0, 0]) >= min_kps_dis)
    )

    kps_mask = np.logical_and(kps_mask, (kps[:, 0] >= 0).all(axis=1))
    kps_mask = np.logical_and(kps_mask, (kps[:, 1] >= 0).all(axis=1))
    kps_mask = np.logical_and(
        kps_mask, (kps[:, :, 0] < img_roi[2]).all(axis=1)
    )
    kps_mask = np.logical_and(
        kps_mask, (kps[:, :, 1] < img_roi[3]).all(axis=1)
    )
    kps = kps[kps_mask]
    bbox = bbox[kps_mask]

    return bbox, kps, kps_mask


@OBJECT_REGISTRY.register
class KPSDetInputPadding(DetInputPadding):
    def __init__(
        self,
        kps_num: int,
        input_padding: Tuple[int],
    ):
        super().__init__(input_padding)
        self.kps_num = kps_num

    def __call__(self, data):
        data = super().__call__(data)

        for i in range(self.kps_num):
            data["gt_boxes"][:, 4 + 3 * i] += self.input_padding[0]
            data["gt_boxes"][:, 5 + 3 * i] += self.input_padding[2]

        return data
