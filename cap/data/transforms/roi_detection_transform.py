# flake8: noqa

from copy import deepcopy

import cv2
import numpy as np

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .affine import (
    AffineAugMat,
    AffineMat2DGenerator,
    AffineMatFromROIBoxGenerator,
    ImageAffineTransform,
    LabelAffineTransform,
    Point2DAffineTransform,
    _pad_array,
    resize_affine_mat,
)
from .bbox import (
    clip_bbox,
    remap_bbox_label_by_area,
    remap_bbox_label_by_clip_area_ratio,
)
from .common import Cast
from .detection import DetInputPadding


@OBJECT_REGISTRY.register
class ROIDetectionIterableDetRoITransform:
    def __init__(
        self,
        target_wh,
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
        allow_outside_center=True,
        rand_sampling_bbox=False,
        resize_wh=None,
        keep_aspect_ratio=False,
        merge_subbox=False,
        max_subbox_num=0,
        max_gt_boxes_num=100,
        max_ig_regions_num=100,
    ):
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
            "allow_outside_center": allow_outside_center,
        }
        self._target_wh = target_wh
        self._resize_wh = resize_wh
        self._keep_aspect_ratio = keep_aspect_ratio
        self.merge_subbox = merge_subbox
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num
        if merge_subbox:
            assert (
                max_subbox_num > 0
            ), "when merge_subbox is true, max_subbox_num must bigger then 0."
        self.max_subbox_num = max_subbox_num

    def __call__(self, data):
        def _pad_subbox(subbox_list, max_subbox_num):
            pad_subbox_list = []
            for subbox_i in subbox_list:
                if len(subbox_i) <= max_subbox_num:
                    temp_bbox = np.array(subbox_i, dtype=np.float32)
                    pad_shape = list(temp_bbox.shape)
                    pad_shape[0] = max_subbox_num
                    boxes = _pad_array(temp_bbox, pad_shape, "wheel_box")
                    pad_subbox_list.append(boxes)
                else:
                    temp_bbox = np.array(subbox_i, dtype=np.float32)[
                        :max_subbox_num, :
                    ]
                    pad_subbox_list.append(temp_bbox)
            if len(pad_subbox_list) > 0:
                pad_subbox_list = np.array(pad_subbox_list, dtype=np.float32)
            else:
                pad_subbox_list = np.zeros(
                    (0, max_subbox_num, 5), dtype=np.float32
                )

            return pad_subbox_list

        def _merge_box(parent_gt_boxes, subbox_gt_boxes, max_subbox_num):

            merged_parent_bbox_list = []
            merged_subbox_list = []
            for parent_box_i, subbox_box_i in zip(
                parent_gt_boxes, subbox_gt_boxes
            ):
                merge_flag = False
                for idx, merged_parent_box_i in enumerate(
                    merged_parent_bbox_list
                ):
                    if all(parent_box_i[:4] == merged_parent_box_i[:4]):
                        merge_flag = True
                        break

                    if not merge_flag:
                        merged_parent_bbox_list.append(parent_box_i)
                        merged_subbox_list.append([subbox_box_i])
                    else:
                        merged_subbox_list[idx].append(subbox_box_i)
                        # merge hard sample
                        merged_parent_bbox_list[idx][4] = (
                            merged_parent_bbox_list[idx][4] * parent_box_i[4]
                        )  # noqa

            merged_parent_bbox_list = (
                np.array(merged_parent_bbox_list, dtype=np.float32)
                if len(merged_parent_bbox_list) > 0
                else np.zeros((0, 5), dtype=np.float32)
            )  # noqa

            merged_subbox_list = _pad_subbox(
                merged_subbox_list, max_subbox_num
            )

            return merged_parent_bbox_list, merged_subbox_list

        # assert isinstance(data, (list, tuple))
        # assert len(data) in [2, 3], 'Invalid number of input data %s' % len(data)  # noqa
        # if len(data) == 2:
        #     data = tuple(data) + (None, )
        img, gt_boxes, ig_regions = (
            data["img"],
            data["gt_boxes"],
            data.get("ig_regions", None),
        )
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

            affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=affine_mat, flipped=affine_aug_param.flipped
            )

            ts_img = self._img_ts(img, affine_aug_param.mat)
            ts_img_wh = ts_img.shape[:2][::-1]

            parent_gt_boxes, parent_ig_regions = data.get(
                "parent_gt_boxes", None
            ), data.get("parent_ig_regions", None)
            if parent_gt_boxes is not None and parent_ig_regions is not None:

                (
                    ts_gt_boxes,
                    ts_ig_regions,
                    ts_parent_gt_boxes,
                    ts_parent_ig_regions,
                ) = _transform_bboxes_with_parent(
                    gt_boxes,
                    ig_regions,
                    parent_gt_boxes,
                    parent_ig_regions,
                    (0, 0, ts_img_wh[0], ts_img_wh[1]),
                    affine_aug_param,
                    **self._bbox_ts_kwargs
                )
                assert len(ts_parent_gt_boxes) == len(ts_gt_boxes)

                if self.merge_subbox:
                    ts_parent_gt_boxes, ts_gt_boxes = _merge_box(
                        ts_parent_gt_boxes, ts_gt_boxes, self.max_subbox_num
                    )

                data = {
                    "img": ts_img,
                    "gt_boxes": ts_gt_boxes,
                    "ig_regions": ts_ig_regions,
                    "parent_gt_boxes": ts_parent_gt_boxes,
                    "parent_ig_regions": ts_parent_ig_regions,
                }
            else:
                ts_gt_boxes, ts_ig_regions = _transform_bboxes(
                    gt_boxes,
                    ig_regions,
                    (0, 0, ts_img_wh[0], ts_img_wh[1]),
                    affine_aug_param,
                    **self._bbox_ts_kwargs
                )

                data = {
                    "img": ts_img,
                    "gt_boxes": ts_gt_boxes,
                    "ig_regions": ts_ig_regions,
                }

            data = ped_roi_detection(
                data,
                self._target_wh,
                self.max_gt_boxes_num,
                self.max_ig_regions_num,
            )
            data["img"] = data["img"].transpose(2, 0, 1)
            return data


def filter_bbox_mask(
    bbox, img_roi, allow_outside_center=True, min_edge_size=0.1
):
    """Filter out bboxes and return mask.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number
        of bounding boxes. The second axis represents attributes
        of the bounding box. Specifically, these are
        :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates,
        which stay intact during bounding box transformations.
    img_roi : tuple or list
        Tuple of length 4. :math:`(x_{min}, y_{min}, x_{max}, y_{max})`
    allow_outside_center : bool, default=True
        If `False`, remove bounding boxes which have centers
        outside cropping area.
    min_edge_size : float, deftaut=0.1
        The minimal length of edge to be valid.

    Returns
    -------
    numpy.ndarray
        Filter mask shape (N,).
    """
    assert bbox.ndim == 2
    assert len(img_roi) == 4
    roi = np.array(img_roi)
    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        #
        mask = np.logical_and(roi[:2] <= centers, centers < roi[2:]).all(
            axis=1
        )
        # only non hard
        # mask = np.logical_and((roi[:2] <= centers),
        #                       centers < roi[2:]).all(axis=1)
        # mask = np.logical_and(mask,
        #                       bbox[:, 4] > 0)

    mask = np.logical_and(
        mask, (bbox[:, :2] + min_edge_size < bbox[:, 2:4]).all(axis=1)
    )
    return mask


def filter_bbox(
    bbox,
    kps,
    img_roi,
    allow_outside_center=True,
    min_edge_size=0.1,
    min_kps_dis=4,
):
    """Filter out bboxes.

    Filter out bboxes if its center outside img_roi or the maximum length
    of width and height is smaller than min_edge_size
    """
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

    if kps.size == 0:
        bbox = bbox[mask]
        return bbox

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


def _transform_bboxes(
    gt_boxes,
    ig_regions,
    img_roi,
    affine_aug_param,
    gt_kps,
    clip=True,
    min_valid_area=8,
    min_valid_clip_area_ratio=0.5,
    min_edge_size=2,
    min_kps_dis=4,
):
    bbox_ts = LabelAffineTransform(label_type="box")

    ts_gt_boxes = bbox_ts(
        gt_boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
    )
    if clip:
        clip_gt_boxes = clip_bbox(ts_gt_boxes, img_roi)
    else:
        clip_gt_boxes = ts_gt_boxes

    ts_gt_kps = np.zeros((0, 1, 2), dtype=np.float32)
    if gt_kps.size != 0:
        kps_ts = Point2DAffineTransform()
        ts_gt_kps = [
            kps_ts(i, affine_aug_param.mat) if i.size > 0 else i
            for i in _as_list(gt_kps)
        ]
        if len(ts_gt_kps) == 1:
            ts_gt_kps = ts_gt_kps[0]
        clip_gt_boxes, ts_gt_kps, mask = filter_bbox(
            clip_gt_boxes,
            img_roi,
            kps=ts_gt_kps,
            allow_outside_center=True,
            min_edge_size=min_edge_size,
            min_kps_dis=min_kps_dis,
        )

        return clip_gt_boxes, ts_gt_kps, mask

    clip_gt_boxes = remap_bbox_label_by_area(clip_gt_boxes, min_valid_area)
    clip_gt_boxes = remap_bbox_label_by_clip_area_ratio(
        ts_gt_boxes, clip_gt_boxes, min_valid_clip_area_ratio
    )
    clip_gt_boxes = filter_bbox(
        clip_gt_boxes,
        img_roi,
        kps=ts_gt_kps,
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


def _transform_bboxes_with_parent(
    gt_boxes,
    ig_regions,
    parent_gt_boxes,
    parent_ig_regions,
    img_roi,
    affine_aug_param,
    clip=True,
    min_valid_area=8,
    min_valid_clip_area_ratio=0.5,
    min_edge_size=2,
    allow_outside_center=True,
):
    # TODO(zhengwei.hu): add notes.
    bbox_ts = LabelAffineTransform(label_type="box")

    ts_gt_boxes = bbox_ts(
        gt_boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
    )
    ts_parent_gt_boxes = bbox_ts(
        parent_gt_boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
    )
    if clip:
        clip_gt_boxes = clip_bbox(ts_gt_boxes, img_roi)
        clip_parent_gt_boxes = clip_bbox(ts_parent_gt_boxes, img_roi)
    else:
        clip_gt_boxes = ts_gt_boxes
        clip_parent_gt_boxes = ts_parent_gt_boxes

    clip_gt_boxes = remap_bbox_label_by_area(clip_gt_boxes, min_valid_area)
    clip_gt_boxes = remap_bbox_label_by_clip_area_ratio(
        ts_gt_boxes, clip_gt_boxes, min_valid_clip_area_ratio
    )

    clip_parent_gt_boxes = remap_bbox_label_by_area(
        clip_parent_gt_boxes, min_valid_area
    )
    clip_parent_gt_boxes = remap_bbox_label_by_clip_area_ratio(
        ts_parent_gt_boxes, clip_parent_gt_boxes, min_valid_clip_area_ratio
    )

    clip_gt_mask = filter_bbox_mask(
        clip_gt_boxes,
        img_roi,
        allow_outside_center=allow_outside_center,
        min_edge_size=min_edge_size,
    )
    clip_parent_gt_mask = filter_bbox_mask(
        clip_parent_gt_boxes,
        img_roi,
        allow_outside_center=allow_outside_center,
        min_edge_size=min_edge_size,
    )

    clip_gt_boxes = clip_gt_boxes[clip_gt_mask & clip_parent_gt_mask]
    clip_parent_gt_boxes = clip_parent_gt_boxes[
        clip_gt_mask & clip_parent_gt_mask
    ]

    if ig_regions is not None:
        ts_ig_regions = bbox_ts(
            ig_regions, affine_aug_param.mat, flip=affine_aug_param.flipped
        )
        ts_parent_ig_regions = bbox_ts(
            parent_ig_regions,
            affine_aug_param.mat,
            flip=affine_aug_param.flipped,
        )
        if clip:
            clip_ig_regions = clip_bbox(ts_ig_regions, img_roi)
            clip_parent_ig_regions = clip_bbox(ts_parent_ig_regions, img_roi)
        else:
            clip_ig_regions = ts_ig_regions
            clip_parent_ig_regions = ts_parent_ig_regions
    else:
        clip_ig_regions = None
        clip_parent_ig_regions = None

    return (
        clip_gt_boxes,
        clip_ig_regions,
        clip_parent_gt_boxes,
        clip_parent_ig_regions,
    )


def ped_roi_detection(data, target_wh, max_gt_boxes_num, max_ig_regions_num):
    img, gt_boxes, ig_regions = (
        data["img"],
        data["gt_boxes"],
        data.get("ig_regions", None),
    )

    im_hw = np.array(img.shape[:2]).reshape((2,))

    # Cast gt_boxes
    cast = Cast(np.float32)
    im_hw = cast(im_hw)
    gt_boxes = cast(gt_boxes)

    if ig_regions is None:
        return {
            "img": img,
            "im_hw": im_hw,
            "gt_boxes": gt_boxes,
        }

    # Cast ig_regions
    ig_regions = cast(ig_regions)

    parent_gt_boxes, parent_ig_regions = data.get(
        "parent_gt_boxes", None
    ), data.get("parent_ig_regions", None)
    if parent_gt_boxes is not None and parent_ig_regions is not None:
        # Cast ig_regions
        parent_gt_boxes = cast(parent_gt_boxes)
        parent_ig_regions = cast(parent_ig_regions)
        return {
            "img": img,
            "im_hw": im_hw,
            "gt_boxes": gt_boxes,
            "ig_regions": ig_regions,
            "parent_gt_boxes": parent_gt_boxes,
            "parent_ig_regions": parent_ig_regions,
        }
    return {
        "img": img,
        "im_hw": im_hw,
        "gt_boxes": gt_boxes,
        "ig_regions": ig_regions,
    }


@OBJECT_REGISTRY.register
class ROIDetectionIterableMultiDetRoITransform:
    def __init__(
        self,
        target_wh,
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
        allow_outside_center=True,
        rand_sampling_bbox=False,
        resize_wh=None,
        keep_aspect_ratio=False,
        max_subbox_num=0,
        ig_region_match_thresh=0.8,
        min_valid_sub_area=8,
        min_sub_edge_size=8,
    ):
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
            "allow_outside_center": allow_outside_center,
            "min_valid_sub_area": min_valid_sub_area,
            "min_sub_edge_size": min_sub_edge_size,
        }
        self._resize_wh = resize_wh
        self._keep_aspect_ratio = keep_aspect_ratio
        self.max_subbox_num = max_subbox_num
        self.ig_region_match_thresh = ig_region_match_thresh
        self.target_wh = target_wh

    def __call__(self, data):

        img = data["img"]
        gt_boxes = data["gt_boxes"]
        ig_regions = data.get("ig_regions", None)
        parent_gt_boxes = data["parent_gt_boxes"]
        parent_ig_regions = None

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

        roi = parent_gt_boxes.copy()

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

        parent_gt_boxes, gt_boxes = self._merge_sub_box(
            parent_gt_boxes,
            gt_boxes,
        )
        ig_regions = self._merge_ig_regions(
            parent_gt_boxes,
            ig_regions,
        )

        for affine_aug_param in self._roi_ts(roi, img_wh):
            cur_affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=cur_affine_mat, flipped=affine_aug_param.flipped
            )

            ts_img = self._img_ts(img, affine_aug_param.mat)
            ts_img_wh = ts_img.shape[:2][::-1]

            (
                ts_gt_boxes,
                ts_ig_regions,
                ts_parent_gt_boxes,
                ts_parent_ig_regions,
            ) = _transform_multi_bboxes_with_parent(
                gt_boxes,
                ig_regions,
                parent_gt_boxes,
                parent_ig_regions,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                affine_aug_param,
                **self._bbox_ts_kwargs
            )
            assert len(ts_parent_gt_boxes) == len(ts_gt_boxes)

            if len(ts_img.shape) == 3:
                assert ts_img.shape[2] in [
                    1,
                    3,
                ], "ts_img should with shape HWC"

            pad_shape = list(ts_img.shape)
            pad_shape[0] = self.target_wh[1]
            pad_shape[1] = self.target_wh[0]
            im_hw = np.array(ts_img.shape[:2]).reshape((2,))
            ts_img = _pad_array(ts_img, pad_shape, "img")

            # gt_boxes
            FloatCast = Cast(np.float32)
            im_hw = FloatCast(im_hw)
            ts_gt_boxes = FloatCast(ts_gt_boxes)
            ts_parent_gt_boxes = FloatCast(ts_parent_gt_boxes)

            ret_dict = {
                "img": ts_img,
                "im_hw": im_hw,
                "gt_boxes": ts_gt_boxes,
                "parent_gt_boxes": ts_parent_gt_boxes,
            }

            if ts_ig_regions is not None:
                ts_ig_regions = FloatCast(ts_ig_regions)
                ret_dict.update(
                    {
                        "ig_regions": ts_ig_regions,
                    }
                )

            if ts_parent_ig_regions is not None:

                ts_parent_ig_regions = FloatCast(ts_parent_ig_regions)
                ret_dict.update(
                    {
                        "parent_ig_regions": ts_parent_ig_regions,
                    }
                )

            ret_dict["img"] = ret_dict["img"].transpose(2, 0, 1)
            return ret_dict

    def _merge_sub_box(self, parent_gt_boxes, subbox_gt_boxes):
        merged_parent_bbox_list = []
        merged_subbox_list = []
        for parent_box_i, subbox_box_i in zip(
            parent_gt_boxes, subbox_gt_boxes
        ):
            merge_flag = False
            for _idx, merged_parent_box_i in enumerate(
                merged_parent_bbox_list
            ):
                if all(parent_box_i[:4] == merged_parent_box_i[:4]):
                    merge_flag = True
                    break

            if not merge_flag:
                merged_parent_bbox_list.append(parent_box_i)
                merged_subbox_list.append([subbox_box_i])
            else:
                merged_subbox_list[_idx].append(subbox_box_i)
                # merge hard sample
                merged_parent_bbox_list[_idx][4] = (
                    merged_parent_bbox_list[_idx][4] * parent_box_i[4]
                )  # noqa

        merged_parent_bbox_list = (
            np.array(merged_parent_bbox_list, dtype=np.float32)
            if len(merged_parent_bbox_list) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )  # noqa

        merged_subbox_list = self._pad_subbox(merged_subbox_list)

        return merged_parent_bbox_list, merged_subbox_list

    def _pad_subbox(self, subbox_list):
        pad_subbox_list = []
        for subbox_i in subbox_list:
            if len(subbox_i) <= self.max_subbox_num:
                temp_bbox = np.array(subbox_i, dtype=np.float32)
                pad_shape = list(temp_bbox.shape)
                pad_shape[0] = self.max_subbox_num
                boxes = _pad_array(temp_bbox, pad_shape, "wheel_box")
                pad_subbox_list.append(boxes)
            else:
                temp_bbox = np.array(subbox_i, dtype=np.float32)[
                    : self.max_subbox_num, :
                ]
                pad_subbox_list.append(temp_bbox)
        if len(pad_subbox_list) > 0:
            pad_subbox_list = np.array(pad_subbox_list, dtype=np.float32)
        else:
            pad_subbox_list = np.zeros(
                (0, self.max_subbox_num, 5), dtype=np.float32
            )

        return pad_subbox_list

    def _merge_ig_regions(self, parent_gt_boxes, child_ig_regions):
        def _pad_regions(input_regions):
            _regions = np.zeros((self.max_subbox_num, 5))
            if len(input_regions) > 0:
                input_regions = np.asarray(input_regions)
                num_ig = min(self.max_subbox_num, input_regions.shape[0])
                _regions[:num_ig] = input_regions[:num_ig]
            return _regions

        merged_ig_region_list = []
        for parent_box in parent_gt_boxes:
            matched_ig_regions = []
            for ig_region in child_ig_regions:
                contain_value = self._contain_value(parent_box, ig_region)
                if contain_value > self.ig_region_match_thresh:
                    matched_ig_regions.append(ig_region)

            matched_ig_regions = _pad_regions(matched_ig_regions)
            merged_ig_region_list.append(matched_ig_regions)

        merged_ig_region_list = np.asarray(merged_ig_region_list).reshape(
            -1, self.max_subbox_num, 5
        )
        return merged_ig_region_list

    @staticmethod
    def _contain_value(one, other):
        one = np.asarray(one[0:4])
        other = np.asarray(other[0:4])
        other_wh = np.maximum(other[2:4] - other[0:2], 0)
        other_area = np.maximum(other_wh[0] * other_wh[1], 0)
        union_wh = np.maximum(
            np.minimum(one[2:4], other[2:4])
            - np.maximum(one[0:2], other[0:2]),  # noqa
            0,
        )
        union_area = np.maximum(union_wh[0] * union_wh[1], 0)

        if other_area > 0:
            return union_area / other_area
        else:
            return 0


def _transform_multi_bboxes_with_parent(
    gt_boxes,
    ig_regions,
    parent_gt_boxes,
    parent_ig_regions,
    img_roi,
    affine_aug_param,
    clip=True,
    min_valid_area=8,
    min_valid_clip_area_ratio=0.5,
    min_edge_size=2,
    allow_outside_center=True,
    min_valid_sub_area=8,
    min_sub_edge_size=8,
):
    bbox_ts = LabelAffineTransform(label_type="box")

    def _transform(boxes):
        invalid = (boxes == 0).all(axis=-1)
        boxes = bbox_ts(
            boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
        )
        boxes[invalid, :] = 0

        return boxes

    # with shape (num_instance*max_num_sub,5+)
    gt_boxes_shape = gt_boxes.shape
    gt_boxes = gt_boxes.reshape(-1, gt_boxes_shape[-1])
    ts_gt_boxes = _transform(gt_boxes)
    # with shape (num_instance,5+)
    parent_gt_boxes_shape = parent_gt_boxes.shape
    parent_gt_boxes = parent_gt_boxes.reshape(-1, parent_gt_boxes_shape[-1])
    ts_parent_gt_boxes = _transform(parent_gt_boxes)

    if clip:
        clip_gt_boxes = clip_bbox(ts_gt_boxes, img_roi, True)
        clip_parent_gt_boxes = clip_bbox(ts_parent_gt_boxes, img_roi, True)
    else:
        clip_gt_boxes = ts_gt_boxes
        clip_parent_gt_boxes = ts_parent_gt_boxes

    # set low quality box to be hard
    clip_gt_boxes = _remap_bbox_label(
        clip_gt_boxes,
        min_valid_sub_area,
        min_sub_edge_size,
    )
    clip_gt_boxes = remap_bbox_label_by_clip_area_ratio(
        ts_gt_boxes,
        clip_gt_boxes,
        min_valid_clip_area_ratio,
        copy=True,
    )
    clip_parent_gt_boxes = _remap_bbox_label(
        clip_parent_gt_boxes,
        min_valid_area,
    )
    clip_parent_gt_boxes = remap_bbox_label_by_clip_area_ratio(
        ts_parent_gt_boxes,
        clip_parent_gt_boxes,
        min_valid_clip_area_ratio,
        copy=True,
    )

    # filter invalid bboxes
    # with shape (num_instance*max_num_sub)
    clip_gt_mask = filter_bbox_mask(
        clip_gt_boxes,
        img_roi,
        allow_outside_center=allow_outside_center,
        min_edge_size=min_sub_edge_size,
    )
    # with shape (num_instance)
    clip_parent_gt_mask = filter_bbox_mask(
        clip_parent_gt_boxes,
        img_roi,
        allow_outside_center=allow_outside_center,
        min_edge_size=min_edge_size,
    )
    # merge child mask and parent mask
    # with shape (num_instance,max_num_sub)
    clip_gt_mask = clip_gt_mask.reshape(gt_boxes_shape[:-1])
    # with shape (num_instance)
    gt_mask = np.logical_and(
        clip_gt_mask.any(axis=-1),
        clip_parent_gt_mask,
    )
    # with shape (num_instance,max_num_sub,5+)
    clip_gt_boxes = clip_gt_boxes.reshape(gt_boxes_shape)[gt_mask]
    clip_gt_mask = clip_gt_mask[gt_mask]
    clip_gt_boxes = np.where(
        clip_gt_mask[..., None].repeat(gt_boxes_shape[-1], axis=-1),
        clip_gt_boxes,
        np.zeros_like(clip_gt_boxes),
    )
    # with shape (num_instance,5+)
    clip_parent_gt_boxes = clip_parent_gt_boxes.reshape(parent_gt_boxes_shape)[
        gt_mask
    ]  # noqa

    clip_ig_regions = None
    if ig_regions is not None:
        # with shape (num_instance*max_num_sub,5+)
        ig_regions_shape = ig_regions.shape
        ig_regions = ig_regions.reshape(-1, ig_regions_shape[-1])
        ts_ig_regions = _transform(ig_regions)
        if clip:
            clip_ig_regions = clip_bbox(ts_ig_regions, img_roi)
        else:
            clip_ig_regions = ts_ig_regions
        # with shape (num_instance&max_num_sub)
        clip_ig_mask = filter_bbox_mask(
            clip_ig_regions,
            img_roi,
            allow_outside_center=allow_outside_center,
            min_edge_size=min_sub_edge_size,
        )
        # with shape (num_instance,max_num_sub)
        clip_ig_mask = clip_ig_mask.reshape(ig_regions_shape[:-1])
        # with shape (num_instance,max_num_sub,5+)
        clip_ig_regions = clip_ig_regions.reshape(ig_regions_shape)[gt_mask]
        clip_ig_mask = clip_ig_mask[gt_mask]
        clip_ig_regions = np.where(
            clip_ig_mask[..., None].repeat(ig_regions_shape[-1], axis=-1),
            clip_ig_regions,
            np.zeros_like(clip_ig_regions),
        )

    clip_parent_ig_regions = None
    if parent_ig_regions is not None:
        # with shape (num_instance*max_num_sub,5+)
        parent_ig_regions_shape = parent_ig_regions.shape
        parent_ig_regions = parent_ig_regions.reshape(
            -1, parent_ig_regions_shape[-1]
        )  # noqa
        ts_parent_ig_regions = _transform(parent_ig_regions)
        if clip:
            clip_parent_ig_regions = clip_bbox(ts_parent_ig_regions, img_roi)
        else:
            clip_parent_ig_regions = ts_parent_ig_regions

    return (
        clip_gt_boxes,
        clip_ig_regions,
        clip_parent_gt_boxes,
        clip_parent_ig_regions,
    )


def _remap_bbox_label(
    bboxes,
    min_area=8,
    min_edge_size=None,
):
    """Map bbox label to be hard if its area < min_area."""
    bboxes = deepcopy(bboxes)
    width = np.maximum(bboxes[:, 2] - bboxes[:, 0], 0)
    height = np.maximum(bboxes[:, 3] - bboxes[:, 1], 0)
    area = width * height

    to_be_hard_flag = np.logical_and(area < min_area, bboxes[:, 4] > 0)
    if min_edge_size is not None:
        edge_mask = np.logical_and(
            width > min_edge_size, height > min_edge_size
        )
        to_be_hard_flag = np.logical_and(edge_mask, to_be_hard_flag)
    bboxes[to_be_hard_flag, 4] *= -1

    return bboxes


@OBJECT_REGISTRY.register
class PadRoIDetData(object):
    def __init__(self, max_gt_boxes_num=100, max_ig_regions_num=100):
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num
        self.FloatCast = Cast(np.float32)

    def __call__(self, data):
        img, gt_boxes, ig_regions = (
            data["img"],
            data["gt_boxes"],
            data.get("ig_regions", None),
        )

        def _pad_boxes(boxes, num, name):
            pad_shape = list(boxes.shape)
            pad_shape[0] = num
            boxes_num = np.array(boxes.shape[0]).reshape((1,))
            boxes = _pad_array(boxes, pad_shape, name)
            return boxes, boxes_num

        gt_boxes, gt_boxes_num = _pad_boxes(
            gt_boxes, self.max_gt_boxes_num, "gt_boxes"
        )
        # Cast gt_boxes
        im_hw = self.FloatCast(data["im_hw"])
        gt_boxes = self.FloatCast(gt_boxes)
        gt_boxes_num = self.FloatCast(gt_boxes_num)

        if ig_regions is None:
            return dict(
                img=img,
                im_hw=im_hw,
                gt_boxes=gt_boxes,
                gt_boxes_num=gt_boxes_num,
            )

        ig_regions, ig_regions_num = _pad_boxes(
            ig_regions, self.max_ig_regions_num, "ig_regions"
        )
        # Cast ig_regions
        ig_regions = self.FloatCast(ig_regions)
        ig_regions_num = self.FloatCast(ig_regions_num)

        parent_gt_boxes, parent_ig_regions = data.get(
            "parent_gt_boxes", None
        ), data.get("parent_ig_regions", None)
        if parent_gt_boxes is not None and parent_ig_regions is not None:
            parent_gt_boxes, parent_gt_boxes_num = _pad_boxes(
                parent_gt_boxes, self.max_gt_boxes_num, "parent_gt_boxes"
            )
            parent_ig_regions, parent_ig_regions_num = _pad_boxes(
                parent_ig_regions, self.max_ig_regions_num, "parent_ig_regions"
            )
            # Cast ig_regions
            parent_gt_boxes = self.FloatCast(parent_gt_boxes)
            parent_gt_boxes_num = self.FloatCast(parent_gt_boxes_num)
            parent_ig_regions = self.FloatCast(parent_ig_regions)
            parent_ig_regions_num = self.FloatCast(parent_ig_regions_num)
            return dict(
                img=img,
                im_hw=im_hw,
                gt_boxes=gt_boxes,
                gt_boxes_num=gt_boxes_num,
                ig_regions=ig_regions,
                ig_regions_num=ig_regions_num,
                parent_gt_boxes=parent_gt_boxes,
                parent_gt_boxes_num=parent_gt_boxes_num,
                parent_ig_regions=parent_ig_regions,
                parent_ig_regions_num=parent_ig_regions_num,
            )
        return dict(
            img=img,
            im_hw=im_hw,
            gt_boxes=gt_boxes,
            gt_boxes_num=gt_boxes_num,
            ig_regions=ig_regions,
            ig_regions_num=ig_regions_num,
        )


@OBJECT_REGISTRY.register
class PadVehicleWheelData(object):
    def __init__(self, max_gt_boxes_num=100, max_ig_regions_num=100):
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num
        self.FloatCast = Cast(np.float32)

    def __call__(self, data):
        img = data["img"]
        im_hw = data["im_hw"]
        gt_boxes = data["gt_boxes"]
        parent_gt_boxes = data["parent_gt_boxes"]
        ig_regions = data.get("ig_regions", None)
        parent_ig_regions = data.get("parent_ig_regions", None)

        def _pad_boxes(boxes, num, name):
            pad_shape = list(boxes.shape)
            pad_shape[0] = num
            boxes_num = np.array(boxes.shape[0]).reshape((1,))
            boxes = _pad_array(boxes, pad_shape, name)
            return boxes, boxes_num

        # gt_boxes
        gt_boxes, gt_boxes_num = _pad_boxes(
            gt_boxes, self.max_gt_boxes_num, "gt_boxes"
        )
        im_hw = self.FloatCast(im_hw)
        gt_boxes = self.FloatCast(gt_boxes)
        gt_boxes_num = self.FloatCast(gt_boxes_num)

        # parent_gt_boxes
        parent_gt_boxes, parent_gt_boxes_num = _pad_boxes(
            parent_gt_boxes, self.max_gt_boxes_num, "parent_gt_boxes"
        )
        parent_gt_boxes = self.FloatCast(parent_gt_boxes)
        parent_gt_boxes_num = self.FloatCast(parent_gt_boxes_num)

        ret_dict = {
            "img": img,
            "im_hw": im_hw,
            "gt_boxes": gt_boxes,
            "gt_boxes_num": gt_boxes_num,
            "parent_gt_boxes": parent_gt_boxes,
            "parent_gt_boxes_num": parent_gt_boxes_num,
        }

        # ig_regions
        if ig_regions is not None:
            ig_regions, ig_regions_num = _pad_boxes(
                ig_regions, self.max_gt_boxes_num, "ig_regions"
            )
            ret_dict["ig_regions"] = self.FloatCast(ig_regions)
            ret_dict["ig_regions_num"] = self.FloatCast(ig_regions_num)

        if parent_ig_regions is not None:
            parent_ig_regions, parent_ig_regions_num = _pad_boxes(
                parent_ig_regions, self.max_ig_regions_num, "parent_ig_regions"
            )
            # Cast ig_regions
            ret_dict["parent_ig_regions"] = self.FloatCast(parent_ig_regions)
            ret_dict["parent_ig_regions_num"] = self.FloatCast(
                parent_ig_regions_num
            )

        return ret_dict


@OBJECT_REGISTRY.register
class RoIDetInputPadding(DetInputPadding):
    def __call__(self, data):
        data = super().__call__(data)

        data["parent_gt_boxes"][..., :4:2] += self.input_padding[0]
        data["parent_gt_boxes"][..., 1:4:2] += self.input_padding[2]

        if "parent_ig_regions" in data:
            data["parent_ig_regions"][..., :4:2] += self.input_padding[0]
            data["parent_ig_regions"][..., 1:4:2] += self.input_padding[2]

        return data
