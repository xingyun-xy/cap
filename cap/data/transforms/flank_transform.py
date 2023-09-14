import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cap.registry import OBJECT_REGISTRY
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
from .common import Cast
from .detection import DetInputPadding


def pad_flank(
    img,
    bboxes,
    flanks,
):

    im_hw = np.array(img.shape[:2]).reshape((2,))
    # pad ignore regions

    ig_regions = np.zeros([0, 5], dtype=np.float32)
    cast = Cast(np.float32)

    return {
        "img": img,
        "im_hw": cast(im_hw),
        "gt_boxes": cast(bboxes),
        "gt_flanks": cast(flanks),
        "ig_regions": cast(ig_regions),
    }


@OBJECT_REGISTRY.register
class VehicleFlankRoiTransform:
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
        rand_sampling_bbox=False,
        resize_wh=None,
        keep_aspect_ratio=False,
        min_flank_width=4,
        min_flank_width_overlap=0.1,
    ):
        super(VehicleFlankRoiTransform, self).__init__()
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
        self.min_valid_area = min_valid_area
        self.min_valid_clip_area_ratio = min_valid_clip_area_ratio
        self.min_flank_width = min_flank_width
        self.min_flank_width_overlap = min_flank_width_overlap
        self._resize_wh = resize_wh
        self._keep_aspect_ratio = keep_aspect_ratio
        self._bbox_ts = LabelAffineTransform(label_type="box")
        self._kps_ts = Point2DAffineTransform()

    def __call__(self, data):
        img = data["img"]
        ret = data["anno"]
        bbox_instances = ret["bboxes"]  # {'data':, 'class_id':}
        flank_instances = ret["flanks"]  # {'data':, 'class_id':}
        assert len(bbox_instances) == len(flank_instances)
        bboxes, bbox_clses = [], []
        for instance in bbox_instances:
            bboxes.append(instance["data"])  # [4]
            bbox_clses.append(instance["class_id"])
        bboxes = np.array(bboxes, np.float32).reshape([-1, 4])  # [n,4]
        bbox_clses = np.array(bbox_clses, np.int).flatten()  # [n]
        flanks, flank_clses = [], []
        for instance in flank_instances:
            flanks.append(instance["data"])  # [4,2]
            flank_clses.append(instance["class_id"])
        flanks = np.array(flanks, np.float32).reshape([-1, 4, 2])  # [n,4,2]
        flank_clses = np.array(flank_clses, np.int).flatten()  # [n]

        rois = deepcopy(bboxes)  # [n,4]
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
            rois = self._bbox_ts(rois, affine_mat, flip=False)

        for affine_aug_param in self._roi_ts(rois, img_wh):
            cur_affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=cur_affine_mat, flipped=affine_aug_param.flipped
            )

            ts_img = self._img_ts(img, affine_aug_param.mat)
            ts_img_wh = ts_img.shape[:2][::-1]

            (
                ts_bboxes,
                ts_bbox_clses,
                ts_flanks,
                ts_flank_clses,
            ) = transform_roi_and_flank(
                self._bbox_ts,
                self._kps_ts,
                affine_aug_param,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                bboxes,
                bbox_clses,
                flanks,
                flank_clses,
                self.min_valid_area,
                self.min_valid_clip_area_ratio,
                self.min_flank_width,
                self.min_flank_width_overlap,
            )

            ts_bboxes = ts_bboxes.reshape([-1, 4])  # [n,4]
            ts_bbox_clses = ts_bbox_clses.reshape([-1, 1])  # [n,1]
            ts_flanks = ts_flanks.reshape([-1, 8])  # [n,8]
            ts_flank_clses = ts_flank_clses.reshape([-1, 1])  # [n,1]

            # [n,5] <x1,y1,x2,y2,cls>
            ret_ts_bboxes = np.concatenate([ts_bboxes, ts_bbox_clses], 1)
            # [n,9] <lb_pt,rb_pt,lt_pt,rt_pt,cls>
            ret_ts_flanks = np.concatenate([ts_flanks, ts_flank_clses], 1)

            data = pad_flank(
                ts_img,
                ret_ts_bboxes,
                ret_ts_flanks,
            )
            data["img"] = data["img"].transpose(2, 0, 1)
            return data

    @staticmethod
    def vis_origin(img, record, show=False, vis_dir=None):
        bbox_lst = record["bboxes"]
        flank_lst = record["flanks"]
        ignore_lst = record["ignore_regions"]

        plt.figure(figsize=(20.48, 12.80), dpi=100)
        fig = plt.imshow(img.astype(np.float32) / 255.0)

        # draw positive and negative
        for bbox, flank in zip(bbox_lst, flank_lst):
            x1, y1, x2, y2 = bbox["data"]
            flank_points = flank["data"]  # noqa
            flank_cls = flank["class_id"]
            if flank_cls == 0:
                # negative
                _draw_bbox(fig, [x1, y1, x2, y2], "N", "blue")
            else:
                # positive
                _draw_bbox(fig, [x1, y1, x2, y2], "P", "red")
                pt0, pt1, pt2, pt3 = flank_points
                _draw_square([pt0, pt1, pt3, pt2], "yellow")

        # draw ignore
        for ignore in ignore_lst:
            x1, y1 = ignore["left_top"]
            x2, y2 = ignore["right_bottom"]
            _draw_bbox(fig, [x1, y1, x2, y2], "I", "skyblue")

        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(
                vis_dir, "render_" + os.path.basename("image_url")
            )  # noqa
            plt.savefig(vis_path)
        if show:
            plt.show()

        plt.close()

    @staticmethod
    def vis_label(img, bboxes, flanks, show=True):
        plt.figure(figsize=(20.48, 12.80), dpi=100)
        fig = plt.imshow(img.astype(np.float32) / 255.0)
        for bbox, flank in zip(bboxes, flanks):
            if flank[-1] == 0:
                _draw_bbox(fig, bbox[0:4], color="blue")
            else:
                _draw_bbox(fig, bbox[0:4], color="red")
                pt0, pt1, pt2, pt3 = flank[:8].reshape(-1, 2)
                _draw_square([pt0, pt1, pt3, pt2], "yellow")

        if show:
            plt.show()

        plt.close()


def transform_roi_and_flank(
    bbox_ts,
    kps_ts,
    affine_aug_param,
    img_roi,
    bboxes,
    bbox_clses,
    flanks,
    flank_clses,
    min_valid_area=8,
    min_valid_clip_area_ratio=0.5,
    min_flank_width=4,
    min_flank_width_overlap=0.1,
):
    """
    Transform roi and floak.

    Parameters
    ----------
    bbox_ts : LabelAffineTransform
    kps_ts : Point2DAffineTransform
    affine_aug_param : AffineAugMat
    img_roi : tuple
    bboxes : np.ndarray
    bbox_clses : np.ndarray
    flanks : np.ndarray
    flank_clses : np.ndarray
    min_valid_area : float
    min_valid_clip_area_ratio : float
    min_flank_width : float
    min_flank_width_overlap : float

    Returns
    -------
    ts_bboxes : np.ndarray
        with shape [m,4]
    ts_bbox_clses : np.ndarray
        with shape [m]
    ts_flanks : np.ndarray
        with shape [m,4,2]
    ts_flank_clses : np.ndarray
        with shape [m]

    """
    # transform bboxes and flanks
    ts_bboxes = bbox_ts(
        bboxes, affine_aug_param.mat, flip=affine_aug_param.flipped
    )  # noqa
    ts_flanks = np.array(
        [[kps_ts(pt, affine_aug_param.mat)] for pt in flanks], dtype=np.float32
    ).reshape([-1, 4, 2])
    if affine_aug_param.flipped:
        lb, rb, lt, rt = np.split(ts_flanks, 4, 1)  # [B,1,2]
        ts_flanks = np.concatenate([rb, lb, rt, lt], 1)  # [B,4,2]
    # clip bboxes and flanks
    cp_bboxes, cp_bbox_clses, cp_flanks, cp_flank_clses = clip_bbox_and_flank(
        ts_bboxes,
        bbox_clses,
        ts_flanks,
        flank_clses,
        img_roi,
        min_valid_area,
        min_valid_clip_area_ratio,
        min_flank_width,
        min_flank_width_overlap,
    )
    # remap bboxes and flanks classes
    ret_bboxes, ret_bbox_clses, ret_flanks, ret_flank_clses = remap_classes(
        cp_bboxes, cp_bbox_clses, cp_flanks, cp_flank_clses
    )

    return ret_bboxes, ret_bbox_clses, ret_flanks, ret_flank_clses


def remap_classes(bboxes, bbox_clses, flanks, flank_clses):
    """
    Map the instance to hard and normal instance; now just return all.

    TODO @feng02.li

    Parameters
    ----------
    bboxes
    bbox_clses
    flanks
    flank_clses

    """
    return bboxes, bbox_clses, flanks, flank_clses


def clip_bbox_and_flank(
    bboxes,
    bbox_clses,
    flanks,
    flank_clses,
    img_roi,
    min_valid_area,
    min_valid_clip_area_ratio,
    min_flank_width,
    min_flank_width_overlap,
):
    """
    Clip bounding boxes and flanks.

    Parameters
    ----------
    bboxes : np.ndarray
        with shape [n,4]
    bbox_clses : np.ndarray
        with shape [n]
    flanks : np.ndarray
        with shape [n,4,2]
    flank_clses : np.ndarray
         with shape [n]
    img_roi : tuple
    min_valid_area : float
    min_valid_clip_area_ratio : float
    min_flank_width : float
    min_flank_width_overlap : float


    """
    new_bboxes, new_flanks = [], []
    new_bbox_clses, new_flank_clses = [], []
    for bbox, flank, bbox_cls, flank_cls in zip(
        bboxes, flanks, bbox_clses, flank_clses
    ):  # noqa
        # filter with bbox
        clipped_bbox = _clip_bbox(bbox, img_roi)
        old_bbox_area = _get_bbox_area(bbox)
        new_bbox_area = _get_bbox_area(clipped_bbox)
        bbox_clip_ratio = new_bbox_area / np.maximum(old_bbox_area, 1e-14)
        if (
            new_bbox_area < min_valid_area
            or bbox_clip_ratio < min_valid_clip_area_ratio
        ):  # noqa
            continue

        # filter with flank
        if flank_cls == 0:
            clipped_flank = np.zeros_like(flank, np.float32)
        else:
            clipped_flank = _clip_flank(flank, img_roi)
            flank_width = _get_flank_width(clipped_flank)
            flank_width_overlap = flank_width / (
                clipped_bbox[2] - clipped_bbox[0]
            )  # noqa
            if (
                flank_width < min_flank_width
                or flank_width_overlap < min_flank_width_overlap
            ):  # noqa
                continue

        new_bboxes.append(clipped_bbox)
        new_bbox_clses.append(bbox_cls)
        new_flanks.append(clipped_flank)
        new_flank_clses.append(flank_cls)

    new_bboxes = np.array(new_bboxes, np.float32).reshape([-1, 4])
    new_bbox_clses = np.array(new_bbox_clses, np.float32).flatten()
    new_flanks = np.array(new_flanks, np.float32).reshape([-1, 4, 2])
    new_flank_clses = np.array(new_flank_clses, np.float32).flatten()

    return new_bboxes, new_bbox_clses, new_flanks, new_flank_clses


def _clip_bbox(bbox, img_roi):
    """
    Clip bounding boxes.

    Parameters
    ----------
    bbox : np.ndarray
        with shape [4]
    img_roi : tuple

    """
    img_x1, img_y1, img_x2, img_y2 = img_roi
    min_val = np.array([img_x1, img_y1, img_x1, img_y1], np.float32)
    max_val = np.array([img_x2, img_y2, img_x2, img_y2], np.float32)
    bbox = np.clip(bbox, min_val, max_val)

    return bbox


def _clip_flank(flank, img_roi):
    """
    Clip flanks.

    Parameters
    ----------
    flank : np.ndarray
        with shape [4,2]
    img_roi : tuple


    """
    img_x1, img_y1, img_x2, img_y2 = img_roi
    min_val = np.array([img_x1, img_y1], np.float32)
    max_val = np.array([img_x2, img_y2], np.float32)
    left_bottom, right_bottom, left_top, right_top = flank
    left_top = np.clip(left_top, min_val, max_val)
    right_top = np.clip(right_top, min_val, max_val)
    left_bottom_y, right_bottom_y = _get_intersections_to_vertical(
        left_bottom, right_bottom, [left_top[0], right_top[0]]
    )
    left_bottom = np.array([left_top[0], left_bottom_y], np.float32)
    right_bottom = np.array([right_top[0], right_bottom_y], np.float32)

    flank = np.asarray(
        [left_bottom, right_bottom, left_top, right_top], np.float32
    )  # noqa

    return flank


def _get_intersections_to_vertical(one_point, other_point, loc_x_lst):
    """
    Get intersections.

    Parameters
    ----------
    one_point : np.ndarray
        with shape [2]
    other_point : np.ndarray
        with shape [2]
    loc_x_lst : list


    """
    one_x, one_y = one_point
    other_x, other_y = other_point
    delta_x = one_x - other_x
    delta_y = one_y - other_y
    if delta_x == 0:
        raise ValueError("Slope is not Valid")
    slope = delta_y / delta_x
    return [slope * (x - other_x) + other_y for x in loc_x_lst]


def _get_bbox_area(bbox):
    width = np.maximum(bbox[2] - bbox[0], 0)
    height = np.maximum(bbox[3] - bbox[1], 0)

    return width * height


def _get_flank_width(flank):
    flank_width = flank[1][0] - flank[0][0]
    return flank_width


def _draw_line(start, end, color="red"):
    plt.plot((start[0], end[0]), (start[1], end[1]), color=color)


def _draw_bbox(figure, bbox, title="", color="red"):
    rect = plt.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False,
        edgecolor=color,
        linewidth=1,
    )
    figure.axes.add_patch(rect)
    if title:
        figure.axes.text(
            rect.xy[0] + 25,
            rect.xy[1] + 10,
            title,
            va="center",
            ha="center",
            fontsize=10,
            color="blue",
            bbox={"facecolor": "m", "lw": 0},
        )

    return figure


def _draw_square(points, color="red"):
    for i in range(len(points) - 1):
        _draw_line(points[i], points[i + 1], color)
    _draw_line(points[len(points) - 1], points[0], color)


@OBJECT_REGISTRY.register
class PadFlankData(object):
    def __init__(self, max_gt_boxes_num=100, max_ig_regions_num=100):  # noqa
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num

    def __call__(self, data):

        # pad bboxes
        pad_shape = list(data["gt_boxes"].shape)
        pad_shape[0] = self.max_gt_boxes_num
        data["gt_boxes_num"] = (
            np.array(data["gt_boxes"].shape[0])
            .reshape((1,))
            .astype(data["gt_boxes"].dtype)
        )
        data["gt_boxes"] = _pad_array(data["gt_boxes"], pad_shape, "gt_boxes")

        # pad flanks
        pad_shape = list(data["gt_flanks"].shape)
        pad_shape[0] = self.max_gt_boxes_num
        data["gt_flanks_num"] = np.array(data["gt_flanks"].shape[0]).reshape(
            (1,)
        )
        data["gt_flanks"] = _pad_array(
            data["gt_flanks"], pad_shape, "gt_flanks"
        )

        if "ig_regions" not in data:
            data["ig_regions"] = np.zeros([0, 5], dtype=np.float32)
        pad_shape = list(data["ig_regions"].shape)
        pad_shape[0] = self.max_ig_regions_num
        data["ig_regions_num"] = np.array(data["ig_regions"].shape[0]).reshape(
            (1,)
        )
        data["ig_regions"] = _pad_array(
            data["ig_regions"], pad_shape, "ig_regions"
        )

        return data


@OBJECT_REGISTRY.register
class FlankDetInputPadding(DetInputPadding):
    def __call__(self, data):
        data = super().__call__(data)

        data["gt_flanks"][..., :7:2] += self.input_padding[0]
        data["gt_flanks"][..., 1:8:2] += self.input_padding[2]

        return data
