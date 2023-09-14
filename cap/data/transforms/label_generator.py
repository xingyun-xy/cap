import math
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as coco_mask
from scipy.interpolate import interp2d

from cap.core.affine import point_affine_transform
from cap.core.heatmap import draw_heatmap
from cap.core.undistort_lut import getUnDistortPoints
from .utils_3d import (
    affine_transform,
    convert_alpha,
    get_affine_transform,
    get_dense_locoffset,
    get_gaussian2D,
    get_reg_map,
    xywh_to_x1y1x2y2,
)

__all__ = [
    "roi_heatmap_label_encoding",
    "roi_heatmap_label_encoding_undistort_uv_depth",
    "label_encoding",
    "dense3d_pad_after_label_generator",
]

GlobalEQFUMap = defaultdict()
GlobalEQFVMap = defaultdict()


def roi_heatmap_label_encoding(
    img,
    label,
    meta,
    num_classes,
    classid_map,
    normalize_depth,  # noqa
    focal_length_default,
    max_gt_boxes_num,
    filtered_name,  # noqa
    use_bbox2d,
    shift,
    max_depth,
    use_project_bbox2d,
):  # noqa

    calib = meta["calib"]
    # get trans_mat
    trans_mat = meta["trans_matrix"]

    depth = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    proj_2d_bboxes = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)
    ig_regions = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)

    dimensions = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    locations = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    rotation_y = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    location_offsets = np.zeros([max_gt_boxes_num, 2], dtype=np.float32)

    if filtered_name not in meta["file_name"]:
        gt_boxes_num = 0
        for ann in label:
            in_camera = ann["in_camera"] if "in_camera" in ann.keys() else ann
            # set id
            cls_id = int(classid_map[ann["category_id"]])
            # only vehicle
            if cls_id < 0:
                continue
            if in_camera["depth"] > max_depth:
                continue
            # set box3D
            locs = np.array(in_camera["location"])
            rot_y = np.array(in_camera["rotation_y"])

            if use_bbox2d and "bbox_2d" in ann and ann["bbox_2d"] is not None:
                bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                a_bbox = ann["bbox_2d"].copy()
                bbox_cx = a_bbox[0] + a_bbox[2] / 2.0
                bbox_cy = a_bbox[1] + a_bbox[3] / 2.0
                proj_p = cv2.undistortPoints(
                    np.array([bbox_cx, bbox_cy]),
                    np.array(calib[:3, :3]),
                    np.array(meta["distCoeffs"]),
                    None,
                    np.array(calib[:3, :3]),
                )  # noqa
                proj_p = proj_p.reshape(-1, 2)
                proj_p = proj_p.astype(np.int)
                bbox_cx = proj_p[0, 0]
                bbox_cy = proj_p[0, 1]
                cx, cy = calib[0, 2], calib[1, 2]
                location_offset = np.array(
                    [
                        elem * in_camera["depth"] / calib[0, 0]
                        for elem in [bbox_cx - cx, bbox_cy - cy, calib[0, 0]]
                    ]  # noqa
                ).astype(np.float64)
                location_offset[1] += in_camera["dim"][0] / 2.0
                location_offset = in_camera["location"] - location_offset
                location_offset = location_offset.tolist()
            elif use_project_bbox2d:
                bbox = xywh_to_x1y1x2y2(ann["bbox"])
                location_offset = in_camera["location_offset"]
            else:
                continue
            # origin_ctx = (bbox[0] + bbox[2]) / 2
            # origin_cty = (bbox[1] + bbox[3]) / 2
            bbox[:2] = point_affine_transform(bbox[:2], trans_mat)
            bbox[2:] = point_affine_transform(bbox[2:], trans_mat)

            _wh = bbox[2:] - bbox[:2]
            # bbox = np.hstack([bbox, [cls_id]])
            if np.any(_wh <= 0):
                continue
            proj_2d_bboxes[gt_boxes_num] = bbox

            depth[gt_boxes_num] = (
                np.array(in_camera["depth"])
                * focal_length_default
                / calib[0, 0]
            )
            location_offsets[gt_boxes_num] = (
                np.array(location_offset)[:2]
                * focal_length_default
                / calib[0, 0]
            )

            dimensions[gt_boxes_num] = np.array(in_camera["dim"])
            locations[gt_boxes_num] = locs
            rotation_y[gt_boxes_num] = rot_y
            # alpha_x[gt_boxes_num] = alpha
            gt_boxes_num += 1
            if gt_boxes_num >= max_gt_boxes_num:
                break

    gt_boxes = np.hstack(
        (
            proj_2d_bboxes,  # 4
            location_offsets,  # 2
            depth,  # 1
            dimensions,  # 3
            locations,  # 3
            rotation_y,  # 1
        )
    )
    ret = {
        "trans_mat": np.array(trans_mat, np.float32),
        "calib": np.array(calib, np.float32),
        "gt_boxes": gt_boxes.astype(np.float32),
        "gt_boxes_num": np.array(gt_boxes_num, np.float32),
        "ig_regions_num": np.array(0.0, np.float32),
        "ig_regions": ig_regions.astype(np.float32),
    }
    del label

    return ret


def roi_heatmap_label_encoding_undistort_uv_depth(
    img,
    label,
    meta,
    num_classes,
    classid_map,
    normalize_depth,  # noqa
    focal_length_default,
    max_gt_boxes_num,
    filtered_name,  # noqa
    use_bbox2d,
    shift,
    max_depth,
    use_project_bbox2d,
):  # noqa

    calib = meta["calib"]
    # get trans_mat
    trans_mat = meta["trans_matrix"]

    depth_u = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    depth_v = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    proj_2d_bboxes = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)
    ig_regions = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)

    dimensions = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    locations = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    rotation_y = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    location_offsets = np.zeros([max_gt_boxes_num, 2], dtype=np.float32)

    # undistort uv
    center, size = meta["center"], meta["size"]
    width, height = meta["img_wh"]
    trans_output = get_affine_transform(
        center, size, 0, [width, height], shift=shift
    )

    eq_fu_mat, eq_fv_mat = cal_equivalent_focal_length_uv_mat(
        size[0], size[1], calib, meta["distCoeffs"]
    )  # noqa
    resized_eq_fu = cv2.warpAffine(eq_fu_mat, trans_output, (width, height))
    resized_eq_fv = cv2.warpAffine(eq_fv_mat, trans_output, (width, height))
    meta["eq_fu"] = resized_eq_fu
    meta["eq_fv"] = resized_eq_fv

    if "image_key" in meta.keys():
        meta["file_name"] = meta["image_key"]
    if filtered_name not in meta["file_name"]:
        gt_boxes_num = 0
        for _, ann in enumerate(label):
            in_camera = ann["in_camera"] if "in_camera" in ann.keys() else ann
            # set id
            cls_id = int(classid_map[ann["category_id"]])
            # only vehicle
            if cls_id < 0:
                continue
            if in_camera["depth"] > max_depth:
                continue
            # set box3D
            locs = np.array(in_camera["location"])
            rot_y = np.array(in_camera["rotation_y"])

            if use_bbox2d and "bbox_2d" in ann and ann["bbox_2d"] is not None:
                bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                a_bbox = ann["bbox_2d"].copy()
                bbox_cx = a_bbox[0] + a_bbox[2] / 2.0
                bbox_cy = a_bbox[1] + a_bbox[3] / 2.0
                proj_p = getUnDistortPoints(
                    np.array([[bbox_cx, bbox_cy]]),
                    np.array(calib[:3, :3]),  # noqa
                    np.array(meta["distCoeffs"]),
                    img_wh=meta["orgin_wh"],
                )  # noqa
                proj_p = proj_p.reshape(-1, 2)
                proj_p = proj_p.astype(np.int)
                bbox_cx = proj_p[0, 0]
                bbox_cy = proj_p[0, 1]
                cx, cy = calib[0, 2], calib[1, 2]
                location_offset = np.array(
                    [
                        elem * in_camera["depth"] / calib[0, 0]
                        for elem in [bbox_cx - cx, bbox_cy - cy, calib[0, 0]]
                    ]  # noqa
                ).astype(np.float64)
                location_offset[1] += in_camera["dim"][0] / 2.0
                location_offset = in_camera["location"] - location_offset
                location_offset = location_offset.tolist()
            elif use_project_bbox2d:
                bbox = xywh_to_x1y1x2y2(ann["bbox"])
                location_offset = in_camera["location_offset"]
            else:
                continue
            origin_ctx = (bbox[0] + bbox[2]) / 2
            origin_cty = (bbox[1] + bbox[3]) / 2
            bbox[:2] = point_affine_transform(bbox[:2], trans_mat)
            bbox[2:] = point_affine_transform(bbox[2:], trans_mat)

            _wh = bbox[2:] - bbox[:2]
            # bbox = np.hstack([bbox, [cls_id]])
            if np.any(_wh <= 0):
                continue
            proj_2d_bboxes[gt_boxes_num] = bbox

            eq_fu = eq_fu_mat[int(origin_cty), int(origin_ctx)]
            eq_fv = eq_fv_mat[int(origin_cty), int(origin_ctx)]
            depth_u[gt_boxes_num] = (
                np.array(in_camera["depth"]) * focal_length_default / eq_fu
            )  # noqa
            depth_v[gt_boxes_num] = (
                np.array(in_camera["depth"]) * focal_length_default / eq_fv
            )  # noqa

            location_offsets[gt_boxes_num] = (
                np.array(location_offset)[:2]
                * focal_length_default
                / calib[0, 0]
            )

            dimensions[gt_boxes_num] = np.array(in_camera["dim"])
            locations[gt_boxes_num] = locs
            rotation_y[gt_boxes_num] = rot_y
            # alpha_x[gt_boxes_num] = alpha
            gt_boxes_num += 1
            if gt_boxes_num >= max_gt_boxes_num:
                break

    gt_boxes = np.hstack(
        (
            proj_2d_bboxes,  # 4
            location_offsets,  # 2
            depth_u,  # 1
            depth_v,  # 1
            dimensions,  # 3
            locations,  # 3
            rotation_y,  # 1
        )
    )
    ret = {
        "trans_mat": np.array(trans_mat, np.float32),
        "calib": np.array(calib, np.float32),
        "gt_boxes": gt_boxes.astype(np.float32),
        "gt_boxes_num": np.array(gt_boxes_num, np.float32),
        "ig_regions_num": np.array(0.0, np.float32),
        "ig_regions": ig_regions.astype(np.float32),
    }
    del label

    return ret


def _convert_alpha(alpha, alpha_in_degree):
    return math.radians(alpha + 45) if alpha_in_degree else alpha


def label_encoding(
    label,
    meta,
    num_classes,
    classid_map,
    normalize_depth,
    focal_length_default,
    alpha_in_degree,
    down_stride,
    use_bbox2d,
    enable_ignore_area,
    shift,
    filtered_name,
    min_box_edge,
    max_depth,
    max_objs,
    use_project_bbox2d,
    undistort_2dcenter,
    undistort_depth_uv,
):

    calib = meta["calib"]
    center, size = meta["center"], meta["size"]
    width, height = meta["img_wh"]
    output_width, output_height = [width // down_stride, height // down_stride]
    # get affine transform from original img size(1600,900) to feature map(down_stride=4, after neck, (256,160)) note by zmj
    trans_output = get_affine_transform(
        center, size, 0, [output_width, output_height], shift=shift
    )

    if enable_ignore_area:
        if "ignore_mask" not in meta:
            ignore_mask = np.zeros((output_height, output_width, 1))
        else:
            ignore_mask = coco_mask.decode(meta["ignore_mask"]).astype(
                np.uint8
            )
            ignore_mask = cv2.warpAffine(
                ignore_mask,
                trans_output,
                (output_width, output_height),
                flags=cv2.INTER_NEAREST,
            )
            ignore_mask = ignore_mask.astype(np.float32)[:, :, np.newaxis]

    hm = np.zeros((output_height, output_width, num_classes), dtype=np.float32)
    wh = np.zeros((output_height, output_width, 2), dtype=np.float32)
    depth = np.zeros((output_height, output_width), dtype=np.float32)
    dim = np.zeros((output_height, output_width, 3), dtype=np.float32)
    loc_offset = np.zeros((output_height, output_width, 2), dtype=np.float32)
    weight_hm = np.zeros((output_height, output_width), dtype=np.float32)
    point_pos_mask = np.zeros((output_height, output_width), dtype=np.float32)
    # sin cos
    alpha_x = np.zeros((max_objs, 1), dtype=np.float32)
    ind_ = np.zeros((max_objs), dtype=np.int64)
    ind_mask_ = np.zeros((max_objs), dtype=np.float32)
    rot_y_ = np.zeros((max_objs, 1), dtype=np.float32)
    loc_ = np.zeros((max_objs, 3), dtype=np.float32)
    dim_ = np.zeros((max_objs, 3), dtype=np.float32)
    if "image_key" in meta.keys():
        meta["file_name"] = meta["image_key"]
    if filtered_name not in meta["file_name"]:
        ann_idx = -1
        # Addressing all objects in one image    note by zmj
        for ann in label:
            in_camera = ann["in_camera"] if "in_camera" in ann.keys() else ann
            if use_bbox2d and "bbox_2d" in ann and ann["bbox_2d"] is not None:
                # use image 2d bbox
                bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                if undistort_2dcenter:
                    a_bbox = ann["bbox_2d"]
                    bbox_cx = a_bbox[0] + a_bbox[2] / 2.0
                    bbox_cy = a_bbox[1] + a_bbox[3] / 2.0
                    proj_p = getUnDistortPoints(
                        np.array([[bbox_cx, bbox_cy]]),
                        np.array(calib[:3, :3]),  # noqa
                        np.array(meta["distCoeffs"]),
                        img_wh=meta["orgin_wh"],
                    )  # noqa
                    proj_p = proj_p.reshape(-1, 2)
                    proj_p = proj_p.astype(np.int)
                    bbox_cx = proj_p[0, 0]
                    bbox_cy = proj_p[0, 1]
                    cx, cy = calib[0, 2], calib[1, 2]
                    location_offset = np.array(
                        [
                            elem * in_camera["depth"] / calib[0, 0]
                            for elem in [
                                bbox_cx - cx,
                                bbox_cy - cy,
                                calib[0, 0],
                            ]
                        ]  # noqa             # back-projected 2dbox center to CCS   zmj
                    ).astype(np.float64)
                    location_offset[1] += in_camera["dim"][0] / 2.0    # turn to center of bottom   zmj
                    location_offset = in_camera["location"] - location_offset
                else:
                    location_offset = in_camera["location_offset_2d"]
                alpha = convert_alpha(in_camera["alpha_2d"], alpha_in_degree)
            elif use_project_bbox2d:
                # use lidar 3d projection 2d bbox
                bbox = xywh_to_x1y1x2y2(ann["bbox"])
                location_offset = in_camera["location_offset"]
                alpha = convert_alpha(in_camera["alpha"], alpha_in_degree)
            else:
                continue
            cls_id = int(classid_map[ann["category_id"]])
            if cls_id < 0:
                continue

            # 将bbox转换到特征图上的位置和尺寸    note by zmj
            bbox[:2] = affine_transform([bbox[:2]], trans_output)
            bbox[2:] = affine_transform([bbox[2:]], trans_output)

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)
            _wh = bbox[2:] - bbox[:2]
            if np.any(_wh <= 0):
                continue
            # filter bbox
            if (
                # ann["bbox_2d"][2] < min_box_edge
                # or ann["bbox_2d"][3] < min_box_edge
                ann["bbox"][2] < min_box_edge
                or ann["bbox"][3] < min_box_edge
            ):  # noqa
                ignore_mask[
                    int(bbox[1]) : int(bbox[3] + 1),
                    int(bbox[0]) : int(bbox[2] + 1),
                    :,
                ] = 1.0  # noqa
                continue
            # filter by depth
            if in_camera["depth"] > max_depth:
                ignore_mask[
                    int(bbox[1]) : int(bbox[3] + 1),
                    int(bbox[0]) : int(bbox[2] + 1),
                    :,
                ] = 1.0  # noqa
                continue
            # w, h = _wh
            ct = (bbox[:2] + bbox[2:]) / 2
            ct_int = tuple(ct.astype(np.int32).tolist())

            ann_idx += 1
            if ann_idx >= max_objs:
                break
            alpha_x[ann_idx] = alpha  # alpha will be delete, next version
            ind_[ann_idx] = ct_int[1] * output_width + ct_int[0]
            ind_mask_[ann_idx] = 1
            loc_[ann_idx] = in_camera["location"]
            rot_y_[ann_idx] = in_camera["rotation_y"]
            dim_[ann_idx] = in_camera["dim"]
            # ttfnet style
            insert_hm = get_gaussian2D(_wh)

            insert_hm_wh = insert_hm.shape[:2][::-1]    # insert_hm shape: (h,w)     note by zmj
            insert_reg_map_list = [
                get_reg_map(insert_hm_wh, in_camera["depth"]),
                get_reg_map(insert_hm_wh, _wh),
                get_reg_map(insert_hm_wh, in_camera["dim"]),
                get_dense_locoffset(
                    insert_hm_wh,
                    ct_int,
                    location_offset[:2],
                    in_camera["location"],
                    in_camera["dim"],
                    calib,
                    trans_output,
                    meta.get("distCoeffs"),
                    undistort_2dcenter,
                ),
            ]
            reg_map_list = [
                depth,
                wh,
                dim,
                loc_offset,  # rotbin, rotres
            ]
            draw_heatmap(hm[:, :, cls_id], insert_hm, ct_int)
            draw_heatmap(
                weight_hm, insert_hm, ct_int, reg_map_list, insert_reg_map_list
            )
            point_pos_mask[ct_int[1], ct_int[0]] = 1

    if normalize_depth:
        if undistort_depth_uv:
            eq_fu, eq_fv = cal_equivalent_focal_length_uv_mat(
                size[0], size[1], calib, meta["distCoeffs"]
            )  # noqa
            resized_eq_fu = cv2.warpAffine(
                eq_fu, trans_output, (output_width, output_height)
            )  # noqa
            resized_eq_fv = cv2.warpAffine(
                eq_fv, trans_output, (output_width, output_height)
            )  # noqa
            depth_u = depth * focal_length_default / resized_eq_fu
            depth_v = depth * focal_length_default / resized_eq_fv
            depth = np.stack([depth_u, depth_v], axis=-1)
        else:
            depth *= focal_length_default / calib[0, 0]
            depth = depth[:, :, np.newaxis]
        loc_offset *= focal_length_default / calib[0, 0]

    ret = {
        "heatmap": hm,
        "box2d_wh": wh,
        "dimensions": dim,
        "location_offset": loc_offset,  # noqa
        "depth": depth,  # noqa
        "heatmap_weight": weight_hm[:, :, np.newaxis],
        "point_pos_mask": point_pos_mask[:, :, np.newaxis],
    }
    if enable_ignore_area:
        ret["ignore_mask"] = ignore_mask

    for k in ret.keys():
        ret[k] = ret[k].transpose(2, 0, 1)
    ret.update(
        {
            "index": ind_,
            "index_mask": ind_mask_,
            "alpha_x": alpha_x,
            "location": loc_,
            "rotation_y": rot_y_,
            "dimensions_": dim_,
        }
    )
    del label

    return ret


def dense3d_pad_after_label_generator(gt, input_padding, down_stride):
    # heatmap (1, 80,120) -> (1, 80, 128)
    output_width = gt["heatmap"].shape[2]
    left_w = input_padding[0] // down_stride
    right_w = input_padding[1] // down_stride
    gt["heatmap"] = np.pad(
        gt["heatmap"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["box2d_wh"] = np.pad(
        gt["box2d_wh"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["dimensions"] = np.pad(
        gt["dimensions"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["location_offset"] = np.pad(
        gt["location_offset"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["depth"] = np.pad(
        gt["depth"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["heatmap_weight"] = np.pad(
        gt["heatmap_weight"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["point_pos_mask"] = np.pad(
        gt["point_pos_mask"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["ignore_mask"] = np.pad(
        gt["ignore_mask"],
        ((0, 0), (0, 0), (left_w, right_w)),
        "constant",
        constant_values=(1, 1),
    )

    center_x = gt["index"] % output_width
    center_y = gt["index"] // output_width

    gt["index"] = (
        center_y * (output_width + left_w + right_w) + center_x + left_w
    )
    return gt


def cal_equivalent_focal_length_uv_mat(
    width,
    height,
    calib,
    dist,
    downsample=4,
    undistort_by_cv=False,
):
    if isinstance(dist, list):
        dist = np.array(dist, dtype=np.float32)
    hash_distcoeffs = dist.tobytes()
    if hash_distcoeffs in GlobalEQFUMap.keys():
        eq_fu = GlobalEQFUMap[hash_distcoeffs]
        eq_fv = GlobalEQFVMap[hash_distcoeffs]
    else:
        ori_width = width
        ori_height = height
        width //= downsample
        height //= downsample

        u_pos = np.arange(0, ori_width, downsample, dtype=np.float32)
        u_pos = np.tile(u_pos, (height, 1))
        v_pos = np.arange(0, ori_height, downsample, dtype=np.float32)
        v_pos = np.tile(v_pos, (width, 1)).transpose()

        uv_points = np.stack([u_pos, v_pos]).reshape((2, -1)).transpose()
        uv_points = uv_points.reshape(-1, 1, 2)
        e_f = cal_equivalent_focal_length(
            uv_points, calib, dist, (ori_width, ori_height), undistort_by_cv
        )
        e_f_mat = e_f.reshape(height, width)

        cu = calib[0, 2]
        cv = calib[1, 2]
        dfu = np.zeros_like(e_f_mat)
        dfv = np.zeros_like(e_f_mat)
        offset_u = u_pos - cu
        offset_v = v_pos - cv

        dfu[:, 1:-1] = (e_f_mat[:, 2:] - e_f_mat[:, :-2]) / (2 * downsample)
        dfu[:, 0] = (e_f_mat[:, 1] - e_f_mat[:, 0]) / downsample
        dfu[:, -1] = (e_f_mat[:, -1] - e_f_mat[:, -2]) / downsample
        eq_fu = e_f_mat / (1 - dfu * offset_u / e_f_mat).clip(min=1)
        eq_fu.clip(min=1)

        dfv[1:-1] = (e_f_mat[2:] - e_f_mat[:-2]) / (2 * downsample)
        dfv[0] = (e_f_mat[1] - e_f_mat[0]) / downsample
        dfv[-1] = (e_f_mat[-1] - e_f_mat[-2]) / downsample
        eq_fv = e_f_mat / (1 - dfv * offset_v / e_f_mat).clip(min=1)
        eq_fv.clip(min=1)

        interp_eq_fu = interp2d(
            np.arange(0, ori_width, downsample),
            np.arange(0, ori_height, downsample),
            eq_fu,
            kind="linear",
        )
        interp_eq_fv = interp2d(
            np.arange(0, ori_width, downsample),
            np.arange(0, ori_height, downsample),
            eq_fv,
            kind="linear",
        )
        eq_fu = interp_eq_fu(np.arange(ori_width), np.arange(ori_height))
        eq_fv = interp_eq_fv(np.arange(ori_width), np.arange(ori_height))
        eq_fu = eq_fu.astype(np.float32)
        eq_fv = eq_fv.astype(np.float32)

        GlobalEQFUMap[hash_distcoeffs] = eq_fu
        GlobalEQFVMap[hash_distcoeffs] = eq_fv

    return eq_fu, eq_fv


def cal_equivalent_focal_length(
    uv_points,
    calib,
    dist,
    img_wh,
    undistort_by_cv=False,
):
    """Summary.

    Parameters
    ----------
    uv_points : numpy.NDArray
        Distorted uv points, in shape (num_points, 1, 2)
    calib : numpy.NDArray
        Calibration mat
    dist : numpy.NDArray
        Distort mat
    img_wh : tuple
        Image width and height
    undistort_by_cv : bool, default False
        Whether use opencv to undistort
    """
    if isinstance(calib, list):
        calib = np.array(calib, dtype=np.float32)
    if calib.shape[0] > 3:
        calib = calib[:3]
    if calib.shape[1] > 3:
        calib = calib[:, :3]
    if isinstance(dist, list):
        dist = np.array(dist, dtype=np.float32)

    fu = calib[0, 0]
    fv = calib[1, 1]
    f = (fu + fv) / 2
    cu = calib[0, 2]
    cv = calib[1, 2]
    dist_u = uv_points[:, 0, 0]
    dist_v = uv_points[:, 0, 1]

    if undistort_by_cv:
        undist_uv = cv2.undistortPointsIter(
            src=uv_points,
            cameraMatrix=calib,
            distCoeffs=dist,
            R=None,
            P=calib,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER, 4, 0.1),
        )
        undist_u = undist_uv[:, 0, 0]
        undist_v = undist_uv[:, 0, 1]
    else:
        undist_uv = getUnDistortPoints(
            points=uv_points[:, 0, :],
            calib=calib,
            distCoeffs=dist,
            img_wh=img_wh,
        )
        undist_u = undist_uv[:, 0]
        undist_v = undist_uv[:, 1]
    x = (undist_u - cu) / fu
    y = (undist_v - cv) / fv
    e_x = (dist_u - cu) / fu
    e_y = (dist_v - cv) / fv

    r = np.sqrt(x * x + y * y)
    e_r = np.sqrt(e_x * e_x + e_y * e_y)

    e_f = e_r.clip(min=1e-4) * f / r.clip(min=1e-4)

    # image center points is not stable
    e_f[e_r < 1e-3] = f
    e_f = e_f.clip(min=f * 0.1, max=f)

    return e_f
