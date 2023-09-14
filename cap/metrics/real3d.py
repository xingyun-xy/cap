# Copyright (c) Changan Auto. All rights reserved.

import collections
import copy
import logging
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from cap.metrics.metric import EvalMetric
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import convert_numpy as to_numpy
from cap.utils.distributed import get_dist_info
from cap.visualize.real3d import camera2velo, compute_2d_box_V2

__all__ = ["Real3dEval"]

logger = logging.getLogger(__name__)


def xywh_to_x1y1x2y2(bboxes):
    if isinstance(bboxes, (list, tuple)):
        bboxes = np.array(bboxes)

    bboxes[..., 2:] += bboxes[..., :2]
    return bboxes


def get_orientation(rotation_y):
    if isinstance(rotation_y, list):
        rotation_y = rotation_y[0]
    if np.pi / 4 < abs(rotation_y) < 3 * np.pi / 4:
        return "vertical"
    else:
        return "horizontal"


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
    x_min1, y_min1, x_max1, y_max1 = np.split(boxes1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = np.split(boxes2, 4, axis=1)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape, dtype="f4"),
        all_pairs_min_xmax - all_pairs_max_xmin,
    )

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape, dtype="f4"),
        all_pairs_min_ymax - all_pairs_max_ymin,
    )
    return intersect_widths * intersect_heights


def IoU(boxes1, boxes2):
    eps = 1e-6
    intersect = intersection(boxes1, boxes2)
    area1 = np.expand_dims(area(boxes1), axis=1)
    area2 = np.expand_dims(area(boxes2), axis=0)
    union = area1 + area2 - intersect
    return intersect / (union + eps)


def IoU_based_matching(
    det_boxes,
    det_loc,
    gt_boxes,
    gt_loc,
    iou_thresh,
    det_scores,
    gt_ignore_mask=None,
):
    det_loc = np.array(det_loc)
    gt_loc = np.array(gt_loc)
    assert len(det_boxes) == len(det_scores)

    overlaps = np.zeros(shape=gt_boxes.shape[0])
    det_assign = np.zeros(shape=gt_boxes.shape[0], dtype=np.int) - 1
    matched_det = np.zeros(shape=det_boxes.shape[0], dtype=np.int)

    valid_gt = np.ones(gt_boxes.shape[0], dtype=bool)
    pairwise_iou = IoU(det_boxes, gt_boxes)
    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_boxes.shape[0], dtype=bool)

    for dt_idx in det_sorted_idx:
        dt_idx = dt_idx[0]
        dxy = (
            np.sum((det_loc[dt_idx, [0, 2]] - gt_loc[:, [0, 2]]) ** 2, axis=1)
            ** 0.5
        )
        gt_dist = np.sum(gt_loc[:, [0, 2]] ** 2, axis=1) ** 0.5
        dist_err = dxy / gt_dist
        gt_mask = ((gt_dist <= 50) | (dist_err <= 0.2)) & valid_gt
        det_iou = pairwise_iou[dt_idx] * gt_mask.astype(float)
        max_ind = np.argmax(det_iou)
        max_iou = det_iou[max_ind]
        if max_iou > iou_thresh:
            if gt_ignore_mask is not None and gt_ignore_mask[max_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[max_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[max_ind] = max_iou
                valid_gt[max_ind] = 0

    redundant = np.where(matched_det == 0)[0]

    if gt_ignore_mask is not None:
        return (
            {"overlaps": overlaps, "det_assign": det_assign},
            redundant,
            det_ignored_mask,
        )
    else:
        return {"overlaps": overlaps, "det_assign": det_assign}, redundant


# TODO xuefangwang
def get_closest_points(loc, yaw_rad, length):
    addition1 = [
        (np.cos(yaw_rad) * length / 2.0)[:, None],
        (-np.sin(yaw_rad) * length / 2.0)[:, None],
    ]
    addition1 = np.concatenate(addition1, axis=1)
    addition2 = [
        (-np.cos(yaw_rad) * length / 2.0)[:, None],
        (np.sin(yaw_rad) * length / 2.0)[:, None],
    ]
    addition2 = np.concatenate(addition2, axis=1)

    addition = np.zeros_like(addition1)
    mask = addition1[:, 1] < addition2[:, 1]
    # mask = addition1 < addition2

    # addition[mask] = addition1[mask]
    # mask = np.invert(mask)
    # addition[mask] = addition2[mask]

    addition[mask[0]] = addition1[mask[0]]
    mask = np.invert(mask)
    addition[mask[0]] = addition2[mask[0]]

    # closest_points = loc + addition  #TODO xuefangwang
    closest_points = loc + addition[:, :, 0]
    # closest_points = loc + addition.reshape(loc.shape)
    return closest_points


# TODO xuefangwang
def collect_data(batch, outputs, gt_cids, det_cids):
    _gt_group_by_cid = {cid: [] for cid in gt_cids}
    _det_group_by_cid = {cid: [] for cid in det_cids}
    _images = []

    result_keys = [
        "dim",
        "category_id",
        "score",
        "center",
        "bbox",
        "dep",
        "alpha",
        "location",
        "rotation_y",
    ]

    for output in outputs:
        if "track_offset" in output:
            result_keys.append("track_offset")
        results = {k: output[k] for k in result_keys}
        image_id = output["image_id"][0]

        pred = {key: val for key, val in results.items()}
        pred["image_id"] = image_id
        cid = pred["category_id"][0]
        if cid in _det_group_by_cid:
            _det_group_by_cid[cid].append(pred)

    # image info
    img_info = {
        "file_name": batch["image_name"],
        "img_size": (
            to_numpy(batch["image_width"]),
            to_numpy(batch["image_height"]),
        ),
        "calib": to_numpy(batch["calib"]),
        "id": image_id,
        "distCoeffs": to_numpy(batch["distCoeffs"]),
    }
    if "Tr_vel2cam" in batch:
        img_info.update({"Tr_vel2cam": to_numpy(batch["Tr_vel2cam"])})
    else:
        img_info.update({"Tr_vel2cam": np.zeros((4, 4), dtype=np.float32)})
    if "index" in batch:
        img_info.update({"img_index": to_numpy(batch["index"])})
    # else:
    #     img_info.update({"img_index": i})
    _images.append(img_info)

    # ground truth
    annotations = batch["annotations"]
    # cid = to_numpy(annotations["category_id"])[0]
    # if cid in _gt_group_by_cid:
    #     _gt_group_by_cid[cid].append(annotations)

    for anno in annotations:
        cid = to_numpy(anno["category_id"])[0]
        new_anno = {}
        for key in anno.keys():
            if key == "in_camera":
                in_camera = anno[key]
                for in_key in in_camera.keys():
                    # new_anno[in_key] = to_numpy(in_camera[in_key])
                    if type(to_numpy(in_camera[in_key])) is np.ndarray:
                        new_anno[in_key] = to_numpy(in_camera[in_key]).tolist()
                    # elif in_key == "dim":
                    #     new_anno[in_key] = [
                    #         sub_in[0] for sub_in in to_numpy(in_camera[in_key])
                    #     ]
                    # elif in_key == "dim":
                    #     new_anno[in_key] = [
                    #         sub_in[0] for sub_in in to_numpy(in_camera[in_key])
                    #     ]
                    else:
                        # new_anno[in_key] = to_numpy(in_camera[in_key])
                        new_anno[in_key] = [
                            sub_in[0] for sub_in in to_numpy(in_camera[in_key])
                        ]

            if type(to_numpy(anno[key])) is np.ndarray:
                new_anno[key] = to_numpy(anno[key]).tolist()
            elif key == "bbox_2d":
                new_anno[key] = [box[0] for box in to_numpy(anno[key])]
            elif key == "bbox":
                new_anno[key] = [box[0] for box in to_numpy(anno[key])]
            else:
                new_anno[key] = to_numpy(anno[key])

        if cid in _gt_group_by_cid:
            _gt_group_by_cid[cid].append(new_anno)
    return _gt_group_by_cid, _det_group_by_cid, _images


def match(
    det_res,
    annotation,
    eval_camera,
    iou_threshold,
    score_threshold,
    gt_max_depth,
    fisheye,
    match_basis,
):
    assert match_basis in ["det2d", "proj2d", "bev2d"]

    img_count = 0
    gt_missed, redundant_det = [], []
    all_dets = defaultdict(list)
    all_gts = defaultdict(list)
    dets_matched = {}
    diou_error = {}
    hard_img_infos = {}

    for det in det_res:
        if det["score"][0] < score_threshold:
            continue
        all_dets[det["image_id"]].append(det)

    for gt in annotation["annotations"]:
        gt_depth = gt["depth"][0]
        if gt_depth > gt_max_depth:
            continue
        # if gt.get("ignore", False):
        # gt["ignore"] = gt["ignore"][0]
        # if gt.get("ignore", False):
        #     continue
        all_gts[gt["image_id"][0]].append(gt)
    imgid_dict = {}
    for img_info in annotation["images"]:
        # if eval_camera not in img_info["file_name"]:
        #     continue
        calib = img_info["calib"]
        img_id = img_info["id"]
        # img_index = int(img_info["img_index"])
        img_index = img_info["id"]
        # img_id = img_index
        imgid_dict[img_id] = img_index
        img_size = img_info["img_size"]
        dist_coeff = img_info.get("distCoeffs", None)
        Tr_vel2cam = img_info.get("Tr_vel2cam", None)
        det_bboxes, det_scores, det_locs, det_yaw = [], [], [], []
        gt_bboxes, gt_locs, gt_yaw = [], [], []

        # if len(all_gts[img_id]) == 0:
        if len(all_gts[img_index]) == 0:
            for det in all_dets[img_id]:
                redundant_det.append(
                    [float(det["dep"][0]), float(det["rotation_y"][0])]
                )
            continue

        hard_img_info = []
        # for gt in all_gts[img_id]:
        for gt in all_gts[img_index]:
            dim = gt["dim"]
            location = gt["location"]
            yaw = gt["rotation_y"]

            # dim = [dim_i[0] for dim_i in gt["dim"]]
            # location = [location_i[0] for location_i in gt["location"]]
            # yaw = gt["rotation_y"][0]
            if match_basis == "proj2d":
                _, bbox2d, _ = compute_2d_box_V2(
                    dim,
                    location,
                    yaw,
                    calib,
                    dist_coeff,
                    fisheye,
                    img_w=float(img_size[0]),
                    img_h=float(img_size[1]),
                )
            elif match_basis == "det2d":
                # original gt bbox format is `x1,y1,w,h`
                # bbox2d = xywh_to_x1y1x2y2(gt["bbox"])
                # gt_bbox = [box[0] for box in gt["bbox"]]
                bbox2d = xywh_to_x1y1x2y2(gt["bbox"])
            elif match_basis == "bev2d":
                bbox2d = [
                    location[0] - dim[1] / 2,
                    -location[2] - dim[2] / 2,
                    location[0] + dim[1] / 2,
                    -location[2] + dim[2] / 2,
                ]
            else:
                raise KeyError(match_basis)

            if bbox2d is None:
                continue
            gt_bboxes.append(bbox2d)
            gt_locs.append(location)
            gt_yaw.append(yaw)
            hard_img_info.append(
                [gt["category_id"][0], gt["depth"][0], 1, 360]
            )

        if len(all_dets[img_id]) == 0:
            for loc, yaw in zip(gt_locs, gt_yaw):
                gt_missed.append([loc[-1], yaw])
            if len(gt_bboxes) > 0:
                hard_img_infos[img_index] = hard_img_info
            continue

        # project 3d to 2d box
        for det in all_dets[img_id]:
            dim = det["dim"]
            location = det["location"]
            yaw = det["rotation_y"]
            if match_basis == "proj2d":
                _, bbox2d, _ = compute_2d_box_V2(
                    dim,
                    location,
                    yaw,
                    calib,
                    dist_coeff,
                    fisheye,
                    img_w=float(img_size[0]),
                    img_h=float(img_size[1]),
                )
            elif match_basis == "det2d":
                bbox2d = det["bbox"]
            elif match_basis == "bev2d":
                bbox2d = [
                    location[0] - dim[1] / 2,
                    -location[2] - dim[2] / 2,
                    location[0] + dim[1] / 2,
                    -location[2] + dim[2] / 2,
                ]
            else:
                raise KeyError(match_basis)
            if bbox2d is None:
                continue
            det_bboxes.append(bbox2d)
            assert det["score"][0] >= 0
            det_scores.append(det["score"])
            det_locs.append(location)
            det_yaw.append(yaw)

        if len(gt_bboxes) == 0:
            for loc, yaw in zip(det_locs, det_yaw):
                redundant_det.append([loc[-1], yaw])
            continue
        if len(det_bboxes) == 0:
            for loc, yaw in zip(gt_locs, gt_yaw):
                gt_missed.append([loc[-1], yaw])
            continue

        det_bboxes = np.array(det_bboxes)
        det_scores = np.array(det_scores)
        gt_bboxes = np.array(gt_bboxes)
        assert det_scores.shape[0] == det_bboxes.shape[0]

        # if match_basis in ["det2d", "proj2d"]:
        #     gt_bboxes = np.clip(gt_bboxes, [0, 0, 0, 0], img_size * 2)
        #     det_bboxes = np.clip(det_bboxes, [0, 0, 0, 0], img_size * 2)

        (matched_dict, redundant) = IoU_based_matching(
            det_bboxes,
            det_locs,
            gt_bboxes,
            gt_locs,
            iou_threshold,
            det_scores,
        )

        for i in redundant:
            redundant_det.append(
                [
                    float(all_dets[img_id][i]["dep"][0]),
                    float(all_dets[img_id][i]["rotation_y"][0]),
                ]
            )
        tp = np.ones(len(det_scores), dtype=bool)
        tp[redundant] = 0

        det_assigns = matched_dict["det_assign"]
        inds = np.array(list(range(len(det_assigns))))

        mask = det_assigns != -1
        for i_miss, flag in enumerate(det_assigns == -1):
            if flag:
                gt_missed.append([gt_locs[i_miss][-1], gt_yaw[i_miss]])
        if np.sum(mask) == 0:
            hard_img_infos[img_index] = hard_img_info
            continue
        diou_error[img_id] = 1.0 - matched_dict["overlaps"][mask]
        det_assigns, inds = det_assigns[mask], inds[mask]

        dets_matched[img_id] = collections.OrderedDict()
        if np.sum(np.invert(mask)) > 0:
            hard_img_infos[img_index] = [
                hard_img_info[i] for i in range(len(mask)) if not mask[i]
            ]

        for i, j in zip(inds, det_assigns):
            # gt_info = all_gts[img_id][i]
            gt_info = all_gts[img_index][i]
            det_info = all_dets[img_id][j]
            if np.sum(np.abs(Tr_vel2cam)) > 0.001:
                Tr_vel2cam = np.array(Tr_vel2cam, dtype=np.float32)
                gt_loc_vel, gt_yaw_vel = camera2velo(
                    gt_info["location"],
                    gt_info["rotation_y"],
                    Tr_vel2cam,
                )
                det_loc_vel, det_yaw_vel = camera2velo(
                    det_info["location"],
                    det_info["rotation_y"],
                    Tr_vel2cam,
                )
                gt_info.update({"loc_vel": gt_loc_vel})
                gt_info.update({"rotation_vel": gt_yaw_vel})
                det_info.update({"loc_vel": det_loc_vel})
                det_info.update({"rotation_vel": det_yaw_vel})

            global_id = gt_info.get("global_id", None)
            global_id = global_id if global_id else i
            dets_matched[img_id][global_id] = [gt_info, det_info]
        img_count += 1
    return (
        dets_matched,
        diou_error,
        gt_missed,
        redundant_det,
        img_count,
        hard_img_infos,
        imgid_dict,
    )


def metric_sub(det_matched, eps=1e-7):
    pred_dim = np.array([d["dim"] for g, d in det_matched.values()])
    pred_loc = np.array([d["location"] for g, d in det_matched.values()])
    pred_yaw_rad = np.array([d["rotation_y"] for g, d in det_matched.values()])
    pred_yaw = np.rad2deg(pred_yaw_rad) % 360.0

    gt_dim = np.array([g["dim"] for g, d in det_matched.values()])
    gt_loc = np.array([g["location"] for g, d in det_matched.values()])
    gt_yaw_rad = np.array([g["rotation_y"] for g, d in det_matched.values()])
    gt_yaw = np.rad2deg(gt_yaw_rad) % 360.0

    dx = np.abs(pred_loc[:, 2] - gt_loc[:, 2])
    dy = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
    dxy = (dx ** 2 + dy ** 2) ** 0.5
    dw = np.abs(pred_dim[:, 1] - gt_dim[:, 1])
    dl = np.abs(pred_dim[:, 2] - gt_dim[:, 2])

    # gt_closest_points = get_closest_points(gt_loc[:, [0, 2]].tolist(),
    #                                        gt_yaw_rad.tolist(),
    #                                        gt_dim[:, 2].tolist())
    # pred_closest_points = get_closest_points(pred_loc[:, [0, 2]].tolist(),
    #                                          pred_yaw_rad.tolist(),
    #                                          pred_dim[:, 2].tolist())

    gt_closest_points = get_closest_points(
        gt_loc[:, [0, 2]], gt_yaw_rad, gt_dim[:, 2]
    )
    pred_closest_points = get_closest_points(
        pred_loc[:, [0, 2]], pred_yaw_rad, pred_dim[:, 2]
    )
    # gt_closest_points = np.expand_dims(gt_closest_points.squeeze(), axis=0)
    # pred_closest_points = np.expand_dims(pred_closest_points.squeeze(), axis=0)
    dcxy = (
        np.sum((gt_closest_points - pred_closest_points) ** 2, axis=1) ** 0.5
    )

    dxp = dx / np.abs(gt_loc[:, 2])
    dyp = dy / np.abs(gt_loc[:, 0])
    dxyp = dxy / np.abs(gt_loc[:, 0] ** 2 + gt_loc[:, 2] ** 2 + eps) ** 0.5
    dxy_10p_error = dxyp <= 0.1
    dwp = dw / np.abs(gt_dim[:, 1])
    dlp = dl / np.abs(gt_dim[:, 2])
    dcxyp = (
        dcxy
        / np.abs(
            gt_closest_points[:, 0] ** 2 + gt_closest_points[:, 1] ** 2 + eps
        )
        ** 0.5
    )

    abs_rot = np.abs(gt_yaw - pred_yaw)
    drot = np.minimum(abs_rot, 360.0 - abs_rot)
    res = {
        "dx": dx,
        "dy": dy,
        "dxy": dxy,
        "dw": dw,
        "dl": dl,
        "dxp": dxp,
        "dyp": dyp,
        "dxyp": dxyp,
        "dwp": dwp,
        "dlp": dlp,
        "drot": drot,
        "dxy_10p_error": dxy_10p_error,
        "dcxy": dcxy,
        "dcxyp": dcxyp,
    }
    return res


def do_metric(metrics, dep_thresh, dets_matched, diou_error, orientations):
    eps = 1e-5
    dep_thresh = np.array(dep_thresh)
    Metrics = {}
    for orientation in orientations:
        Metrics[orientation] = {}
        for metric_name in metrics:
            Metrics[orientation][metric_name] = [
                [] for _ in range(len(dep_thresh) + 1)
            ]

    hard_img_infos = []
    for img_id, det_matched in dets_matched.items():
        res = metric_sub(det_matched)
        res.update({"diou_error": diou_error[img_id]})
        for i, (g, _) in enumerate(det_matched.values()):
            gt_depth = g["depth"]
            gt_orientation = get_orientation(g["rotation_y"])
            dep_ind = np.sum(gt_depth > dep_thresh, axis=-1)
            for orientation in ["all", gt_orientation]:
                if orientation not in orientations:
                    continue
                for metric_name in metrics:
                    Metrics[orientation][metric_name][dep_ind].append(
                        res[metric_name][i]
                    )
                    # res[metric_name][:, i])

        key = "dtxyp"
        gt_trk_offset = []
        det_trk_offset = []
        gt_depth_valid = []
        for g, d in det_matched.values():
            if "track_offset" in g and "track_offset" in d:
                if key not in res:
                    res.update({key: None})
                    Metrics.update(
                        {key: [[] for _ in range(len(dep_thresh) + 1)]}
                    )
                if (
                    abs(g["track_offset"][0]) < eps
                    and abs(g["track_offset"][1]) < eps
                ):
                    continue
                gt_trk_offset.append(g["track_offset"])
                det_trk_offset.append(d["track_offset"])
                gt_depth_valid.append(g["depth"])
        if key in res and len(gt_depth_valid) > 0:
            gt_trk_offset = np.array(gt_trk_offset)
            det_trk_offset = np.array(det_trk_offset)
            gt_depth_valid = np.array(gt_depth_valid)
            dtxyp = (
                np.sum((gt_trk_offset - det_trk_offset) ** 2, axis=1)
                / np.sum((gt_trk_offset) ** 2, axis=1)
                + eps
            ) ** 0.5
            dtxyp = np.clip(dtxyp, 0, 1)
            res.update({key: dtxyp})
            dep_thresh_inds = np.sum(
                gt_depth_valid[:, np.newaxis] > dep_thresh, axis=-1
            )
            dep_inds, cnts = np.unique(dep_thresh_inds, return_counts=True)
            for dep_ind in dep_inds:
                mask = dep_thresh_inds == dep_ind
                Metrics[key][dep_ind] += res[key][mask].tolist()

        hard_img_info = []
        for i, (g, _) in enumerate(det_matched.values()):
            dxyp, drot = res["dxyp"][i], res["drot"][i]
            if dxyp > 0.01 or drot > 1:  # TODO @xuefangwang
                # if dxyp.all() > 0.01 or drot.all() > 1:
                hard_img_info.append(
                    [g["category_id"], g["depth"], dxyp, drot]
                )
        if len(hard_img_info) > 0:
            hard_img_infos.append([img_id, hard_img_info])

    return Metrics, hard_img_infos


def statistic_orient_dep(dep_thresh, orientations, values):
    counts = {}
    for orientation in orientations:
        counts[orientation] = np.zeros(len(dep_thresh) + 1)
    if len(values) > 0:
        for dep, yaw in values:
            orientation = get_orientation(yaw)
            dep_thresh_ind = np.sum(
                np.array([dep])[:, np.newaxis] > dep_thresh, axis=-1
            )
            if orientation in counts:
                counts[orientation][dep_thresh_ind] += 1
            if "all" in counts:
                counts["all"][dep_thresh_ind] += 1
    return counts


def auto_eval(
    det_res,
    annotation,
    eval_camera,
    dep_thresh,
    iou_threshold,
    score_threshold,
    gt_max_depth,
    fisheye=False,
    metrics=("dxyp", "drot"),
    match_basis="det2d",
    orientations=("all",),
):
    (
        dets_matched,
        diou_error,
        gt_missed,
        redundant_det,
        img_count,
        hard_img_infos,
        imgid_dict,
    ) = match(
        det_res,
        annotation,
        eval_camera,
        iou_threshold,
        score_threshold,
        gt_max_depth,
        fisheye,
        match_basis,
    )

    res, hard_img_infos_error = do_metric(
        metrics, dep_thresh, dets_matched, diou_error, orientations
    )
    for hard_img_info_error in hard_img_infos_error:
        hard_img_id = imgid_dict[hard_img_info_error[0]]
        if hard_img_id not in hard_img_infos:
            hard_img_infos[hard_img_id] = []
        hard_img_infos[hard_img_id].extend(hard_img_info_error[1])

    res.update({"dets_matched": dets_matched})

    res["img_count"] = img_count
    count_gt_missed = statistic_orient_dep(dep_thresh, orientations, gt_missed)
    count_redundant_det = statistic_orient_dep(
        dep_thresh, orientations, redundant_det
    )
    for orientation in orientations:
        res[orientation]["fn"] = count_gt_missed[orientation]
        res[orientation]["fp"] = count_redundant_det[orientation]
        res[orientation]["tp"] = [len(x) for x in res[orientation][metrics[0]]]
    res.update({"hard_img_infos": hard_img_infos})
    return res


@OBJECT_REGISTRY.register
class Real3dEval(EvalMetric):
    """The real3d evla metrics.

    Args:
        num_dist (int): Length of dist coeffs.
        need_eval_categories (dict): The mapping between id between training
            and labeling.
        eval_camera_names (tuple of str): Tuple of camera names.
        metrics (tuple of str, default: None): Tuple of eval metrics, using
            ("dxyp", "drot") if not special. These two are most used metric.
            "dxyp" means precentage of xy position error, "drot" means rotation
            error.
        depth_intervals (tuple of int, default: None): Depth range to
            validation, using (20, 50, 80, 120) if not special.
        iou_threshold (float, defualt: 0.2): Threshold for IoU.
        score_threshold (float, defualt: 0.1): Threshold for score.
        gt_max_depth (float, default: 300): Max depth for gts.
        save_path (str, default: None): Path to save the predictions
        fisheye (bool, default: False): whether fish data
        match_basis (str, default: None): which bbox used in IoU match. You can
            use "det2d","proj2d","bev2d"
        orientations (tuple of str, default: ("all")): Tuple of eval orientation,
            using ("all", "horizontal", "vertical") to eval metric by tagert's
            orientation.
    """

    def __init__(
        self,
        *,
        num_dist: int,
        need_eval_categories: Dict,
        eval_camera_names: Optional[List] = None,
        metrics: Optional[Sequence[str]] = ("dxyp", "drot"),
        depth_intervals: Optional[Sequence[int]] = (10, 20, 50, 80, 120),
        iou_threshold: float = 0.2,
        score_threshold: float = 0.1,
        gt_max_depth: float = 300,
        save_path: Optional[str] = None,
        fisheye=False,
        match_basis=None,
        orientations: Optional[Sequence[str]] = ("all",),
    ):

        self.need_eval_categories = need_eval_categories
        self.eval_camera_names = eval_camera_names
        self.metrics = metrics
        self.orientations = orientations
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.gt_max_depth = gt_max_depth
        self.save_path = save_path
        self.fisheye = fisheye
        if match_basis is None:
            match_basis = "det2d"
        self.match_basis = match_basis
        self.depth_intervals = depth_intervals

        assert (
            self.gt_max_depth > depth_intervals[-1]
        ), "gt_max_depth must be greater than depth_intervals[-1]"

        self.eps = 1e-9
        self.num_dist = num_dist
        depth_intervals = [0] + list(depth_intervals) + [self.gt_max_depth]
        self.full_depth_intervals = [
            "({},{})".format(start, end)
            for start, end in zip(depth_intervals[:-1], depth_intervals[1:])
        ]

        gt_cids = []
        det_cids = []
        for _, cids in need_eval_categories.items():
            gt_cids.extend(cids[0])
            det_cids.append(cids[1])
        self.gt_cids = list(set(gt_cids))
        self.det_cids = list(set(det_cids))

        self.metric_index = 0
        self.real_save_path = None
        self.real_matched_path = None
        self.show_num_metric = ["tp", "fn", "fp"]

        super(Real3dEval, self).__init__("real3d")

    def _init_states(self):
        for name in self.get_names():
            self.add_state(
                name,
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )

    def save_result(self, batch, output):
        pkl_res = self.convert_to_save_format(batch, output)
        with open(self.real_save_path, "ab") as f:
            pickle.dump(pkl_res, f)

    def save_matched_result(self, dets_matched):
        with open(self.real_matched_path, "ab") as f:
            pickle.dump(dets_matched, f)

    def save_hard_image_infos(self, hard_img_infos):
        with open(self.real_hard_img_path, "ab") as f:
            pickle.dump(hard_img_infos, f)

    def update(self, batch, outputs):
        if self.save_path:
            dets_matched_res = {
                cate_name: {} for cate_name in self.need_eval_categories.keys()
            }
            self.save_result(batch, outputs)

        _gt_group_by_cid, _det_group_by_cid, _images = collect_data(
            batch, outputs, self.gt_cids, self.det_cids
        )

        hard_img_infos = {}
        for cam_name in self.eval_camera_names:
            for cate_name, (
                gt_ids,
                dt_id,
            ) in self.need_eval_categories.items():
                gt = {"images": _images, "annotations": []}
                for cate_id in gt_ids:
                    gt["annotations"].extend(_gt_group_by_cid[cate_id])

                if gt["annotations"] == []:
                    continue

                det = _det_group_by_cid[dt_id]
                res = auto_eval(
                    det,
                    gt,
                    cam_name,
                    self.depth_intervals,
                    self.iou_threshold,
                    self.score_threshold,
                    self.gt_max_depth,
                    self.fisheye,
                    metrics=self.metrics,
                    match_basis=self.match_basis,
                    orientations=self.orientations,
                )
                for hard_img_id, infos in res["hard_img_infos"].items():
                    if hard_img_id not in hard_img_infos:
                        hard_img_infos[hard_img_id] = infos
                    else:
                        hard_img_infos[hard_img_id].extend(infos)

                if self.save_path and res.get("dets_matched", None):
                    dets_matched_res[cate_name].update(res["dets_matched"])

                for orientation in self.orientations:
                    for i_dep, dep_interval in enumerate(
                        self.full_depth_intervals
                    ):
                        prefix = "{}_{}_{}_{}_".format(
                            cate_name, cam_name, orientation, dep_interval
                        )
                        for m_num in self.show_num_metric:
                            setattr(
                                self,
                                prefix + m_num,
                                getattr(self, prefix + m_num, None)
                                + res[orientation][m_num][i_dep],
                            )
                        for metric_name in self.metrics:
                            name = prefix + metric_name
                            setattr(
                                self,
                                name,
                                getattr(self, name, None)
                                + np.sum(res[orientation][metric_name][i_dep]),
                            )

        if self.save_path:
            self.save_matched_result(dets_matched_res)
            if len(hard_img_infos) > 0:
                self.save_hard_image_infos(hard_img_infos)

    def get_names(self):
        names = []
        for cam_name in self.eval_camera_names:
            for cate_name, _ in self.need_eval_categories.items():
                for orientation in self.orientations:
                    for dep_interval in self.full_depth_intervals:
                        prefix = "{}_{}_{}_{}_".format(
                            cate_name, cam_name, orientation, dep_interval
                        )
                        for m_num in self.show_num_metric:
                            names += [prefix + m_num]
                        for metric_name in self.metrics:
                            names += [prefix + metric_name]
        return names

    def print_log(self, format_metrics, orientation):
        sp_len = 12
        mp_len = 15
        lp_len = 24
        num_float = 4
        fromat_metrics_show = {}
        fromat_metrics_show["Camera+Category"] = []
        # print metrics
        log_line = "{} Real3d Metircs:\n".format(orientation)
        log_line += "Camera+Category".ljust(lp_len)
        for metric_name in self.metrics:
            for dep_interval in self.full_depth_intervals:
                head_name = f"{dep_interval}_{metric_name}"
                log_line += head_name.ljust(mp_len)
                fromat_metrics_show["Camera+Category"].append(head_name)
        counts_show = ["Recall", "Precision", "tp", "fn", "fp"]
        for m_num in counts_show:
            if len(m_num) == 2:
                log_line += m_num.ljust(sp_len - 4)
            else:
                log_line += m_num.ljust(sp_len)
            fromat_metrics_show["Camera+Category"].append(m_num)
        for dep_interval in self.full_depth_intervals:
            head_name = f"{dep_interval}_tp"
            log_line += head_name.ljust(sp_len)
            fromat_metrics_show["Camera+Category"].append(head_name)
        log_line += "\n"

        for cam, cate_dict in format_metrics.items():
            fromat_metrics_show[cam] = {}
            for cate, metric in cate_dict.items():
                cam_cate = cam.replace("__", "") + "+" + cate
                line = (cam_cate).ljust(lp_len)
                fromat_metrics_show[cam][cate] = [cam_cate]
                kpi_values = []
                for i_metric, _ in enumerate(self.metrics):
                    tp, fn, fp = 0, 0, 0
                    for i_dep, _ in enumerate(self.full_depth_intervals):
                        m_dep = metric[i_dep]
                        tp += m_dep[0]
                        fn += m_dep[1]
                        fp += m_dep[2]

                        kpi_value = round(
                            m_dep[i_metric + 3] / (m_dep[0] + self.eps),
                            num_float,
                        )
                        kpi_values.append(kpi_value)
                        line += str(kpi_value).ljust(mp_len)
                recall = tp / (tp + fn + self.eps)
                presicion = tp / (tp + fp + self.eps)
                line += str(round(recall, num_float)).ljust(sp_len)
                line += str(round(presicion, num_float)).ljust(sp_len)
                line += "".join(
                    [str(int(m)).ljust(sp_len) for m in [tp, fn, fp]]
                )
                for i_dep, _ in enumerate(self.full_depth_intervals):
                    line += str(int(metric[i_dep][0])).ljust(sp_len)
                log_line += line + "\n"

                fromat_metrics_show[cam][cate].extend(kpi_values)
                fromat_metrics_show[cam][cate].extend(
                    [recall, presicion, tp, fn, fp]
                )
                for i_dep, _ in enumerate(self.full_depth_intervals):
                    fromat_metrics_show[cam][cate].append(metric[i_dep][0])
        logger.info(log_line)

        return fromat_metrics_show

    def compute(self):
        self.metric_index += 1
        for orientation in self.orientations:
            format_metrics = {}
            for cam_name in self.eval_camera_names:
                format_metrics[cam_name] = {}
                for cate_name, _ in self.need_eval_categories.items():
                    format_metrics[cam_name][cate_name] = []
                    for dep_interval in self.full_depth_intervals:
                        prefix = "{}_{}_{}_{}_".format(
                            cate_name, cam_name, orientation, dep_interval
                        )

                        tmp_metric = []
                        for m_num in self.show_num_metric:
                            tmp_metric.append(
                                int(
                                    getattr(self, prefix + m_num, None)
                                    .cpu()
                                    .numpy()
                                )
                            )
                        for metric_name in self.metrics:
                            tmp_metric.append(
                                float(
                                    getattr(self, prefix + metric_name, None)
                                    .cpu()
                                    .numpy()
                                )
                            )
                        tmp_metric = np.array(tmp_metric)
                        format_metrics[cam_name][cate_name].append(tmp_metric)
            if len(self.eval_camera_names) > 1:
                format_metrics["cam_mean"] = {}
                for cam_name in self.eval_camera_names:
                    for cate_name, _ in self.need_eval_categories.items():
                        values = format_metrics[cam_name][cate_name]
                        if cate_name not in format_metrics["cam_mean"]:
                            format_metrics["cam_mean"][
                                cate_name
                            ] = copy.deepcopy(values)
                        else:
                            for i in range(len(values)):
                                format_metrics["cam_mean"][cate_name][
                                    i
                                ] += values[i]

            format_metrics_show = self.print_log(format_metrics, orientation)
            if orientation == "all":
                fromat_metrics_show_remaind = format_metrics_show
        return fromat_metrics_show_remaind

    def compute_2(self):
        self.metric_index += 1
        for orientation in self.orientations:
            format_metrics = {}
            for cam_name in self.eval_camera_names:
                format_metrics[cam_name] = {}
                for cate_name, _ in self.need_eval_categories.items():
                    format_metrics[cam_name][cate_name] = []
                    for dep_interval in self.full_depth_intervals:
                        prefix = "{}_{}_{}_{}_".format(
                            cate_name, cam_name, orientation, dep_interval
                        )

                        tmp_metric = []
                        for m_num in self.show_num_metric:
                            tmp_metric.append(
                                int(
                                    getattr(self, prefix + m_num, None)
                                    .cpu()
                                    .numpy()
                                )
                            )
                        for metric_name in self.metrics:
                            tmp_metric.append(
                                float(
                                    getattr(self, prefix + metric_name, None)
                                    .cpu()
                                    .numpy()
                                )
                            )
                        tmp_metric = np.array(tmp_metric)
                        format_metrics[cam_name][cate_name].append(tmp_metric)
            if len(self.eval_camera_names) > 1:
                format_metrics["cam_mean"] = {}
                for cam_name in self.eval_camera_names:
                    for cate_name, _ in self.need_eval_categories.items():
                        values = format_metrics[cam_name][cate_name]
                        if cate_name not in format_metrics["cam_mean"]:
                            format_metrics["cam_mean"][
                                cate_name
                            ] = copy.deepcopy(values)
                        else:
                            for i in range(len(values)):
                                format_metrics["cam_mean"][cate_name][
                                    i
                                ] += values[i]

            format_metrics_show = self.print_log(format_metrics, orientation)
            if orientation == "all":
                fromat_metrics_show_remaind = format_metrics_show
        return fromat_metrics_show_remaind

    def get(self):
        format_metrics = self.compute_2()
        names = ["Camera+Category"]
        names.extend(format_metrics["Camera+Category"])
        values = [[] for _ in range(len(names))]
        for name, value in format_metrics.items():
            if name == "Camera+Category":
                continue
            for _, val in value.items():
                for i, v in enumerate(val):
                    values[i].append(v)
        return names, values

    def convert_to_save_format(self, batch, output):
        res = []
        batch_size = output["dep"].shape[0]
        det_num = output["dep"].shape[1]

        for i in range(batch_size):
            img_name = batch["image_name"][i]
            meta = {
                "calib": to_numpy(batch["calibration"][i]),
                "img_name": img_name,
                "img_id": batch["image_id"][i],
                "distCoeffs": to_numpy(batch["dist_coeffs"][i]),
                "c": np.array(
                    [
                        to_numpy(batch["image_transform"]["input_size"][0][i]),
                        to_numpy(batch["image_transform"]["input_size"][1][i]),
                    ]
                ),
                "s": np.array(
                    [
                        to_numpy(
                            batch["image_transform"]["original_size"][0][i]
                        ),
                        to_numpy(
                            batch["image_transform"]["original_size"][1][i]
                        ),
                    ]
                ),
                "img_size": np.array(
                    [batch["img"][0][i].shape[2], batch["img"][0][i].shape[1]]
                ),
            }
            if "Tr_vel2cam" in batch:
                meta.update({"Tr_vel2cam": to_numpy(batch["Tr_vel2cam"][i])})
            dets = []
            for j in range(det_num):
                dets.append(
                    {
                        "dep": np.array([output["dep"][i][j].cpu().detach()]),
                        "dim": to_numpy(output["dim"][i][j]),
                        "class_id": to_numpy(
                            output["category_id"][i][j], dtype="int16"
                        ),
                        "score": to_numpy(output["score"][i][j]),
                        "bbox": to_numpy(output["bbox"][i][j]),
                        "center": to_numpy(output["center"][i][j]),
                        "rotation_y": to_numpy(output["rotation_y"][i][j]),
                        "location": to_numpy(output["location"][i][j]),
                        "alpha": to_numpy(output["alpha"][i][j]),
                    }
                )
                if "track_offset" in output:
                    dets[-1].update(
                        {
                            "track_offset": to_numpy(
                                output["track_offset"][i][j]
                            )
                        }
                    )
            if "track_offset" in output:
                gts_trk_offset = []
                for ann in batch["annotations"]:
                    gts_trk_offset.append(
                        [ann["global_id"], ann["track_offset"]]
                    )
                res.append(
                    {
                        "dets": dets,
                        "meta": meta,
                        "gts_trk_offset": gts_trk_offset,
                    }
                )
            else:
                res.append({"dets": dets, "meta": meta})

        return res

    def reset(self) -> None:
        super().reset()
        if self.save_path:
            rank, word_size = get_dist_info()
            self.real_save_path = os.path.join(
                self.save_path, str(self.metric_index), f"dets/{rank}.pkl"
            )
            self.real_matched_path = os.path.join(
                self.save_path,
                str(self.metric_index),
                f"dets_matched/{rank}.pkl",
            )
            self.real_hard_img_path = os.path.join(
                self.save_path,
                str(self.metric_index),
                f"hard_img_ids/{rank}.pkl",
            )
            pkl_save_root = os.path.dirname(self.real_save_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)
            pkl_save_root = os.path.dirname(self.real_matched_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)
            pkl_save_root = os.path.dirname(self.real_hard_img_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)
