# Copyright (c) Changan Auto. All rights reserved.

import json
import math
from collections import defaultdict
from io import BytesIO
from typing import Mapping, Optional, Sequence

import matplotlib
import numpy as np
import pylab

EPSILON = 1e-8

__all__ = [
    "bev3d_bbox_eval",
    "rotate_iou_matching",
    "rotate_iou",
    "inter",
    "rbbox_to_corners",
    "quadrilateral_intersection",
    "point_in_quadrilateral",
    "line_segment_intersection",
    "sort_vertex_in_convex_polygon",
    "trangle_area",
    "area",
    "bev3d_nms",
    "calap",
]


def collect_data(batch, output, gt_cids, det_cids, save_real3d_res=False):
    """Collect data for metric calculate in update stage.

    Args:
        batch (Dict): batch data of dataset.
        output (Dict): the model's output.
        gt_cids (List): eval_category_ids of ground truth.
        det_cids (List): eval_category_ids of pred's output.
        save_real3d_res (bool, optional): use to control the name of timestamp.
            will be remove future. Defaults to False.

    """

    assert "timestamp" in batch
    timestamps = np.array(batch["timestamp"].cpu())

    if save_real3d_res:
        timestamps = [str(int(_time)) for _time in timestamps]
        rec_date = batch["pack_dir"]
        timestamps = [
            date + "__" + time for date, time in zip(rec_date, timestamps)
        ]
    else:
        timestamps = [str(int(_time * 1000)) for _time in timestamps]
    _gt_group_by_cid = {cid: [] for cid in gt_cids}
    _det_group_by_cid = {cid: [] for cid in det_cids}
    _timestamps = []

    result_keys = [
        "bev3d_ct",
        "bev3d_loc_z",
        "bev3d_dim",
        "bev3d_rot",
        "bev3d_score",
        "bev3d_cls_id",
    ]
    results = {k: output[k] for k in result_keys}
    batch_size, num_objs = output[result_keys[0]].shape[:2]

    # process model's pred
    for bs in range(batch_size):
        for obj_idx in range(num_objs):
            pred_items = {
                key: val[bs][obj_idx].cpu().numpy()
                for key, val in results.items()
            }
            pred_items["timestamp"] = timestamps[bs]
            cid = pred_items["bev3d_cls_id"].tolist()
            if cid in _det_group_by_cid:
                _det_group_by_cid[cid] += [pred_items]

        # getting image timastamp
        _timestamps.append(timestamps[bs])

    # process ground truth
    assert (
        "annos_bev_3d" in batch
    ), "Please confirm the annos in batch, \
                check Bev3dTargetGenerator in auto3dv"
    annotations = batch["annos_bev_3d"]

    for cid in det_cids:
        for bs in range(batch_size):
            cur_cid_idxs = (
                (annotations["vcs_cls_"][bs] == cid).nonzero().squeeze(-1)
            )
            gt_cur_cid = {
                key: out[bs][cur_cid_idxs] for key, out in annotations.items()
            }
            num_objs_gt = gt_cur_cid[list(gt_cur_cid.keys())[0]].shape[0]
            for gt_obj_idx in range(num_objs_gt):
                gt_items = {
                    key: val[gt_obj_idx].cpu().numpy()
                    for key, val in gt_cur_cid.items()
                }
                gt_items["timestamp"] = timestamps[bs]
                _gt_group_by_cid[cid] += [gt_items]

    return _gt_group_by_cid, _det_group_by_cid, _timestamps


def bev3d_bbox_eval(
    det_res: Mapping,
    annotation: Mapping,
    dep_thresh: Optional[Sequence[str]],
    score_threshold: float,
    iou_threshold: float,
    gt_max_depth: float,
    enable_ignore: bool,
    vis_intervals: Optional[Sequence[str]],
) -> Mapping:
    """Eval the metric between GT and pred boxes of bev3d.

    Args:
        det_res (Dict): the predict 3d boxes info.
        annotation (Dict): the ground truth 3d boxes info.
        dep_thresh (tuple of int, default: None): Depth range to
            validation.
        score_threshold (float): Threshold for score.
        iou_threshold (float): Threshold for IoU.
        gt_max_depth (float): Max depth for gts.
        enable_ignore (bool): Whether to use ignore_mask.
        vis_intervals (tuple of str): Piecewise interval of visibility.
    Returns:
        (Dict): Dict contains the results.
    """

    timestamp_count = 0
    total_timestamp_count = len(annotation["timestamps"])

    all_dets = defaultdict(list)
    all_gts = defaultdict(list)
    for det in det_res:
        if det["bev3d_score"] < score_threshold:
            continue
        all_dets[det["timestamp"]].append(det)

    for gt in annotation["annotations"]:
        gt_depth = abs(gt["vcs_loc_"][0])  # vcs: abs(x)=depth
        if gt_depth > gt_max_depth:
            continue
        all_gts[gt["timestamp"]].append(gt)

    all_metric = [
        "dx",
        "dxp",
        "dy",
        "dyp",
        "dxy",
        "dxyp",
        "dw",
        "dwp",
        "dl",
        "dlp",
        "dh",
        "dhp",
        "drot",
    ]
    metrics = {}
    for k in all_metric:
        metrics[k] = {}
        for vi in vis_intervals:
            metrics[k][vi] = [[] for _ in range(len(dep_thresh) + 1)]

    gt_matched = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    gt_missed = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    redundant_det = np.zeros(len(dep_thresh) + 1)

    num_gt = 0
    det_tp_mask = []
    det_gt_mask = []
    det_tp_pred_loc = []
    all_scores = []

    for timestamp in annotation["timestamps"]:
        det_bbox3d, det_scores, det_locs, det_yaw = [], [], [], []
        gt_bbox3d, gt_locs, gt_yaw, gt_ignore, gt_visible = [], [], [], [], []

        # fetch 3d ground truth box
        for gt in all_gts[timestamp]:
            dim = gt["vcs_dim_"]
            yaw = gt["vcs_rot_z_"]
            loc = gt["vcs_loc_"]
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            gt_bbox3d.append(bbox3d)
            gt_locs.append(gt["vcs_loc_"])
            gt_yaw.append(gt["vcs_rot_z_"])
            gt_visible.append(gt["vcs_visible_"])
            if enable_ignore:
                gt_ignore.append(gt["vcs_ignore_"])

        for det in all_dets[timestamp]:
            dim = det["bev3d_dim"]
            yaw = det["bev3d_rot"]
            loc = det["bev3d_ct"]
            # [x, y, l, w, -yaw], -yaw means change the yaw from \
            # counterclockwise -> clockwise
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            det_bbox3d.append(bbox3d)
            assert det["bev3d_score"] >= 0
            det_scores.append(det["bev3d_score"])
            det_locs.append(det["bev3d_ct"])
            det_yaw.append(det["bev3d_rot"])

        det_locs = np.array(det_locs)
        gt_locs = np.array(gt_locs)
        gt_visible = np.array(gt_visible)
        if len(all_gts[timestamp]) == 0:
            if det_locs.any():
                pred_dep_thresh_inds = np.sum(
                    abs(det_locs[:, 0:1]) > dep_thresh, axis=-1
                )
                pred_dep_inds, pred_cnts = np.unique(
                    pred_dep_thresh_inds, return_counts=True
                )
                redundant_det[pred_dep_inds] += pred_cnts
            continue
        else:
            timestamp_count += 1

        if len(all_dets[timestamp]) == 0:
            for vis_idx, interval in enumerate(vis_intervals):
                vis_lthr, vis_rthr = [
                    float(_.translate({ord(i): None for i in "()"}))
                    for _ in interval.split(",")
                ]
                visib_gt_ind = (gt_visible >= vis_lthr) * (
                    gt_visible < vis_rthr
                )

                if enable_ignore:
                    valid_gt_ind = np.array(gt_ignore) == 0
                    gt_dep_thresh_inds = np.sum(
                        abs(gt_locs[visib_gt_ind * valid_gt_ind][:, 0:1])
                        > dep_thresh,
                        axis=-1,
                    )
                else:
                    gt_dep_thresh_inds = np.sum(
                        abs(gt_locs[visib_gt_ind][:, 0:1]) > dep_thresh,
                        axis=-1,
                    )

                gt_dep_inds, gt_cnts = np.unique(
                    gt_dep_thresh_inds, return_counts=True
                )
                gt_missed[vis_idx][gt_dep_inds] += gt_cnts
            continue

        det_bbox3d = np.array(det_bbox3d)
        det_scores = np.array(det_scores)
        gt_bbox3d = np.array(gt_bbox3d)

        assert det_bbox3d.shape[0] == det_scores.shape[0]

        (matched_dict, redundant, det_ignored_mask) = rotate_iou_matching(
            det_bbox3d,
            det_locs,
            gt_bbox3d,
            gt_locs,
            det_scores,
            iou_threshold,
            gt_ignore,
        )
        redundant_det_mask = np.zeros(det_locs.shape[0], dtype=bool)
        for ind in redundant:
            redundant_det_mask[ind] = 1
        redundant_det_dep = np.sum(
            abs(det_locs[redundant_det_mask, 0:1]) > dep_thresh, axis=-1
        )
        redundant_det_dep_inds, redundant_det_dep_cnts = np.unique(
            redundant_det_dep, return_counts=True
        )
        redundant_det[redundant_det_dep_inds] += redundant_det_dep_cnts
        all_scores += det_scores[np.invert(det_ignored_mask)].tolist()
        tp = np.ones(len(det_scores), dtype=bool)
        tp[redundant] = 0
        tp = tp[np.invert(det_ignored_mask)]
        tp_det_loc = det_locs.copy()
        tp_det_loc = tp_det_loc[np.invert(det_ignored_mask)]
        gt_det_loc = gt_locs.copy()
        if enable_ignore:
            gt_det_loc = gt_det_loc[np.invert(gt_ignore)]
        num_gt += len(gt_bbox3d) - sum(gt_ignore)

        det_tp_mask += tp.tolist()
        det_gt_mask += gt_det_loc.tolist()
        det_tp_pred_loc += tp_det_loc.tolist()
        det_assigns = matched_dict["det_assign"]
        inds = np.array(list(range(len(det_assigns))))

        mask = det_assigns != -1
        det_assigns, inds = det_assigns[mask], inds[mask]
        for vis_idx, interval in enumerate(vis_intervals):
            vis_lthr, vis_rthr = [
                float(_.translate({ord(i): None for i in "()"}))
                for _ in interval.split(",")
            ]
            gt_visible_mask = (gt_visible >= vis_lthr) * (
                gt_visible < vis_rthr
            )

            if enable_ignore:
                gt_missed_mask = np.invert(mask) * np.invert(
                    np.array(gt_ignore) == 1
                )
            else:
                gt_missed_mask = np.invert(mask)
            gt_missed_dep_thresh_inds = np.sum(
                abs(gt_locs[gt_visible_mask * gt_missed_mask, 0:1])
                > dep_thresh,
                axis=-1,
            )
            gt_missed_dep_inds, gt_missed_cnts = np.unique(
                gt_missed_dep_thresh_inds, return_counts=True
            )
            gt_missed[vis_idx][gt_missed_dep_inds] += gt_missed_cnts

            if np.sum(mask) == 0:
                continue

            pred_dim = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_dim"]
                    for assign in det_assigns
                ]
            )
            pred_loc = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_ct"]
                    for assign in det_assigns
                ]
            )
            pred_yaw_rad = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_rot"]
                    for assign in det_assigns
                ]
            )
            pred_yaw = np.rad2deg(pred_yaw_rad) % 360.0

            gt_dim = np.array(
                [all_gts[timestamp][ind]["vcs_dim_"] for ind in inds]
            )
            gt_loc = np.array(
                [all_gts[timestamp][ind]["vcs_loc_"] for ind in inds]
            )
            gt_depth = np.array(
                [all_gts[timestamp][ind]["vcs_loc_"][0] for ind in inds]
            )
            gt_yaw_rad = np.array(
                [all_gts[timestamp][ind]["vcs_rot_z_"] for ind in inds]
            )
            gt_vis = np.array(
                [all_gts[timestamp][ind]["vcs_visible_"] for ind in inds]
            )
            gt_yaw = np.rad2deg(gt_yaw_rad) % 360.0
            gt_vis_mask = (gt_vis >= vis_lthr) * (gt_vis < vis_rthr)

            dep_thresh_inds = np.sum(
                abs(gt_depth[gt_vis_mask][:, np.newaxis]) > dep_thresh, axis=-1
            )
            dep_inds, cnts = np.unique(dep_thresh_inds, return_counts=True)
            gt_matched[vis_idx][dep_inds.tolist()] += cnts

            dx = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
            dy = np.abs(pred_loc[:, 1] - gt_loc[:, 1])
            dxy = (dx ** 2 + dy ** 2) ** 0.5
            dw = np.abs(pred_dim[:, 1] - gt_dim[:, 1])
            dl = np.abs(pred_dim[:, 2] - gt_dim[:, 2])
            dh = np.abs(pred_dim[:, 0] - gt_dim[:, 0])

            dxp = dx / np.abs(gt_loc[:, 0])
            dyp = dy / np.abs(gt_loc[:, 1])
            dxyp = dxy / np.abs(gt_loc[:, 0] ** 2 + gt_loc[:, 1] ** 2) ** 0.5
            dxy_10p_error = dxyp <= 0.1
            dwp = dw / np.abs(gt_dim[:, 1])
            dlp = dl / np.abs(gt_dim[:, 2])
            dhp = dh / np.abs(gt_dim[:, 0])
            abs_rot = np.abs(gt_yaw - pred_yaw)
            drot = np.minimum(abs_rot, 360.0 - abs_rot)

            res = {
                "dx": dx,
                "dy": dy,
                "dxy": dxy,
                "dw": dw,
                "dl": dl,
                "dh": dh,
                "dxp": dxp,
                "dyp": dyp,
                "dxyp": dxyp,
                "dwp": dwp,
                "dlp": dlp,
                "dhp": dhp,
                "drot": drot,
                "dxy_10p_error": dxy_10p_error,
            }
            dep_thresh_inds = np.sum(
                abs(gt_depth[:, np.newaxis]) > dep_thresh, axis=-1
            )
            for key, val in metrics.items():
                for dep_ind in dep_inds:
                    dep_mask = dep_thresh_inds == dep_ind
                    val[interval][dep_ind] += res[key][
                        dep_mask * gt_vis_mask
                    ].tolist()

    result_aps = {
        "det_tp_mask": det_tp_mask,
        "det_gt_mask": det_gt_mask,
        "det_tp_pred_loc": det_tp_pred_loc,
        "all_scores": all_scores,
        "num_gt": float(num_gt),
    }
    metrics["counts"] = {
        "gt_matched": gt_matched,
        "gt_missed": gt_missed,
        "redundant_det": redundant_det,
        "timestamp_count": timestamp_count,
        "total_timestamp_count": total_timestamp_count,
        "result_aps": result_aps,
    }
    return metrics


def calap(recall, prec):
    """Calculate ap metric.

    Args:
        recall (np.ndarray): recalls for ap.
        prec (np.ndarray): precisions for ap.

    Returns:
        float: ap metric.
    """
    mrec = [0] + list(recall.flatten()) + [1]
    mpre = [0] + list(prec.flatten()) + [0]
    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0
    for i in range(len(mpre) - 1):
        if mpre[i + 1] > 0:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def draw_pr_curves(eval_files, eval_names):
    """Draw compare p-r curves of multi result file.

    Args:
        eval_files: array-like, a list of file objects
        eval_names: array-like, a list of eval names

    Returns
        b_io: buffer, buffer object contains a image
    """

    fig, ax1 = pylab.subplots(1, 1, figsize=(7, 7))
    line_color = {
        "0": "b",
        "1": "g",
        "2": "r",
    }
    for eval_file, fn in zip(eval_files, eval_names):  # noqa
        results = json.load(eval_file)
        for cid, res in results.items():
            recall = res["recall"]
            precision = res["precision"]
            ax1.plot(
                recall,
                precision,
                line_color[cid],
                label="num_gt:{}, class:{}".format(
                    results["0"]["num_gt"], cid
                ),
            )

    ax1.grid()
    ax1.legend(loc="lower left", borderaxespad=0.0, fontsize="xx-small")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xticks(np.arange(0.0, 1, 0.05))
    ax1.set_yticks(np.arange(0.0, 1, 0.05))
    ax1.set_title("recall vs precision")
    for ax in fig.axes:
        matplotlib.pyplot.sca(ax)
        pylab.xticks(rotation=45)

    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    pylab.close(fig)
    return b_io


def draw_curves(eval_files, eval_names, output_file):
    """Draw compare curves of multi result file.

    Args:
        eval_files: array-like, a list of file objects
        eval_names: array-like, a list of eval names
        output_file: array-like, a list of output file
    """
    b_io = draw_pr_curves(
        [open(eval_file, "r") for eval_file in eval_files], eval_names
    )
    with open(output_file, "wb") as f:
        f.write(b_io.read())


def rotate_iou_matching(
    det_boxes: np.ndarray,
    det_locs: np.ndarray,
    gt_boxes: np.ndarray,
    gt_locs: np.ndarray,
    det_scores: np.ndarray,
    iou_threshold=0.2,
    gt_ignore_mask=None,
) -> Mapping:
    """Calculate the iou between GT and pred rotation boxes on vcs.

    Args:
        det_boxes (np.ndarray): the predict boxes, shape:[N, 5].
        det_locs (np.ndarray): the predict location, shape:[N, 2].
        gt_boxes (np.ndarray): the GT boxes, shape: [K,5].
        gt_locs (np.ndarray): the gt location, shape: [K,3].
        det_scores (np.ndarray): the pred score, shape: [N,].
        iou_threshold (float, optional): Defaults to 0.2.
        gt_ignore_mask (Sequence, optional): the gt igonre list.

    Returns:
        Mapping: results based on iou matching.
    """
    det_ct, gt_ct = det_locs, gt_locs
    assert len(det_boxes) == len(det_scores)
    overlaps = np.zeros(shape=gt_ct.shape[0])
    det_assign = np.zeros(shape=gt_ct.shape[0], dtype=np.int) - 1
    matched_det = np.zeros(shape=det_ct.shape[0], dtype=np.int)

    valid_gt = np.ones(gt_ct.shape[0], dtype=bool)
    pairwise_iou = rotate_iou(det_boxes, gt_boxes)
    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_ct.shape[0], dtype=bool)

    for dt_idx in det_sorted_idx:
        dxy = (
            np.sum((det_ct[dt_idx, [0, 1]] - gt_ct[:, [0, 1]]) ** 2, axis=1)
            ** 0.5
        )
        gt_dist = np.sum(gt_ct[:, [0, 1]] ** 2, axis=1) ** 0.5
        dist_err = dxy / gt_dist
        gt_mask = ((gt_dist <= 50) | (dist_err <= 0.2)) & valid_gt
        det_iou = pairwise_iou[dt_idx] * gt_mask.astype(float)
        max_ind = np.argmax(det_iou)
        max_iou = det_iou[max_ind]
        if max_iou > iou_threshold:
            if gt_ignore_mask and gt_ignore_mask[max_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[max_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[max_ind] = max_iou
                valid_gt[max_ind] = 0
    redundant = np.where(matched_det == 0)[0]
    return (
        {"overlaps": overlaps, "det_assign": det_assign},
        redundant,
        det_ignored_mask,
    )


def rotate_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: int = -1,
) -> np.ndarray:
    """Compute rotated ious between GT and pred bboxes.

    The iou func is mainly modify by the mmdetection3d:
    https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/kitti_utils/rotate_iou.py, # noqa
    which removing the numba and cuda component, only using numpy.
    For more detail, please refer to this official link, if want to
    check the rotate_iou by your own sample, please using the unit_tests
    function: test_metric_3dv.py.

    another Rotate_iou implementation: https://github.com/lilanxiao/Rotated_IoU/blob/master/utiles.py # noqa

    Args:
        pred_boxes (np.ndarray): Predict rotated 2d Boxes,
            shape:[num_pred, 5], num_pred is the number of boxes.
        gt_boxes (np.ndarray): GT rotated 2d boxes. shape:[num_gt, 5],
            num_gt is the number of boxes.
        criterion (int, optional): Indicate different type of iou.
            -1 indicate `area_inter / (area1 + area2 - area_inter)`,
            0 indicate `area_inter / area1`,
            1 indicate `area_inter / area2`.

    Returns:
        np.ndarray: ious between pred and gt boxes.
    """
    num_pred = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    rotate_ious = np.zeros((num_pred, num_gt), dtype=np.float64)
    for idx_pred in range(num_pred):
        for idx_gt in range(num_gt):

            area_pred = pred_boxes[idx_pred][2] * pred_boxes[idx_pred][3]
            area_gt = gt_boxes[idx_gt][2] * gt_boxes[idx_gt][3]
            area_inter = inter(pred_boxes[idx_pred], gt_boxes[idx_gt])
            if criterion == -1:
                tmp_iou = area_inter / (area_pred + area_gt - area_inter)
            elif criterion == 0:
                tmp_iou = area_inter / area_pred
            elif criterion == 1:
                tmp_iou = area_inter / area_gt
            else:
                tmp_iou = area_inter
            rotate_ious[idx_pred][idx_gt] = tmp_iou

    return rotate_ious


def inter(rbbox1, rbbox2) -> float:
    """Compute intersection of two rotated boxes.

    Args:
        rbox1 (np.ndarray, shape=[5]): Rotated 2d box.
        rbox2 (np.ndarray, shape=[5]): Rotated 2d box.
    Returns:
        float: Intersection of two rotated boxes.
    """
    corners1 = np.zeros((8,), dtype=np.float64)
    corners2 = np.zeros((8,), dtype=np.float64)
    intersection_corners = np.zeros((16,), dtype=np.float64)
    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)
    num_intersection = quadrilateral_intersection(
        corners1, corners2, intersection_corners
    )
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


def rbbox_to_corners(corners, rbbox):
    """Generate clockwise corners and rotate it clockwise.

    Args:
        corners (np.ndarray, shape=[8]): the 4 corner point of
            rotate bbox, [x0, y0, x1, y1, x2, y2, x3, y3].
        rbbox (np.ndarray, shape=[5]): the rotated bbox's
            info (x, y, l, w, yaw).
    """

    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.zeros((4,), dtype=np.float64)
    corners_y = np.zeros((4,), dtype=np.float64)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = (
            -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y
        )


def quadrilateral_intersection(pts1, pts2, int_pts) -> int:
    """Find intersection points pf two boxes.

    Args:
        pts1 (np.ndarray, shape=[8]): the 4 points of box1.
        pts2 (np.ndarray, shape=[8]): the 4 points of box2.
        int_pts (np.ndarray, shape=[16]): the intersected
            point between box1 and box2.

    Returns:
        int: the number of intersection points.
    """

    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.zeros((2,), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1
    return num_of_inter


def point_in_quadrilateral(pt_x, pt_y, corners) -> bool:
    """Check whether a point lies in a rectangle defined by corners.

    Args:
        pt_x (float): point's x coordinate value
        pt_y (float): point's y coordinate value
        corners (np.ndarray, shape=[8]): 4 points of a rectangle.

    Returns:
        bool: the point in rectangle or not.
    """

    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]
    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]
    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]
    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1
    return (
        abab + EPSILON >= abap
        and abap + EPSILON >= 0
        and adad + EPSILON >= adap
        and adap + EPSILON >= 0
    )


def line_segment_intersection(pts1, pts2, i, j, temp_pts) -> bool:
    """Find intersection of 2 lines defined by their end points.

    Args:
        pts1 (np.ndarray, shape=[8]): the 4 point of rotate_bbox1.
        pts2 (np.ndarray, shape=[8]): the 4 point of rotate_bbox2.
        i (int): the index of pts1, from [0,3].
        j (int): the index of pts2, from [0,3].
        temp_pts (np.ndarray, shape=[2]): the tmp intersection point.

    Returns:
        bool: whether the two line have the intersected point.
    """

    point_A = np.zeros((2,), dtype=np.float64)
    point_B = np.zeros((2,), dtype=np.float64)
    point_C = np.zeros((2,), dtype=np.float64)
    point_D = np.zeros((2,), dtype=np.float64)
    point_A[0] = pts1[2 * i]
    point_A[1] = pts1[2 * i + 1]
    point_B[0] = pts1[2 * ((i + 1) % 4)]
    point_B[1] = pts1[2 * ((i + 1) % 4) + 1]
    point_C[0] = pts2[2 * j]
    point_C[1] = pts2[2 * j + 1]
    point_D[0] = pts2[2 * ((j + 1) % 4)]
    point_D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = point_B[0] - point_A[0]
    BA1 = point_B[1] - point_A[1]
    DA0 = point_D[0] - point_A[0]
    CA0 = point_C[0] - point_A[0]
    DA1 = point_D[1] - point_A[1]
    CA1 = point_C[1] - point_A[1]

    acd = DA1 * CA0 > CA1 * DA0
    bcd = (point_D[1] - point_B[1]) * (point_C[0] - point_B[0]) > (
        point_C[1] - point_B[1]
    ) * (point_D[0] - point_B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = point_D[0] - point_C[0]
            DC1 = point_D[1] - point_C[1]
            ABBA = point_A[0] * point_B[1] - point_B[0] * point_A[1]
            CDDC = point_C[0] * point_D[1] - point_D[0] * point_C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    """Sort convex_polygon's vertices in clockwise order.

    Args:
        int_pts (np.ndarray, shape=[16]): the intersection points.
        num_of_inter (int): number of intersection points.
    """

    if num_of_inter > 0:
        center = np.zeros((2,), dtype=np.float64)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.zeros((2,), dtype=np.float64)
        vs = np.zeros((16,), dtype=np.float64)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d + EPSILON
            v[1] = v[1] / d + EPSILON
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


def trangle_area(point_a, point_b, point_c) -> float:
    """Calculate the trangle area.

    Args:
        point_a (np.ndarray, shape=[2]): point in trangle.
        point_b (np.ndarray, shape=[2]): point in trangle.
        point_c (np.ndarray, shape=[2]): point in trangle.

    Returns:
        float: trangle_area.
    """
    return (
        (point_a[0] - point_c[0]) * (point_b[1] - point_c[1])
        - (point_a[1] - point_c[1]) * (point_b[0] - point_c[0])
    ) / 2.0


def area(int_pts, num_of_inter) -> float:
    """Calculate the intersection area.

    Args:
        int_pts (np.ndarray, shape=[16]): the intersection points.
        num_of_inter (int): number of intersection points.

    Returns:
        float: the return intersection area.
    """

    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(
                int_pts[:2],
                int_pts[2 * i + 2 : 2 * i + 4],
                int_pts[2 * i + 4 : 2 * i + 6],
            )
        )
    return area_val


def bev3d_nms(boxes, scores, thresh):
    """Nms function for bev3d on vcs plane.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 7]
            ([x, y, z, h, w, l, rot_z]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Threshold.
    Returns:
        np.array: Indexes to keep after nms.
    """
    assert len(boxes) == len(scores), "len(scores) must be equal to len(boxes)"
    boxes = boxes.cpu().numpy()
    # change the boxes [N,7] -> [N,5] for rotate_2d iou calculation
    det_bbox3d = []
    for box in boxes:
        loc = box[:3]
        dim = box[3:6]
        yaw = box[-1]
        bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
        det_bbox3d.append(bbox3d)
    det_bbox3d = np.array(det_bbox3d)

    scores = scores.cpu().numpy()
    order = np.argsort(scores)
    keep = np.zeros(det_bbox3d.shape[0])

    # picked_boxes = []
    while order.size > 0:
        index = order[-1]
        keep[index] = 1
        # picked_boxes.append(det_bbox3d[index])
        current_box = np.expand_dims(det_bbox3d[index], axis=0)
        left_boxes = det_bbox3d[order[:-1]]
        rotate_3d_ious = rotate_iou(current_box, left_boxes)
        rotate_3d_ious = np.squeeze(rotate_3d_ious)
        left = np.where(rotate_3d_ious < thresh)
        order = order[left]
    return keep.astype("bool")


class NpEncoder(json.JSONEncoder):
    """Json encoder for numpy array."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
