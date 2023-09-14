# Copyright (c) Changan Auto. All rights reserved.

import logging
import os
import pickle
from collections import defaultdict
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from cap.metrics.metric import EvalMetric
from cap.metrics.metric_3dv_utils import rotate_iou_matching
from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info

__all__ = ["BEVDiscreteObjectEval"]

logger = logging.getLogger(__name__)


def eval_metric(
    det_res: Mapping,
    annotation: Mapping,
    dep_thresh: Optional[Sequence[str]],
    score_threshold: float,
    iou_threshold: float,
    gt_max_depth: float,
) -> Mapping:
    """Eval the metric between GT and pred boxes.

    Based on bev3d_bbox_eval(). For more details please refer to
    (cap/metrics/metric_3dv_utils.py)

    Args:
        det_res (Dict): The predict discrete obj boxes info.
        annotation (Dict): The ground truth discrete obj boxes info.
        dep_thresh (tuple of int, default: None): Depth range to
            validation.
        score_threshold (float, defualt: 0.1): Threshold for score.
        iou_threshold (float, defualt: 0.2): Threshold for IoU.
        gt_max_depth (float, default: 300): Max depth for gts.

    Returns:
        (Dict): Dict contains the results.
    """
    timestamp_count, gt_missed, redundant_det = 0, 0, 0

    all_dets = defaultdict(list)
    all_gts = defaultdict(list)
    for det in det_res:
        if det["pred_score"] < score_threshold:
            continue
        all_dets[det["timestamp"]].append(det)

    for gt in annotation["annotations"]:
        gt_depth = abs(gt["vcs_discobj_loc"][0])  # vcs: abs(x) = depth
        if gt_depth > gt_max_depth:
            continue
        all_gts[gt["timestamp"]].append(gt)

    metrics = [
        "dx",
        "dxp",
        "dy",
        "dyp",
        "dxy",
        "dxyp",
        "dw",
        "dwp",
        "dh",
        "dhp",
        "drot",
    ]

    metrics = {k: [[] for _ in range(len(dep_thresh) + 1)] for k in metrics}
    gt_matched = np.zeros(len(dep_thresh) + 1)

    num_gt = 0
    det_tp_mask = []
    all_scores = []
    for timestamp in annotation["timestamps"]:
        det_bbox3d, det_scores, det_locs, det_yaws = [], [], [], []  # noqa
        gt_bbox3d, gt_locs, gt_yaws = [], [], []  # noqa

        if len(all_gts[timestamp]) == 0:
            redundant_det += len(all_dets[timestamp])
            continue

        for gt in all_gts[timestamp]:
            dim = gt["vcs_discobj_wh"]
            yaw = gt["vcs_discobj_yaw"]
            loc = gt["vcs_discobj_loc"]
            # [x, y, w, h, -yaw], -yaw means change the yaw from \
            # counterclockwise -> clockwise
            bbox3d = [loc[0], loc[1], dim[0], dim[1], -yaw]
            gt_bbox3d.append(bbox3d)
            gt_locs.append(loc)

        if len(all_dets[timestamp]) == 0:
            gt_missed += len(gt_bbox3d)
            num_gt += len(gt_bbox3d)
            continue

        for det in all_dets[timestamp]:
            dim = det["pred_wh"]
            yaw = det["pred_yaw"]
            loc = det["pred_loc"]
            bbox3d = [loc[0], loc[1], dim[0], dim[1], -yaw]
            det_bbox3d.append(bbox3d)
            assert det["pred_score"] > 0
            det_scores.append(det["pred_score"])
            det_locs.append(det["pred_loc"])

        det_bbox3d = np.array(det_bbox3d)
        det_scores = np.array(det_scores)
        gt_bbox3d = np.array(gt_bbox3d)
        det_locs = np.array(det_locs)
        gt_locs = np.array(gt_locs)

        assert det_bbox3d.shape[0] == det_scores.shape[0]

        (matched_dict, redundant, det_ignored_mask) = rotate_iou_matching(
            det_bbox3d,
            det_locs,
            gt_bbox3d,
            gt_locs,
            det_scores,
            iou_threshold,
        )

        redundant_det += len(redundant)
        all_scores += det_scores[np.invert(det_ignored_mask)].tolist()

        tp = np.ones(len(det_scores), dtype=bool)
        tp[redundant] = 0
        tp = tp[np.invert(det_ignored_mask)]

        det_tp_mask += tp.tolist()
        det_assigns = matched_dict["det_assign"]
        inds = np.array(list(range(len(det_assigns))))

        mask = det_assigns != -1
        det_assigns, inds = det_assigns[mask], inds[mask]
        gt_missed += np.sum(np.invert(mask))
        if np.sum(mask) == 0:
            continue

        pred_dim = np.array(
            [all_dets[timestamp][assign]["pred_wh"] for assign in det_assigns]
        )
        pred_loc = np.array(
            [all_dets[timestamp][assign]["pred_loc"] for assign in det_assigns]
        )
        pred_yaw_rad = np.array(
            [all_dets[timestamp][assign]["pred_yaw"] for assign in det_assigns]
        )
        pred_yaw = np.rad2deg(pred_yaw_rad)

        gt_dim = np.array(
            [all_gts[timestamp][ind]["vcs_discobj_wh"] for ind in inds]
        )
        gt_loc = np.array(
            [all_gts[timestamp][ind]["vcs_discobj_loc"] for ind in inds]
        )
        gt_depth = np.array(
            [all_gts[timestamp][ind]["vcs_discobj_loc"][0] for ind in inds]
        )
        gt_yaw_rad = np.array(
            [all_gts[timestamp][ind]["vcs_discobj_yaw"] for ind in inds]
        )
        gt_yaw = np.rad2deg(gt_yaw_rad)

        dx = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
        dy = np.abs(pred_loc[:, 1] - gt_loc[:, 1])
        dxy = (dx ** 2 + dy ** 2) ** 0.5
        dw = np.abs(pred_dim[:, 0] - gt_dim[:, 0])
        dh = np.abs(pred_dim[:, 1] - gt_dim[:, 1])

        dxp = dx / np.abs(gt_loc[:, 0])
        dyp = dy / np.abs(gt_loc[:, 1])
        dxyp = dxy / np.abs(gt_loc[:, 0] ** 2 + gt_loc[:, 1] ** 2) ** 0.5
        dxy_10p_error = dxyp <= 0.1
        dwp = dw / np.abs(gt_dim[:, 0])
        dhp = dw / np.abs(gt_dim[:, 1])
        abs_rot = np.abs(gt_yaw - pred_yaw)
        drot = np.minimum(abs_rot, 180.0 - abs_rot)

        res = {
            "dx": dx,
            "dy": dy,
            "dxy": dxy,
            "dw": dw,
            "dh": dh,
            "dxp": dxp,
            "dyp": dyp,
            "dxyp": dxyp,
            "dwp": dwp,
            "dhp": dhp,
            "drot": drot,
            "dxy_10p_error": dxy_10p_error,
        }

        dep_thresh_inds = np.sum(
            abs(gt_depth[:, np.newaxis]) > dep_thresh, axis=-1
        )
        dep_inds, cnts = np.unique(dep_thresh_inds, return_counts=True)
        gt_matched[dep_inds.tolist()] += cnts

        for key, val in metrics.items():
            for dep_ind in dep_inds:
                mask = dep_thresh_inds == dep_ind
                val[dep_ind] += res[key][mask].tolist()

        timestamp_count += 1

    metrics["tp"] = gt_matched
    metrics["fn"] = gt_missed
    metrics["fp"] = redundant_det
    metrics["images"] = timestamp_count
    return metrics


@OBJECT_REGISTRY.register
class BEVDiscreteObjectEval(EvalMetric):
    """The BEV discrete object detection eval metrics.

    The BEV discrete object detection metric calculation is based on real3d
    eval metrics. For more details please refer to Real3DEval
    (cap/metrics/real3d.py).

    Args:
        name (str): Name of this metric instance for display
        eval_category_ids (tuple of int): The categories to be evaluation:
        score_threshold (float, defualt: 0.1): Threshold for score.
        iou_threshold (float, defualt: 0.2): Threshold for IoU.
        gt_max_depth (float, default: 100): Max depth for gts.
        metrics (tuple of str, default: None): Tuple of eval metrics, using
            ("dxy", "dw", "dh", "drot") if not special.
        depth_intervals (tuple of int, default: None): Depth range to
            validation, using (20, 50, 70) if not special.
        save_dir (str, default: None): Path to save the predictions.
    """

    def __init__(
        self,
        name: str,
        eval_category_ids: Sequence[int],
        score_threshold=0.1,
        iou_threshold=0.2,
        gt_max_depth=100,
        metrics: Optional[Sequence[str]] = ("dxy", "dw", "dh", "drot"),
        depth_intervals: Optional[Sequence[str]] = (20, 50, 70),
        save_dir=None,
    ):
        self.eval_category_ids = eval_category_ids
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.gt_max_depth = gt_max_depth
        self.metrics = metrics
        self.depth_intervals = depth_intervals

        assert (
            self.gt_max_depth > depth_intervals[-1]
        ), "gt_max_depth must be greater than depth_intervals[-1]"

        self.eps = 1e-9
        depth_intervals = [0] + list(depth_intervals) + [self.gt_max_depth]
        self.full_depth_intervals = [
            f"({start},{end})"
            for start, end in zip(depth_intervals[:-1], depth_intervals[1:])
        ]

        self.metric_index = 0
        self.save_dir = save_dir
        self.show_num_metric = ["fn", "fp", "images"]
        super(BEVDiscreteObjectEval, self).__init__(name)

    def _init_states(self):
        for name in self.get_names():
            self.add_state(
                name,
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )

    def save_result(self, batch, output):
        with open(self.det_save_path, "ab") as f:
            pickle.dump([batch, output], f)

    def reset(self):
        super().reset()
        if self.save_dir:
            rank, _ = get_dist_info()
            self.det_save_path = os.path.join(
                self.save_dir, str(self.metric_index), f"dets/{rank}.pkl"
            )
            pkl_save_root = os.path.dirname(self.det_save_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)

    def get(self):
        """Get evaluation metrics."""
        format_metrics = self.compute()
        names = format_metrics[0]
        values = [[] for _ in range(len(names))]
        for value in format_metrics[1:]:
            for i, v in enumerate(value):
                values[i].append(v)
        return names, values

    def update(
        self,
        timestamp,
        annos_bev_discobj,
        pred_ct,
        pred_wh,
        pred_score,
        pred_yaw,
        pred_bev_discobj_cls_id,
    ):
        """Update.

        Args:
            timestamp (torch.Tensor): Input image timestamp, (batch_size,).
            annos_bev_discobj (dict): A dict contains GT bbox info,
                "vcs_discobj_loc": GT bbox center coord,
                    (batch_size, max_num,2).
                "vcs_discobj_wh": GT bbox width and height,
                    (batch_size, max_num,2).
                "vcs_discobj_valid": Valid GT bbox, (batch_size, max_num).
                "vcs_discobj_cls": GT bbox category, (batch_size, max_num).
                "vcs_discobj_yaw": GT bbox yaw, (batch_size, max_num).
            pred_ct (torch.Tensor): Predict bbox center, (batch_size, topk, 2).
            pred_wh (torch.Tensor): Predict bbox width and height,
                (batch_size, topk, 2).
            pred_score (torch.Tensor): Predict bbox score, (batch_size, topk).
            pred_yaw (torch.Tensor): Predict bbox yaw, (batch_size, topk).
            pred_bev_discobj_cls_id (torch.Tensor): Predict bbox category,
                (batch_size, topk).
        """
        batch = {
            "timestamp": timestamp,
            "annos_bev_discobj": annos_bev_discobj,
        }
        output = {
            "pred_loc": pred_ct,
            "pred_wh": pred_wh,
            "pred_score": pred_score,
            "pred_yaw": pred_yaw,
            "pred_bev_discobj_cls_id": pred_bev_discobj_cls_id,
        }

        if self.save_dir:
            self.save_result(batch, output)

        batch_timestamps = batch["timestamp"].cpu().numpy()
        batch_timestamps = [str(int(_ts * 1000)) for _ts in batch_timestamps]
        eval_timestamps = batch_timestamps
        det_objs_by_cid = {cid: [] for cid in self.eval_category_ids}
        gt_objs_by_cid = {cid: [] for cid in self.eval_category_ids}

        # process model's output
        assert "pred_wh" in output.keys()
        batch_size, num_objs = output["pred_wh"].shape[:2]
        for bs in range(batch_size):
            for obj_idx in range(num_objs):
                pred_items = {
                    key: val[bs][obj_idx].cpu().numpy()
                    for key, val in output.items()
                }
                pred_items["timestamp"] = batch_timestamps[bs]
                cid = pred_items["pred_bev_discobj_cls_id"].tolist()
                if cid in det_objs_by_cid:
                    det_objs_by_cid[cid] += [pred_items]

        # process ground truth
        assert (
            "annos_bev_discobj" in batch
        ), "Please confirm the annos in batch, \
                check BevDiscreteObjectTargetGenerator in auto3dv"
        annotations = batch["annos_bev_discobj"]  # in vcs

        for cid in self.eval_category_ids:
            for bs in range(batch_size):
                cur_cid_idxs = (
                    (annotations["vcs_discobj_cls"][bs] == cid)
                    .nonzero()
                    .squeeze(-1)
                )
                gt_cur_cid = {
                    key: val[bs][cur_cid_idxs]
                    for key, val in annotations.items()
                }
                num_objs_gt = gt_cur_cid[list(gt_cur_cid.keys())[0]].shape[0]
                for gt_obj_idx in range(num_objs_gt):
                    gt_items = {
                        key: val[gt_obj_idx].cpu().numpy()
                        for key, val in gt_cur_cid.items()
                    }
                    gt_items["timestamp"] = batch_timestamps[bs]
                    gt_objs_by_cid[cid] += [gt_items]

        for cid in self.eval_category_ids:
            gt = {
                "timestamps": eval_timestamps,
                "annotations": gt_objs_by_cid[cid],
            }
            det = det_objs_by_cid[cid]
            res = eval_metric(
                det,
                gt,
                self.depth_intervals,
                self.score_threshold,
                self.iou_threshold,
                self.gt_max_depth,
            )

            for i_dep, dep_interval in enumerate(self.full_depth_intervals):
                name = f"{cid}_{dep_interval}_tp"
                val = getattr(self, name, 0.0) + res["tp"][i_dep]
                setattr(self, name, val)
                for metric_name in self.metrics:
                    name = f"{cid}_{dep_interval}_{metric_name}"
                    val = getattr(self, name, 0.0) + np.sum(
                        res[metric_name][i_dep]
                    )
                    setattr(self, name, val)
            for num in self.show_num_metric:
                name = f"{cid}_{num}"
                val = getattr(self, name, 0.0) + np.sum(res[num])
                setattr(self, name, val)

    def get_names(self):
        names = []
        for cate in self.eval_category_ids:
            for dep_interval in self.full_depth_intervals:
                for metric in self.metrics:
                    names.append(f"{cate}_{dep_interval}_{metric}")
                names.append(f"{cate}_{dep_interval}_tp")
            for num in self.show_num_metric:
                names.append(f"{cate}_{num}")
        return names

    def print_log(self, format_metrics):
        format_metrics_show = []
        head_names = ["Category"]
        log_line = "\n BEV_disc_obj Metircs:\n"
        log_line += "".ljust(10, "*")
        for dep_interval in self.full_depth_intervals:
            log_line += dep_interval.ljust(40, "*")
        log_line += "ALL".ljust(60, "*")
        log_line += "\n"
        log_line += "Category".ljust(10)
        for dep_interval in self.full_depth_intervals:
            for metric_name in self.metrics:
                head_name = f"{dep_interval}_{metric_name}"
                log_line += metric_name.ljust(8)
                head_names.append(head_name)
            log_line += "tp".ljust(8)
        counts_show = ["tp", "fn", "fp", "Recall", "Precision", "vImages"]
        for m_num in counts_show:
            log_line += m_num.ljust(10)
            head_names.append(m_num)
        format_metrics_show.append(head_names)
        log_line += "\n"

        for cate, cate_metrics in format_metrics.items():
            tp = 0
            kpi_values = [cate]
            log_line += str(cate).ljust(10)
            for name, vals in cate_metrics.items():
                if name in self.show_num_metric:
                    continue
                tp_ = vals.get("tp", 0)
                tp += tp_
                for metric_name in self.metrics:
                    val = round(vals[metric_name] / (tp_ + self.eps), 3)
                    log_line += str(val).ljust(8)
                    kpi_values.append(val)
                log_line += str(int(tp_)).ljust(8)

            tp = int(tp)
            fn = int(cate_metrics["fn"])
            fp = int(cate_metrics["fp"])
            valid_images = int(cate_metrics["images"])

            recall = round(tp / (tp + fn + self.eps), 3)
            presicion = round(tp / (tp + fp + self.eps), 3)
            counts_num = [tp, fn, fp, recall, presicion, valid_images]
            log_line += "".join([str(m).ljust(10) for m in counts_num])
            kpi_values.extend(counts_num)
            format_metrics_show.append(kpi_values)
            log_line += "\n"
        logger.info(log_line)
        return format_metrics_show

    def compute(self):
        self.metric_index += 1
        format_metrics = {}
        for cate in self.eval_category_ids:
            cate = str(cate)
            format_metrics[cate] = {}
            for dep_interval in self.full_depth_intervals:
                format_metrics[cate][dep_interval] = {}
                tp_num = f"{cate}_{dep_interval}_tp"
                val = getattr(self, tp_num, None).cpu().numpy()
                format_metrics[cate][dep_interval]["tp"] = val
                for metric in self.metrics:
                    name = f"{cate}_{dep_interval}_{metric}"
                    val = getattr(self, name, None).cpu().numpy()
                    format_metrics[cate][dep_interval][metric] = val
            for num in self.show_num_metric:
                format_metrics[cate][num] = (
                    getattr(self, f"{cate}_{num}", None).cpu().numpy()
                )
        format_metrics_show = self.print_log(format_metrics)
        return format_metrics_show
