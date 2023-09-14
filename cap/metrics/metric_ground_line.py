import logging
from typing import Dict, List

import numpy as np

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import convert_numpy
from .metric import EvalMetric

__all__ = ["GroundLineMetric"]

logger = logging.getLogger(__name__)

GT_TYPE_MAP = {"normal": 0, "hard": 1, "ignore": 2}

GT_TYPE_MAP_BACK = {0: "normal", 1: "hard", 2: "ignore"}


def clip_bbox(bbox, img_roi):
    x1, y1, x2, y2 = bbox[0:4]
    xmin, ymin, xmax, ymax = img_roi
    x1 = np.clip(x1, xmin, xmax)
    x2 = np.clip(x2, xmin, xmax)
    y1 = np.clip(y1, ymin, ymax)
    y2 = np.clip(y2, ymin, ymax)

    return float(x1), float(y1), float(x2), float(y2)


def caldist(detkps, gtkps):
    detkps = np.array(detkps).astype(np.float32)
    gtkps = np.array(gtkps).astype(np.float32)

    dists = (detkps - gtkps).reshape(-1, 2)
    dists = np.sum(dists ** 2, axis=1, keepdims=True) ** 0.5
    return dists


def calfp(fp, rec):
    ap = 0
    for i in range(len(fp) - 1):
        ap += (fp[i + 1] - fp[i]) * rec[i]
        if fp[i + 1] > 100:
            break
    return ap / 100


def calar(fppi, rec):
    trans_fppi = np.log10(fppi) / 3.0 + 1
    mfppi = trans_fppi[(trans_fppi >= 0) & (trans_fppi <= 1)]
    mrec = rec[(trans_fppi >= 0) & (trans_fppi <= 1)]
    ar = 0
    for i in range(len(mrec) - 1):
        ar += (mfppi[i + 1] - mfppi[i]) * max(mrec[i], mrec[i + 1])
    return ar


def calap(recall, prec):
    mrec = [0] + list(recall.flatten()) + [1]
    mpre = [0] + list(prec.flatten()) + [0]

    ap = 0
    for i in range(len(mpre) - 1):
        if mpre[i + 1] > 0:  # 0.9:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap, mrec[1:-1], mpre[1:-1]


def cal_mean_var(values):
    mean = values.mean()
    std = values.std()

    abs_mean = np.abs(values).mean()
    abs_std = np.abs(values).std()

    return mean, std, abs_mean, abs_std


class GroundLineEval(object):
    def __init__(
        self,
        gt_annos: Dict,
        predictions: Dict,
        iou_types: List[str] = ["iou", "iou", "iou"],  # noqa B006
        num_kps: int = 2,
        ap_sort_base: str = "score",
        overlap_threshes: List[float] = [0.7, 0.5, 0.5],  # noqa B006
        oks_threshes: List[float] = [0.7, 0.5, 0.5],  # noqa B006
        angle_threshes: List[float] = None,  # default not used
        remove_bbox_fp: bool = False,
        remove_bbox_fn: bool = False,
    ):
        """Ground Line Evaluator.

        Args:
            gt_annos (Dict): Ground truth.
            predictions (Dict): Prediction results.
            iou_types (str): Iou model. Defaults to "iou".
            num_kps (int): Number of key points. Defaults to 2.
            ap_sort_base (str): Key by which AP is sorted. Defaults to "score".
            overlap_threshes (List[float]): Overlap iou thresh for `["normal",
                "hard, and "ignore"]`. Defaults to [0.7, 0.5, 0.5].
            oks_threshes (List[float]): Key points iou thresh for `["normal",
                "hard, and "ignore"]`. Defaults to [0.7, 0.5, 0.5].
            angle_threshes (List[float]): Angle iou thresh for `["normal",
                "hard, and "ignore"]`. Defaults to None.
            remove_bbox_fp (bool): Whether to remove FP bbox samples.
            remove_bbox_fn (bool): Whether to remove FN bbox samples.
        """

        self.gt_records = gt_annos
        self.pr_records = predictions

        self.iou_types = iou_types
        self.num_kps = num_kps
        self.ap_sort_base = ap_sort_base
        self.overlap_threshes = overlap_threshes
        self.oks_threshes = oks_threshes
        self.angle_threshes = angle_threshes
        self.remove_bbox_fp = remove_bbox_fp
        self.remove_bbox_fn = remove_bbox_fn

    def evaluate(self):
        num_image = len(self.pr_records)

        eval_results = []
        tp_all = []
        fp_all = []
        tp_kps_all = []
        fp_kps_all = []
        score_all = []
        oks_all = []
        tp_gt_pair_all = []
        angle_error_all = []
        offset_error_all = []
        num_gt_all = 0

        assert len(self.gt_records) == len(self.pr_records)
        for gt_record, pr_record in zip(self.gt_records, self.pr_records):
            (
                eval_result,
                gt_detected_mask,
                tp_mask,
                fp_mask,
                tp_kps_array,
                fp_kps_array,
                tp_gt_pair,
                dt_kps_score,
                oks_array,
                angle_error_array,
                offset_error_array,
                num_gt,
            ) = self.evaluate_record(gt_record, pr_record)
            eval_results.append(eval_result)
            tp_all.append(tp_mask)
            fp_all.append(fp_mask)
            tp_kps_all.append(tp_kps_array)
            fp_kps_all.append(fp_kps_array)
            tp_gt_pair_all.append(tp_gt_pair)
            score_all.append(dt_kps_score)
            oks_all.append(oks_array.mean(axis=1, keepdims=True))
            angle_error_all.append(angle_error_array)
            offset_error_all.append(offset_error_array)
            num_gt_all += float(num_gt)

        tp = np.vstack(tp_all)
        fp = np.vstack(fp_all)
        score = np.vstack(score_all)  # []
        oks = np.vstack(oks_all)  # [N,1]
        angle_error = np.vstack(angle_error_all)  # [N,1]
        offset_error = np.vstack(offset_error_all)  # [N,2]

        # valid mask
        valid_mask = np.logical_or(tp == 1, fp == 1).flatten()
        tp = tp[valid_mask]
        fp = fp[valid_mask]
        score = score[valid_mask]
        oks = oks[valid_mask]
        angle_error = angle_error[valid_mask]
        offset_error = offset_error[valid_mask]

        if self.ap_sort_base == "oks":
            arg_sort = np.argsort(-oks.flatten())
        elif self.ap_sort_base == "score":
            arg_sort = np.argsort(-score.flatten())
        else:
            raise NotImplementedError
        score = score[arg_sort]
        oks = oks[arg_sort]
        tp = tp[arg_sort]
        fp = fp[arg_sort]
        tp = np.cumsum(np.require(tp, dtype=np.float))
        fp = np.cumsum(np.require(fp, dtype=np.float))
        recall = tp / (num_gt_all + 1e-10)
        prec = tp / (tp + fp + 1e-10)
        fppi = fp / float(num_image) + 1e-10
        accuracy = (tp - fp) / (num_gt_all + 1e-10)
        ar = calar(fppi, recall)
        ap, recall, prec = calap(recall, prec)

        angle_error = angle_error[arg_sort]
        angle_mean, angle_std, abs_angle_mean, abs_angle_std = cal_mean_var(
            angle_error
        )  # noqa
        offset_error = offset_error[arg_sort]  # [N,num_kps]

        results = dict()  # noqa C408
        results["aps"] = {
            "ap": float(ap),
            "rec": float(recall[-1] if len(recall) > 0 else np.inf),
            "acc": float(max(accuracy) if len(accuracy) > 0 else np.inf),
            "ar": float(ar),
            "num_image": float(num_image),
            "num_gt": float(num_gt_all),
            "num_tp": float(tp[-1] if len(tp) else 0),
            "num_fp": float(fp[-1] if len(fp) else 0),
        }
        results["recall"] = recall
        results["precision"] = prec
        results["fppi"] = fppi.tolist()
        if self.ap_sort_base == "oks":
            results["conf"] = oks.flatten().tolist()
        elif self.ap_sort_base == "score":
            results["conf"] = score.flatten().tolist()
        else:
            raise NotImplementedError
        results["accuracy"] = accuracy.tolist()
        results["fp"] = fp.tolist()
        results["angle_error_mean"] = float(angle_mean.item())
        results["angle_error_std"] = float(angle_std.item())
        results["abs_angle_error_mean"] = float(abs_angle_mean.item())
        results["abs_angle_error_std"] = float(abs_angle_std.item())
        results["offset_error_mean"] = float(offset_error.mean().item())
        results["offset_error_std"] = float(offset_error.std().item())

        return ap, recall, prec, ar, eval_results, results

    def evaluate_record(self, gt_record, pr_record):
        """Evaluate single record."""
        # compute gts and objs oks on cur image
        # gts: m, obj: n
        # iou: m x n
        num_kps = self.num_kps
        eval_result = {
            # "image_key": image_key,
            "gts": [],
            "dts": [],
        }
        gts, dts = self.transOKSinput(gt_record, pr_record)
        # array with shape [num_dt,num_gt]
        ious_bbox = self.computeIouBbox(gts, dts)

        gt_kps_array, dt_kps_array, dt_bbox_score_array = self.transMatchinput(
            gt_record, pr_record
        )
        num_gt = len(gt_kps_array)
        num_dt = len(dt_kps_array)
        num_ig = (gt_kps_array[..., -1] == 2).sum()

        dt_kps_score = dt_kps_array[:, num_kps * 2].copy().flatten()
        dt_kps_score = dt_kps_score * dt_bbox_score_array.flatten()
        arg_sort = np.argsort(-dt_bbox_score_array.flatten())
        dt_kps_array = dt_kps_array[arg_sort]
        dt_kps_score = dt_kps_score[arg_sort].reshape(-1, 1)
        ious_bbox = ious_bbox[arg_sort]

        # 0: unmatch, 1: normal, 2: hard, 3: ignore, 4: FN
        gt_detected_mask = np.zeros((num_gt, 1))
        tp_mask = np.zeros((num_dt, 1))
        match_hard_mask = np.zeros_like(tp_mask)
        match_ignore_mask = np.zeros_like(tp_mask)
        fp_mask = np.zeros_like(tp_mask)
        oks_array = np.full([num_dt, 2], -10000, np.float)  # [num_dt,2]
        angle_error_array = np.zeros_like(tp_mask)  # [num_dt,1]
        offset_error_array = np.zeros(
            [num_dt, num_kps], np.float
        )  # [num_dt,num_kps]

        tp_gt_pair = []
        for dt_idx in range(num_dt):
            dt_kps = dt_kps_array[dt_idx]
            origin_dt_idx = arg_sort[dt_idx]
            normal_max_idx = -1  # matched to normal
            hard_max_idx = -1  # matched to hard
            ignore_max_idx = -1  # matched to ignore
            normal_max_iou = -1000000
            hard_max_iou = -1000000
            ignore_max_iou = -1000000

            # match the detection to the groundtruth, get the max match
            for gt_idx in range(num_gt):
                # not matched or ignored groundtruth
                if gt_detected_mask[gt_idx] not in [0, 3]:
                    continue
                gt_kps = gt_kps_array[gt_idx]

                cur_iou = ious_bbox[dt_idx, gt_idx]
                gt_type = gt_kps[-1]
                if gt_type == 0:  # normal groundtruth
                    if (
                        cur_iou > normal_max_iou
                        and cur_iou > self.overlap_threshes[0]  # normal
                    ):  # noqa
                        normal_max_iou = cur_iou
                        normal_max_idx = gt_idx
                elif gt_type == 1:  # hard groundtruth
                    if (
                        cur_iou > hard_max_iou
                        and cur_iou > self.overlap_threshes[1]  # hard
                    ):  # noqa
                        hard_max_iou = cur_iou
                        hard_max_idx = gt_idx
                elif gt_type == 2:  # ignore groundtruth
                    if (
                        cur_iou > ignore_max_iou
                        and cur_iou > self.overlap_threshes[2]  # ignore
                    ):  # noqa
                        ignore_max_iou = cur_iou
                        ignore_max_idx = gt_idx
                else:
                    raise NotImplementedError

            # set the result for dt
            if normal_max_idx >= 0:  # match normal gt
                (oks, angle_error, offset_error,) = self.compute_similarity(
                    gt_record["bbox"][normal_max_idx],
                    gt_record["gdl"][normal_max_idx],
                    pr_record["bbox"][origin_dt_idx],
                    pr_record["gdl"][origin_dt_idx],
                )
                if self.oks_threshes is not None:
                    oks_tp = bool(oks.mean() > self.oks_threshes[0])  # normal
                else:
                    oks_tp = True
                if self.angle_threshes is not None:
                    # normal
                    angle_tp = np.abs(angle_error) < self.angle_threshes[0]
                else:
                    angle_tp = True
                if oks_tp and angle_tp:
                    tp_mask[dt_idx] = 1  # TP normal
                    gt_detected_mask[normal_max_idx] = 1
                    gt_kps = gt_kps_array[normal_max_idx]
                    tp_gt_pair.append(
                        (gt_kps[: 2 * num_kps], dt_kps[: 2 * num_kps])
                    )  # noqa
                else:
                    fp_mask[dt_idx] = 1  # FP normal
                    gt_detected_mask[normal_max_idx] = 4  # FN normal
                oks_array[dt_idx, :] = oks.flatten()
                angle_error_array[dt_idx, :] = angle_error
                offset_error_array[dt_idx, :] = offset_error
            elif hard_max_idx >= 0:  # match hard gt
                (
                    oks,
                    angle_error,
                    offset_error,
                ) = self.compute_similarity(  # noqa E501
                    gt_record["bbox"][hard_max_idx],
                    gt_record["gdl"][hard_max_idx],
                    pr_record["bbox"][origin_dt_idx],
                    pr_record["gdl"][origin_dt_idx],
                )
                if self.oks_threshes is not None:
                    oks_tp = bool(oks.mean() > self.oks_threshes[1])  # hard
                else:
                    oks_tp = True
                if self.angle_threshes is not None:
                    # hard
                    angle_tp = np.abs(angle_error) < self.angle_threshes[1]
                else:
                    angle_tp = True
                if oks_tp and angle_tp:
                    match_hard_mask[dt_idx] = 1  # TP hard
                    gt_detected_mask[hard_max_idx] = 2
                else:
                    fp_mask[dt_idx] = 1  # FP hard
                    gt_detected_mask[hard_max_idx] = 4  # FN hard
                oks_array[dt_idx, :] = oks.flatten()
                angle_error_array[dt_idx, :] = angle_error
                offset_error_array[dt_idx, :] = offset_error
            elif ignore_max_idx >= 0:  # match ignore gt
                match_ignore_mask[dt_idx] = 1
                gt_detected_mask[ignore_max_idx] = 3
            elif len(gt_record["bbox"]) > 0:  # unmatched dt
                if not self.remove_bbox_fp:
                    fp_mask[dt_idx] = 1
            else:
                continue

        # process the ground truth
        for gt_idx in range(num_gt):
            gt = gts[gt_idx]
            gdl = np.array(gt["gdl"], dtype=np.float)
            kps = np.array([gdl[0], gdl[1], 1, gdl[2], gdl[3], 1], np.float)
            eval_gt = {
                "bbox": gt["bbox"],
                "keypoints": kps.tolist(),
                "gt_type": GT_TYPE_MAP_BACK[gt["type"]],
            }
            if gt_detected_mask[gt_idx] == 1:  # TP normal
                eval_gt["eval_type"] = "TP"
            elif gt_detected_mask[gt_idx] == 2:  # TP hard
                eval_gt["eval_type"] = "HARD"
            elif gt_detected_mask[gt_idx] == 3:  # ignore
                eval_gt["eval_type"] = "IGNORE"
            elif gt_detected_mask[gt_idx] == 4:  # FN
                eval_gt["eval_type"] = "FN"
            else:
                if gt_record["type"][gt_idx] == 2:  # ignore
                    eval_gt["eval_type"] = "IGNORE"
                else:  # FN
                    if self.remove_bbox_fn:
                        eval_gt["eval_type"] = "IGNORE"
                    else:
                        eval_gt["eval_type"] = "FN"

            eval_result["gts"].append(eval_gt)

        # process the detection
        for dt_idx in range(num_dt):
            dt = dts[arg_sort[dt_idx]]
            gdl = np.array(dt["gdl"], dtype=np.float)
            gdl_score = float(dt["gdl_score"])
            kps = np.array(
                [gdl[0], gdl[1], gdl_score, gdl[2], gdl[3], gdl_score],
                np.float,
            )

            eval_dt = {
                "bbox": dt["bbox"],
                "score": dt["bbox_score"],
                "keypoints": kps.tolist(),
            }
            if tp_mask[dt_idx] == 1 or match_hard_mask[dt_idx] == 1:
                eval_dt["eval_type"] = "TP"
            elif match_ignore_mask[dt_idx] == 1:
                eval_dt["eval_type"] = "IGNORE"
            elif fp_mask[dt_idx] == 1:
                eval_dt["eval_type"] = "FP"
            else:
                eval_dt["eval_type"] = "IGNORE"
            eval_result["dts"].append(eval_dt)

        tp_mask = np.logical_or(tp_mask, match_hard_mask).astype(np.float)
        tp_kps_array = dt_kps_array[tp_mask.flatten() == 1]
        fp_kps_array = dt_kps_array[fp_mask.flatten() == 1]
        if self.remove_bbox_fn:
            fn_mask = gt_detected_mask.flatten() == 4
            num_gt = float(tp_mask.sum() + fn_mask.sum())
        else:
            num_gt = num_gt - num_ig

        return (
            eval_result,
            gt_detected_mask,
            tp_mask,
            fp_mask,
            tp_kps_array,
            fp_kps_array,
            tp_gt_pair,
            dt_kps_score,
            oks_array,
            angle_error_array,
            offset_error_array,
            num_gt,
        )

    def transOKSinput(self, gt_record, pr_record):
        gts, dts = [], []
        if "bbox" in gt_record:
            for bbox, gdl, _type in zip(
                gt_record["bbox"], gt_record["gdl"], gt_record["type"]
            ):
                gt = {}

                x1, y1, x2, y2 = bbox["data"]
                w = x2 - x1
                h = y2 - y1
                kps = np.array(gdl["data"])

                gt["bbox"] = [float(v) for v in [x1, y1, w, h]]
                gt["area"] = w * h
                gt["gdl"] = kps.flatten().tolist()  # [4]
                gt["type"] = _type

                gts.append(gt)

        if "bbox" in pr_record:
            for bbox, gdl in zip(pr_record["bbox"], pr_record["gdl"]):
                dt = {}

                x1, y1, x2, y2 = bbox["data"]
                w = x2 - x1
                h = y2 - y1
                kps = np.array(gdl["data"])  # [num_kps,2]

                dt["bbox"] = [float(v) for v in [x1, y1, w, h]]
                dt["area"] = w * h
                dt["bbox_score"] = bbox["score"]
                dt["gdl"] = kps.flatten().tolist()  # [4]
                dt["gdl_score"] = gdl["score"]

                dts.append(dt)

        return gts, dts

    def computeIouBbox(self, gts, dts):
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            for i, dt in enumerate(dts):
                if gt["type"] in [0, 1, 2]:  # [normal, hard, ignore]
                    iou_mode = self.iou_types[gt["type"]]
                else:
                    raise NotImplementedError

                ious[i, j] = self.compute_iou(gt["bbox"], dt["bbox"], iou_mode)
        return ious

    @staticmethod
    def compute_iou(pos1, pos2, mode="iod"):
        left1, top1, w1, h1 = pos1
        left2, top2, w2, h2 = pos2
        right1 = left1 + w1
        down1 = top1 + h1
        right2 = left2 + w2
        down2 = top2 + h2
        left = max(left1, left2)
        right = min(right1, right2)
        top = max(top1, top2)
        bottom = min(down1, down2)
        if left >= right or top >= bottom:
            return 0
        else:
            area1 = (right1 - left1) * (down1 - top1)
            area2 = (right2 - left2) * (down2 - top2)
            area_sum = area1 + area2
            inter = (right - left) * (bottom - top)
            if mode == "iou":
                return inter / (area_sum - inter)
            elif mode == "iod":
                return inter / area2
            elif mode == "iog":
                return inter / area1
            else:
                raise NotImplementedError

    def transMatchinput(self, gt_record, pr_record):
        """Translate and match gt and prediction.

        Args:
            gt_record(dict):
            pr_record(dict):

        Returns:
            gt_kps_lst(list): each with shape [num_kps*2+1]
                num_kps*2: coordinate of each points
                1: type of bbox; value in [0,1,2],
                    {
                        'normal': 0,
                        'hard': 1,
                        'ignore': 2
                    }
            dt_kps_lst(list): each with shape [num_kps*2+1]
                num_kps*2: coordinate of each points
                1: score of detection, which fuse the score of bbox and kps

        """
        gt_kps_lst = []
        dt_kps_lst = []
        dt_bbox_score_lst = []

        for i, kps in enumerate(gt_record.get("gdl", [])):
            kps_ele_flat = np.array(kps["data"])
            kps_ele_flat = kps_ele_flat.flatten().tolist()
            kps_ele_flat.append(gt_record["type"][i])
            gt_kps_lst.append(kps_ele_flat)

        for _, kps in enumerate(pr_record.get("gdl", [])):
            kps_ele_flat = np.array(kps["data"])
            kps_ele_flat = kps_ele_flat.flatten().tolist()
            kps_ele_flat.append(kps["score"])
            dt_kps_lst.append(kps_ele_flat)

        for bbox in pr_record.get("bbox", []):
            dt_bbox_score_lst.append(bbox["score"])

        assert len(dt_kps_lst) == len(dt_bbox_score_lst)

        return (
            np.array(gt_kps_lst).reshape([-1, 5]),
            np.array(dt_kps_lst).reshape([-1, 5]),
            np.array(dt_bbox_score_lst).reshape(-1, 1),
        )

    def compute_similarity(self, gt_bbox, gt_gdl, dt_bbox, dt_gdl):
        """Compute similarity.

        Args:
            gt_bbox(dict): {'data': <>}
            gt_gdl(dict): {'data': <left_pt,right_pt>}
            dt_bbox(dict): {'data': <>, 'score': <>}
            dt_gdl(dict): {'data': <left_pt,right_pt>,
                           'score': }

        Returns:
            ks(np.ndarray): [left_pt_ks,right_pt_ks]
                with shape [2]
            angle_error(float):
            offset_error(np.ndarray): [[left_x_error, left_y_error],
                                       [right_x_error, right_y_error]
                with shape [2,2]

        """
        gt_bbox_data = gt_bbox["data"]
        x1, y1, x2, y2 = gt_bbox_data
        w, h = x2 - x1, y2 - y1
        area = w * h
        gt_left_pt, gt_right_pt = gt_gdl["data"]
        (
            gt_left,
            gt_right,
            gt_top,
            gt_bottom,
        ) = self.get_intersection_points_to_bbox_edge(
            gt_bbox_data, gt_left_pt, gt_right_pt
        )
        gt_left_pt = [x1, gt_left]
        gt_right_pt = [x2, gt_right]

        dt_left_pt, dt_right_pt = dt_gdl["data"]
        (
            dt_left,
            dt_right,
            dt_top,
            dt_bottom,
        ) = self.get_intersection_points_to_bbox_edge(
            gt_bbox_data, dt_left_pt, dt_right_pt
        )
        dt_left_pt = [x1, dt_left]
        dt_right_pt = [x2, dt_right]

        gt_kps = np.array([gt_left_pt, gt_right_pt], np.float).reshape([-1, 2])
        dt_kps = np.array([dt_left_pt, dt_right_pt], np.float).reshape([-1, 2])
        dx = gt_kps[:, 0] - dt_kps[:, 0]
        dy = gt_kps[:, 1] - dt_kps[:, 1]

        # ks
        sigmas = 0.05
        variance = (sigmas * 2) ** 2
        error = (dx ** 2 + dy ** 2) / (2 * area * variance)
        ks = np.exp(-error)

        # angle
        gt_angle = np.arctan2(
            gt_kps[0, 1] - gt_kps[1, 1], gt_kps[0, 0] - gt_kps[1, 0]
        )
        gt_angle = np.where(gt_angle >= 0, gt_angle, 2 * np.pi + gt_angle)
        dt_angle = np.arctan2(
            dt_kps[0, 1] - dt_kps[1, 1], dt_kps[0, 0] - dt_kps[1, 0]
        )
        dt_angle = np.where(dt_angle >= 0, dt_angle, 2 * np.pi + dt_angle)
        angle_error = (gt_angle - dt_angle) * 180.0 / np.pi

        # distance
        offset_error = np.sqrt(dx ** 2 + dy ** 2)  # [2]

        return ks, angle_error, offset_error

    @staticmethod
    def get_point_to_line_dist(line_pt_one, line_pt_other, point):
        line_pt_one = np.array(line_pt_one, np.float).flatten()
        line_pt_other = np.array(line_pt_other, np.float).flatten()
        point = np.array(point, np.float).flatten()
        vector_line = line_pt_other - line_pt_one
        vector_point = point - line_pt_one
        cross_mul = np.abs(np.cross(vector_line, vector_point))
        dist = (cross_mul / np.linalg.norm(vector_line)).item()

        return float(dist)

    def get_intersection_points_to_bbox_edge(
        self, bbox, pta, ptb, img_roi=None
    ):
        """Get the absolute intersection to bbox left top point.

        Args:
            bbox(np.ndarray): corner representation of bbox; <x1,y1,x2,y2>
            pta: <x,y>; one endpoint of line segment
            ptb: <x,y>; other endpoint of line segment
        """
        if img_roi is not None:
            bbox = clip_bbox(bbox, img_roi)
        x1, y1, x2, y2 = bbox
        ax, ay = np.asarray(pta, dtype=np.float32)
        bx, by = np.asarray(ptb, dtype=np.float32)
        delta_x = ax - bx
        delta_y = ay - by

        if np.abs(delta_x) == 0:  # vertical line
            left_intersection, right_intersection = None, None
            top_intersection, bottom_intersection = ax, ax
        elif np.abs(delta_y) == 0:  # horizontal line
            left_intersection, right_intersection = ay, ay
            top_intersection, bottom_intersection = None, None
        else:  # slope line
            slope = delta_y / delta_x
            left_intersection = slope * (x1 - bx) + by
            right_intersection = slope * (x2 - bx) + by
            top_intersection = (1 / slope) * (y1 - by) + bx
            bottom_intersection = (1 / slope) * (y2 - by) + bx

        return (
            left_intersection,
            right_intersection,
            top_intersection,
            bottom_intersection,
        )


@OBJECT_REGISTRY.register
class GroundLineMetric(EvalMetric):
    """Ground Line detection metric.

    Args:
        num_kps (int): Number of key points. Defaults to 2.
        ap_sort_base (str): Key by which AP is sorted. Defaults to "score".
        iou_types (List[float]): Iou types for `["normal", "hard,
            and "ignore"]`. Defaults to ["oks", "oks", "iod].
        overlap_threshes (List[float]): Overlap iou thresh for `["normal",
                "hard, and "ignore"]`. Defaults to [0.7, 0.5, 0.5].
        oks_threshes (List[float]): Key points iou thresh for `["normal",
                "hard, and "ignore"]`. Defaults to [0.7, 0.5, 0.5].
        angle_threshes (List[float]): Angle iou thresh for `["normal",
                "hard, and "ignore"]`. Defaults to None.
        remove_bbox_fp (bool): Whether to remove FP bbox samples.
        remove_bbox_fn (bool): Whether to remove FN bbox samples.
        bbox_score_thresh (float): Thresh of bbox score thresh.
        gdl_score_thresh (float): Thresh of bbox score thresh.
        name (str): Metric name.
    """

    def __init__(
        self,
        num_kps: int = 2,
        ap_sort_base: str = "score",
        iou_types: List[str] = ["iou", "iou", "iou"],  # noqa B006
        overlap_threshes: List[float] = [0.7, 0.5, 0.5],  # noqa B006
        oks_threshes: List[float] = [0.7, 0.5, 0.5],  # noqa B006
        angle_threshes: List[float] = None,  # default not used
        remove_bbox_fp: bool = True,
        remove_bbox_fn: bool = False,
        bbox_score_thresh: float = 0.1,
        gdl_score_thresh: float = 0.1,
        name: str = "GroundLineMetric",
    ):

        super().__init__(name)

        self.iou_types = iou_types
        self.num_kps = num_kps
        self.ap_sort_base = ap_sort_base
        self.overlap_threshes = overlap_threshes  # [normal, hard, ignore]
        self.oks_threshes = oks_threshes  # [normal, hard, ignore]
        self.angle_threshes = angle_threshes  # [normal, hard, ignore]
        self.remove_bbox_fp = remove_bbox_fp
        self.remove_bbox_fn = remove_bbox_fn
        self.bbox_score_thresh = bbox_score_thresh
        self.gdl_score_thresh = gdl_score_thresh

    def reset(self):
        self.predictions = []
        self.gt_annos = []

    def update(self, batch, preds):

        # 1. convert groundtruth
        batch_data = batch
        # gts
        gt_boxes_num = convert_numpy(
            batch_data["gt_boxes_num"].squeeze(-1), dtype=np.int
        )
        gt_flanks_num = convert_numpy(
            batch_data["gt_flanks_num"].squeeze(-1), dtype=np.int
        )
        gt_boxes = convert_numpy(batch_data["gt_boxes"])
        gt_flanks = convert_numpy(batch_data["gt_flanks"])
        # img_keys = batch_data["image_name"]
        assert (gt_boxes_num == gt_flanks_num).all()
        batch_size = gt_boxes.shape[0]

        # predictions
        preds_parents = preds["detection"]
        preds_gdls = preds["ground_line"]

        assert len(preds_parents) == len(preds_gdls) == batch_size

        for idx in range(batch_size):
            # 1. convert groundtruth
            gt_boxes_i = gt_boxes[idx]
            gt_boxes_num_i = gt_boxes_num[idx]
            gt_flanks_i = gt_flanks[idx]
            gt_flanks_num_i = gt_flanks_num[idx]

            gt_boxes_i = gt_boxes_i[:gt_boxes_num_i, :]
            gt_flanks_i = gt_flanks_i[:gt_flanks_num_i, :]
            gt_boxes_lst, gt_gdl_lst, gt_type_lst = [], [], []
            for box, flank in zip(gt_boxes_i, gt_flanks_i):
                gt_boxes_lst.append(
                    {
                        "data": box[:4].tolist(),
                    }
                )
                left_pt = flank[:2].tolist()
                right_pt = flank[2:4].tolist()
                gt_gdl_lst.append(
                    {
                        "data": [left_pt, right_pt],
                    }
                )
                # 0:normal, 1:hard, 2:ignore
                # Note: all used 0
                gt_type_lst.append(0)
            gt_record = {
                # "image_key": img_key,
                "bbox": gt_boxes_lst,
                "gdl": gt_gdl_lst,
                "type": gt_type_lst,
            }

            self.gt_annos.append(gt_record)

            # 2. convert groundtruth
            # image_key = img_keys[idx]
            preds_parent = preds_parents[idx]
            preds_gdl = preds_gdls[idx]
            pred_bbox_lst, pred_gdl_lst = [], []
            for det_parent, det_child in zip(preds_parent, preds_gdl):
                bbox_score = convert_numpy(det_parent.score)
                assert det_child.num_points == self.num_kps
                point0, point1 = det_child.point0, det_child.point1
                p0_score = convert_numpy(point0.score)
                p1_score = convert_numpy(point1.score)
                kps_scores = np.array([p0_score, p1_score])
                gdl_score = kps_scores.mean()

                if (
                    self.bbox_score_thresh is not None
                    and bbox_score < self.bbox_score_thresh
                ):
                    continue
                if (
                    self.gdl_score_thresh is not None
                    and gdl_score < self.gdl_score_thresh
                ):
                    continue
                pred_bbox_lst.append(
                    {
                        "data": convert_numpy(det_parent.box).tolist(),
                        "score": bbox_score,
                    }
                )
                left_pt = convert_numpy(point0.point).tolist()
                right_pt = convert_numpy(point1.point).tolist()  # noqa E501
                pred_gdl_lst.append(
                    {
                        "data": [left_pt, right_pt],
                        "score": gdl_score,
                    }
                )

            pred_record = {
                # "image_key": image_key,
                "bbox": pred_bbox_lst,
                "gdl": pred_gdl_lst,
            }

            self.predictions.append(pred_record)

    def get(self):
        try:
            gdl_eval = GroundLineEval(
                gt_annos=self.gt_annos,
                predictions=self.predictions,
                iou_types=self.iou_types,
                num_kps=self.num_kps,
                ap_sort_base=self.ap_sort_base,
                overlap_threshes=self.overlap_threshes,
                oks_threshes=self.oks_threshes,
                remove_bbox_fp=self.remove_bbox_fp,
                remove_bbox_fn=self.remove_bbox_fn,
            )
        except IndexError:
            # invalid model may result in empty JSON results, skip it
            return ["AP"], ["0.0"]

        ap, recall, prec, ar, results, pr_results = gdl_eval.evaluate()

        names = ["ap", "ar"]
        values = [ap, ar]

        for k in ["num_gt", "num_tp", "num_fp"]:
            names.append(k)
            values.append(pr_results["aps"][k])

        log_info = "\n"
        for k, v in zip(names, values):
            if isinstance(v, (int, float)):
                log_info += "%s: [%.4f] \n" % (k, v)
            else:
                log_info += "%s: [%s] \n" % (str(k), str(v))
        logger.info(log_info)

        return names[0], values[0]  # ap
