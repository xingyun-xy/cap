import logging
from typing import Dict, List

import numpy as np

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import convert_numpy as to_numpy
from .metric import EvalMetric

logger = logging.getLogger(__name__)


__all__ = ["KpsMetric"]


gt_type_map_back = {0: "normal", 1: "hard", 2: "ignore"}
empty_points_number = -10000


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


def caldist(detkps, gtkps):
    detkps = np.array(detkps).astype(np.float32)
    gtkps = np.array(gtkps).astype(np.float32)

    dists = (detkps - gtkps).reshape(-1, 2)
    dists = np.sum(dists ** 2, axis=1, keepdims=True) ** 0.5
    return dists


class KeyPointsEval(object):
    """KeyPoints Evaluator.

    Args:
        gt_annos (Dict): Ground truth.
        predictions (Dict): Prediction results.
        iou_types (List[str]): Iou model for "normal", "hard", and "ignore".
        num_kps (int): Number of key points. Defaults to 2.
        overlap_threshes (List[float]): Overlap iou thresh for `["normal",
            "hard, and "ignore"]`. Defaults to [0.7, 0.5, 0.5].
        remove_bbox_fp (bool): Whether to remove FP bbox samples.
        remove_bbox_fn (bool): Whether to remove FN bbox samples.
    """

    def __init__(
        self,
        gt_annos: Dict,
        predictions: Dict,
        iou_types: List[str] = ["oks", "oks", "iod"],  # noqa B006
        num_kps: int = 2,
        overlap_threshes: List[float] = [0.7, 0.5, 0.5],  # noqa B006
        remove_bbox_fp: bool = False,
        remove_bbox_fn: bool = False,
    ):

        self.gt_records = gt_annos
        self.pr_records = predictions

        self.iou_types = iou_types  # [normal, hard, ignore]
        self.num_kps = num_kps
        self.overlap_threshes = overlap_threshes
        self.remove_bbox_fp = remove_bbox_fp
        self.remove_bbox_fn = remove_bbox_fn

    def get_kps_score(self, kps, bbox_score):
        kps = np.array(kps)
        kps_each_scores = kps[2::3].copy()
        kps_each_scores[kps_each_scores < 0.1] = 0
        kps_scores_sum = kps_each_scores.sum(axis=0)
        kps_each_scores[kps_each_scores > 0] = 1
        kps_scores_count = kps_each_scores.sum(axis=0)
        # kps_scores_count[kps_scores_count == 0] = 1
        if kps_scores_count == 0:
            kps_score = 0.0
        else:
            kps_score = kps_scores_sum / kps_scores_count
        kps_score = (kps_score + bbox_score) / 2
        return kps_score

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

    def computeIouBbox(self, gts, dts, ignore_iou_mode="iou"):
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            for i, dt in enumerate(dts):
                ious[i, j] = self.compute_iou(
                    gt["bbox"], dt["bbox"], ignore_iou_mode
                )
        return ious

    def computeOKS(self, gts, dts, ignore_iou_mode="iod"):

        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        num_kps = len(gts[0]["keypoints"]) // 3
        if num_kps == 12:
            sigmas = np.array(
                [
                    0.04773482,
                    0.0463056,
                    0.06464321,
                    0.079478,
                    0.0366433,
                    0.05533729,
                    0.0533272,
                    0.04968713,
                    0.03468218,
                    0.03516776,
                    0.03781802,
                    0.04458622,
                ]
            )
        else:
            sigmas = (
                np.full(shape=(num_kps,), fill_value=1.0, dtype=np.float)
                / 10.0
                * 0.5
            )

        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            eval_attrs_types = 0
            k1 = np.count_nonzero(vg > eval_attrs_types)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            tmp = np.full_like(xg, empty_points_number)
            if (
                (xg == tmp).all()
                and (yg == tmp).all()
                or gt["type"] == 2
                or gt["type"] == 1
            ):
                for i, dt in enumerate(dts):
                    ious[i, j] = self.compute_iou(
                        gt["bbox"], dt["bbox"], ignore_iou_mode
                    )
            else:
                for i, dt in enumerate(dts):
                    d = np.array(dt["keypoints"])
                    xd = d[0::3]
                    yd = d[1::3]
                    if k1 > 0:
                        # measure the per-keypoint distance if keypoints
                        # visible
                        dx = xd - xg
                        dy = yd - yg
                    else:
                        # measure minimum distance to keypoints in (x0,y0) &
                        # (x1,y1)
                        z = np.zeros((k))
                        dx = np.max((z, x0 - xd), axis=0) + np.max(
                            (z, xd - x1), axis=0
                        )
                        dy = np.max((z, y0 - yd), axis=0) + np.max(
                            (z, yd - y1), axis=0
                        )
                    e = (
                        (dx ** 2 + dy ** 2)
                        / vars
                        / (gt["area"] + np.spacing(1))
                        / 2
                    )
                    if k1 > 0:
                        e = e[vg > eval_attrs_types]
                    ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]

        return ious

    def transOKSinput(self, gt, dt):
        """Translate input.

        Args:
            gt = {
                    'bbox': [{'data': [x1,y1,x2,y2]}, ...],
                    'keypoints': [{
                        'data':[[x1,y1], [x2,y2]],
                        'score': [s1, s2]}, ...]
                    }
            dt = {
                    'bbox': [{'score': s, 'data': [x1,y1,x2,y2]}, ...],
                    'keypoints': [{
                        'data': [[x1,y1], [x2,y2]],
                        'score': [s1, s2]}, ...]}

        Return:
            gts = [{
                'bbox': [x1, y1, w, h],
                'area': w*h,
                'keypoints': [x1,y1,2.0,x2,y2,1.0],
                'num_keypoints': 2,
                'type': 0
            }, ...]
            dts = [{
                'bbox': [x1, y1, w, h],
                'area': w*h,
                'keypoints': [x1,y1,0.9,x2,y2,0.8],
                'scores': bbox_score*kps_score
            }, ...]
        """
        gts = []
        dts = []
        num_kps = self.num_kps
        if "bbox" in gt.keys():
            for i, bbox_ele in enumerate(gt["bbox"]):
                gts_ele = {}
                x = bbox_ele["data"][0]
                y = bbox_ele["data"][1]
                w = bbox_ele["data"][2] - bbox_ele["data"][0]
                h = bbox_ele["data"][3] - bbox_ele["data"][1]
                gts_ele["bbox"] = [float(v) for v in [x, y, w, h]]
                gts_ele["area"] = w * h
                kps = np.array(gt["keypoints"][i]["data"])
                score = np.array(gt["keypoints"][i]["score"]).reshape(
                    num_kps, -1
                )
                kps_trans = np.hstack((kps, score))
                gts_ele["keypoints"] = kps_trans.flatten().tolist()
                gts_ele["num_keypoints"] = num_kps
                gts_ele["type"] = gt["type"][i]
                gts.append(gts_ele)

        if "bbox" in dt.keys():
            for i, bbox_ele in enumerate(dt["bbox"]):
                dts_ele = {}
                x = bbox_ele["data"][0]
                y = bbox_ele["data"][1]
                w = bbox_ele["data"][2] - bbox_ele["data"][0]
                h = bbox_ele["data"][3] - bbox_ele["data"][1]
                dts_ele["bbox"] = [float(v) for v in [x, y, w, h]]
                dts_ele["area"] = w * h
                kps = np.array(dt["keypoints"][i]["data"]).reshape(num_kps, 2)
                score = np.array(dt["keypoints"][i]["score"])
                score = score.reshape(num_kps, 1)
                kps_trans = np.hstack((kps, score))
                dts_ele["keypoints"] = kps_trans.flatten().tolist()

                dts_ele["score"] = self.get_kps_score(
                    dts_ele["keypoints"], bbox_ele["score"]
                )

                dts.append(dts_ele)
        return dts, gts

    def transMatchinput(self, gt, dt):
        gt_kpss = []
        det_kpss = []
        num_kps = self.num_kps
        if "keypoints" in gt.keys():
            for i, kps_ele in enumerate(gt["keypoints"]):
                kps_ele_flat = np.array(kps_ele["data"])
                kps_ele_flat = kps_ele_flat.flatten().tolist()
                kps_ele_flat.append(gt["type"][i])
                gt_kpss.append(kps_ele_flat)
        if "keypoints" in dt.keys():
            for i, kps_ele in enumerate(dt["keypoints"]):
                kps_ele_flat = np.array(kps_ele["data"])
                kps_ele_flat = kps_ele_flat.flatten().tolist()

                kps = np.array(np.array(kps_ele["data"])).reshape(num_kps, 2)
                score = np.array(np.array(kps_ele["score"]))
                score = score.reshape(num_kps, 1)
                kps_trans = np.hstack((kps, score))
                kps_trans = kps_trans.flatten().tolist()
                kps_ele_flat.append(
                    self.get_kps_score(kps_trans, dt["bbox"][i]["score"])
                )
                det_kpss.append(kps_ele_flat)
        return np.array(det_kpss), np.array(gt_kpss)

    def bboxMatchBeforeOks(self, gt, dt, gts, dts, ious_bbox):
        if len(ious_bbox) == 0:
            return dt, dts
        # if len(gt) == 0:

        num_obj = ious_bbox.shape[0]
        num_gt = ious_bbox.shape[1]
        # 按照dt排序
        dt_conf = []

        for dt_ele in dt["bbox"]:
            dt_conf.append(dt_ele["score"])
        idx = np.argsort(-np.array(dt_conf))
        dt["bbox"] = np.array(dt["bbox"])[idx].tolist()
        dt["keypoints"] = np.array(dt["keypoints"])[idx].tolist()
        dts = np.array(dts)[idx].tolist()
        if len(gt) == 0:
            ious_bbox = ious_bbox
        else:
            ious_bbox = ious_bbox[idx]

        fp = np.zeros((num_obj, 1))
        gt_detected = np.zeros((num_gt, 1))
        dt_after = {"bbox": [], "keypoints": []}
        dts_after = []
        for i in range(num_obj):
            jmax = -1
            iou_max = -1000
            for j in range(num_gt):
                if ious_bbox[i][j] > iou_max and ious_bbox[i][j] > 0.5:
                    iou_max = ious_bbox[i][j]
                    jmax = j
            if jmax >= 0:
                gt_detected[jmax] = 1
                dt_after["bbox"].append(dt["bbox"][i])
                dt_after["keypoints"].append(dt["keypoints"][i])
                dts_after.append(dts[i])
            else:
                # 没有匹配上，算作fp
                fp[i] = 1
        return dt_after, dts_after

    def evaluate(self):
        eval_results = []
        num_kps = self.num_kps
        num_gt = 0
        tp_all, fp_all = [], []

        all_conf = []
        num_image = len(self.pr_records)
        gt_detections = []
        img_idx = []
        all_tp_gt_linked = []
        assert len(self.gt_records) == len(self.pr_records)

        for i, (gt, dt) in enumerate(zip(self.gt_records, self.pr_records)):

            eval_result = {
                # 'image_key': image_key,
                "gts": [],
                "dts": [],
            }

            # compute gts and objs oks on cur image
            # gts: m, obj: n
            # iou: m x n
            dts, gts = self.transOKSinput(gt, dt)
            if self.remove_bbox_fp:
                ious_bbox = self.computeIouBbox(
                    gts, dts, ignore_iou_mode="iou"
                )
                dt_after, dts_after = self.bboxMatchBeforeOks(
                    gt, dt, gts, dts, ious_bbox
                )
            else:
                dt_after = dt
                dts_after = dts

            ious = self.computeOKS(
                gts, dts_after, ignore_iou_mode=self.iou_types[2]
            )
            det_kpss, gt_kpss = self.transMatchinput(gt, dt_after)
            num_obj = det_kpss.shape[0]
            num_gt_i = gt_kpss.shape[0]
            num_gt += gt_kpss.shape[0]
            if len(gt_kpss) == 0:
                num_ignore = 0
            else:
                num_ignore = (
                    gt_kpss[:, 2 * num_kps]
                    == np.full_like(gt_kpss[:, 2 * num_kps], 2)
                ).sum()
            # all_num_ignore += num_ignore
            # import ipdb;ipdb.set_trace()
            num_gt -= num_ignore
            if len(det_kpss) > 0:
                conf_unsorted = det_kpss[:, 2 * num_kps]
                idx = np.argsort(-conf_unsorted)
            else:
                idx = []
            """
            # different behavior caused by numpy version
            det_boxes[i]:  array([], shape=(0, 5), dtype=float64)
            idx: array([], dtype=int64)
            1) for numpy-1.13.3
                return: array([], shape=(0, 5), dtype=float64)
            2) for numpy-1.7.1
                get error:
                *** IndexError: index 0 is out of bounds for axis 0 with size 0
            """
            if len(det_kpss) == 0 and len(idx) == 0:
                det_kp = det_kpss  # just empty numpy array
                ious = ious
            else:
                det_kp = det_kpss[idx]
                if len(gt_kpss) == 0:
                    ious = ious
                else:
                    ious = ious[idx]

            # det_kpss: dim = 12: kps_pairs * 4 + score * 4
            # conf: use min conf of 4 confs
            if len(det_kpss) > 0:
                conf = np.min(det_kpss[idx, 2 * num_kps :], axis=1).reshape(
                    -1, 1
                )

            gt_detected = np.zeros((num_gt_i, 1))
            tp = np.zeros((num_obj, 1))  # final TP flag
            matched_hard_gt = np.zeros(
                (num_obj, 1)
            )  # final matched hard samples
            matched_ignore_gt = np.zeros(
                (num_obj, 1)
            )  # final matched ignore samples
            fp = np.zeros((num_obj, 1))
            # all possible TP boxes, but final TP boxes are decided by final TP
            img_idx.extend([i, j] for j in range(num_obj))
            tp_gt_linked = []

            for j in range(num_obj):

                b = det_kp[j]
                kmax = -1  # matched to normal
                hmax = -1  # matched to hard
                imax = -1  # matched to ignore
                ov_max = -1000000
                iv_max = -1000000
                hv_max = -1000000
                for k in range(num_gt_i):

                    # gt bbox ignore
                    if gt_detected[k] != 0 and gt_detected[k] != 3:
                        continue

                    kpgt = gt_kpss[k]

                    ov = ious[j][k]
                    iv = ious[j][k]
                    hv = ious[j][k]
                    if kpgt[2 * num_kps] == 0:  # normal
                        if ov > ov_max and ov > self.overlap_threshes[0]:
                            ov_max = ov
                            kmax = k
                    elif kpgt[2 * num_kps] == 1:  # hard
                        if hv > hv_max and hv > self.overlap_threshes[1]:
                            hv_max = hv
                            hmax = k
                    elif kpgt[2 * num_kps] == 2:  # ignore
                        if iv > iv_max and iv > self.overlap_threshes[2]:
                            iv_max = iv
                            imax = k

                if kmax >= 0:  # match normal gt
                    tp[j] = 1
                    gt_detected[kmax] = 1
                    # record gt+det pair
                    if int(tp[j]) == 1:
                        dists = caldist(
                            gt_kpss[kmax].tolist()[: 2 * num_kps],
                            b[: 2 * num_kps],
                        ).reshape(-1)

                        tp_gt_linked.append(
                            (
                                gt_kpss[kmax].tolist()[: 2 * num_kps],
                                b[: 2 * num_kps],
                                dists,
                            )
                        )
                elif hmax >= 0:  # match hard gt
                    matched_hard_gt[j] = 1
                    gt_detected[hmax] = 2

                elif imax >= 0:  # match ignore gt
                    matched_ignore_gt[j] = 1
                    gt_detected[imax] = 3

                else:  # not matching
                    if len(gt) > 0:
                        fp[j] = 1
            # list(map(lambda x: x['bbox'], dts))
            for idx_gt in range(len(gt_detected)):
                gt_obj = gts[idx_gt]
                tmp = {
                    "bbox": gt_obj["bbox"],
                    # 'eval_type': 'FN',
                    "keypoints": gt_obj["keypoints"],
                    "gt_type": gt_type_map_back[gt_obj["type"]],
                }
                if gt_detected[idx_gt] == 1:
                    tmp["eval_type"] = "TP"
                elif gt_detected[idx_gt] == 2:
                    tmp["eval_type"] = "HARD"
                elif gt_detected[idx_gt] == 3:
                    tmp["eval_type"] = "IGNORE"
                else:
                    if gt["type"][idx_gt] == 2:
                        # gt ignored, and no detbox matched,
                        # so that gt_detected[idx_gt] == 0
                        tmp["eval_type"] = "IGNORE"
                    else:
                        tmp["eval_type"] = "FN"
                eval_result["gts"].append(tmp)

            for idx_dt in range(len(tp)):
                idx_origin = idx[idx_dt]
                dt_obj = dts_after[idx_origin]
                tmp = {
                    "bbox": dt_obj["bbox"],
                    "score": dt_obj["score"],
                    "keypoints": dt_obj["keypoints"],
                }
                if tp[idx_dt] == 1 or matched_hard_gt[idx_dt] == 1:
                    tmp["eval_type"] = "TP"
                elif matched_ignore_gt[idx_dt] == 1:
                    tmp["eval_type"] = "IGNORE"
                elif fp[idx_dt] == 1:
                    tmp["eval_type"] = "FP"
                else:
                    tmp["eval_type"] = "IGNORE"
                    # empty image, but det box exists, ignore it !
                    # raise NotImplementedError
                eval_result["dts"].append(tmp)
            eval_results.append(eval_result)
            gt_detections.append(gt_detected)
            tp = np.logical_or(tp, matched_hard_gt)
            tp_all.append(tp)
            fp_all.append(fp)
            if num_obj > 0:
                all_conf.append(conf)
            all_tp_gt_linked.append(tp_gt_linked)
        tp = np.vstack(tp_all)
        fp = np.vstack(fp_all)
        # num_pos = tp.sum()

        conf = np.vstack(all_conf)
        idx = np.argsort(-conf, axis=0)
        conf = conf[idx]
        img_idx = np.array(img_idx)[idx]
        img_idx = img_idx.reshape(-1, 2)
        tp = np.require(tp[idx], dtype=np.float)
        fp = fp[idx]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (num_gt + 1e-10)
        prec = tp / (tp + fp + 1e-10)
        fppi = fp / float(num_image)
        fppi += 1e-15
        accuracy = (tp - fp) / (num_gt + 1e-10)
        ar = calar(fppi, recall)
        ap, recall, prec = calap(recall, prec)

        results = {}
        results["aps"] = {
            "ap": ap,
            "rec": recall[-1] if len(recall) > 0 else np.inf,
            "acc": max(accuracy) if len(accuracy) > 0 else np.inf,
            "ar": ar,
            "num_image": num_image,
            "num_gt": num_gt,
            "num_tp": tp[-1] if len(tp) else 0,
            "num_fp": fp[-1] if len(fp) else 0,
        }
        results["recall"] = recall
        results["precision"] = prec
        results["fppi"] = fppi.tolist()
        results["conf"] = conf.flatten().tolist()
        results["accuracy"] = accuracy.tolist()
        results["fp"] = fp.tolist()

        return (
            ap,
            recall,
            prec,
            ar,
            eval_results,
            results,
        )


@OBJECT_REGISTRY.register
class KpsMetric(EvalMetric):
    """KeyPoints metric.

    Args:
        num_kps (int): Number of key points. Defaults to 2.
        bbox_score_thresh (float): Thresh of bbox score thresh.
        overlap_threshes (List[float]): Overlap iou thresh for `["normal",
            "hard, and "ignore"]`. Defaults to [0.7, 0.5, 0.5].
        iou_types (List[float]): Iou types for `["normal", "hard,
            and "ignore"]`. Defaults to ["oks", "oks", "iod].
        remove_bbox_fp (bool): Whether to remove FP bbox samples.
        remove_bbox_fn (bool): Whether to remove FN bbox samples.
        name (str): Metric name.
    """

    def __init__(
        self,
        num_kps: int = 2,
        bbox_score_thresh: float = 0.3,
        overlap_threshes: List[float] = [0.7, 0.5, 0.5],  # noqa B006
        iou_types: List[str] = ["oks", "oks", "iod"],  # noqa B006
        remove_bbox_fp: bool = False,
        remove_bbox_fn: bool = False,
        name="KpsMetric",
    ):
        super().__init__(name)
        self.num_kps = num_kps
        self.overlap_threshes = overlap_threshes  # [normal, hard, ignore]
        self.iou_types = iou_types  # [normal, hard, ignore]
        self.remove_bbox_fp = remove_bbox_fp
        self.remove_bbox_fn = remove_bbox_fn
        self.bbox_score_thresh = bbox_score_thresh

    def reset(self):
        self.predictions = []
        self.gt_annos = []

    def update(self, batch, preds):

        # gts
        batch_data = batch
        gt_boxes_num = to_numpy(
            batch_data["gt_boxes_num"].squeeze(-1), dtype=np.int
        )
        gt_boxes = to_numpy(batch_data["gt_boxes"])
        batch_size = gt_boxes.shape[0]

        # predictions
        preds_parents = preds["detection"]
        preds_kps = preds["kps"]

        assert len(preds_parents) == len(preds_kps) == batch_size

        for idx in range(batch_size):
            # 1. convert ground truth, one image
            gt_boxes_num_i = gt_boxes_num[idx]
            gt_boxes_i = gt_boxes[idx][:gt_boxes_num_i, :]

            gt_boxes_lst, gt_keypoints_lst, gt_type_lst = [], [], []
            for gt_box_i in gt_boxes_i:
                # gt_bbox
                gt_boxes_lst.append(
                    {
                        "data": gt_box_i[:4].tolist(),
                    }
                )
                # gt_keypoints
                gt_kp_p1 = gt_box_i[4:6].tolist()
                gt_kp_p2 = gt_box_i[7:9].tolist()
                gt_keypoints_lst.append(
                    {
                        "data": [gt_kp_p1, gt_kp_p2],
                        "score": [gt_box_i[6], gt_box_i[-1]],
                    }
                )
                # Note: 0:normal, 1:hard, 2:ignore, all used 0
                gt_type_lst.append(0)
            gt_instance = {
                "bbox": gt_boxes_lst,
                "keypoints": gt_keypoints_lst,
                "type": gt_type_lst,
            }
            self.gt_annos.append(gt_instance)

            # 2. convert predictions, one image
            preds_parent_i = preds_parents[idx]
            preds_kps_i = preds_kps[idx]
            pred_boxes_lst, pred_keypoints_lst = [], []
            for det_parent, det_kps in zip(preds_parent_i, preds_kps_i):
                # bbox
                bbox_score = to_numpy(det_parent.score)
                if bbox_score < self.bbox_score_thresh:
                    continue
                pred_boxes_lst.append(
                    {
                        "data": to_numpy(det_parent.box),
                        "score": bbox_score,
                    }
                )
                # keypoint
                assert det_kps.num_points == self.num_kps
                point0, point1 = det_kps.point0, det_kps.point1
                left_pt = to_numpy(point0.point).tolist()
                right_pt = to_numpy(point1.point).tolist()
                left_pt_score = to_numpy(point0.score)
                right_pt_score = to_numpy(point1.score)

                pred_keypoints_lst.append(
                    {
                        "data": [left_pt, right_pt],
                        "score": [left_pt_score, right_pt_score],
                    }
                )
            pred_instance = {
                "bbox": pred_boxes_lst,
                "keypoints": pred_keypoints_lst,
            }

            self.predictions.append(pred_instance)

    def get(self):
        try:
            kps_eval = KeyPointsEval(
                gt_annos=self.gt_annos,
                predictions=self.predictions,
                iou_types=self.iou_types,
                num_kps=self.num_kps,
                overlap_threshes=self.overlap_threshes,
                remove_bbox_fp=self.remove_bbox_fp,
                remove_bbox_fn=self.remove_bbox_fn,
            )
        except IndexError:
            # invalid model may result in empty results, skip it
            return ["AP"], ["0.0"]

        ap, recall, prec, ar, results, pr_results = kps_eval.evaluate()

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
