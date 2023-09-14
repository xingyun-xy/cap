import logging
import multiprocessing
from collections import defaultdict
from typing import Sequence

import numpy as np
from pycocotools.mask import decode
from shapely.geometry import Polygon

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import convert_numpy
from .metric import EvalMetric

__all__ = ["Detection3DMetric"]

logger = logging.getLogger(__name__)


def mioa_ignore(bbox2d, ignore_mask, thresh):
    if ignore_mask is None:
        return False
    bbox2d = np.array(bbox2d)
    ignore_mask = decode(ignore_mask).astype(np.uint8)
    mask = np.zeros_like(ignore_mask).astype(np.uint8)
    bbox2d = np.clip(
        bbox2d,
        [0, 0, 0, 0],
        [mask.shape[1], mask.shape[0], mask.shape[1], mask.shape[0]],
    )  # noqa
    mask[
        int(bbox2d[1]) : int(bbox2d[3] + 1),
        int(bbox2d[0]) : int(bbox2d[2] + 1),
    ] = 1  # noqa
    mask = mask * ignore_mask
    if (
        len(np.where(mask == 1)[0])
        > (bbox2d[2] - bbox2d[0] + 1) * (bbox2d[3] - bbox2d[1] + 1) * thresh
    ):  # noqa
        return True
    else:
        return False


class Box3D:
    def __init__(self, is_gt, enable_ignore=False, **kwargs):
        required_attrs = ["location", "dimensions", "rotation_y", "depth"]
        if enable_ignore:
            required_attrs += ["ignore"]
        for name in required_attrs:
            assert name in kwargs, "{} error!".format(name)
        if not is_gt:
            assert "score" in kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            if enable_ignore and k == "ignore":
                assert v in [True, False]

        assert len(self.location) == 3 and not np.any(np.isnan(self.location))
        assert (
            len(self.dimensions) == 3
            and not np.any(np.isnan(self.dimensions))
            and np.all([x > 0 for x in self.dimensions])
        )  # noqa

        assert not np.any(np.isnan(self.rotation_y))
        assert not np.any(np.isnan(self.depth))

        self.volume = np.prod(self.dimensions)
        self.height, self.width, self.length = self.dimensions

        self.center_x, self.center_y, self.center_z = self.location

        self.min_y = self.center_y - self.height
        self.max_y = self.center_y

        self.compute_ground_corners()

    def compute_ground_corners(self):
        # dimensions: 3
        # location: 3
        # rotation_y: 1
        # return: 8 x 3
        c, s = np.cos(self.rotation_y), np.sin(self.rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        h, w, l = self.dimensions  # noqa
        x_corners = [l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [-h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2]

        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners)
        corners_3d = corners_3d + np.array(
            self.location, dtype=np.float32
        ).reshape(3, 1)
        corners_3d = corners_3d.transpose(1, 0)
        ground_corners = corners_3d[:, [0, 2]]

        self.ground_bbox_coords = ground_corners
        self.ground_bbox_polygon = Polygon(
            [
                (ground_corners[0, 0], ground_corners[0, 1]),
                (ground_corners[1, 0], ground_corners[1, 1]),
                (ground_corners[2, 0], ground_corners[2, 1]),
                (ground_corners[3, 0], ground_corners[3, 1]),
            ]
        )

        assert -np.pi <= self.rotation_y <= np.pi
        angle = np.round(self.rotation_y / (np.pi / 2.0)) * np.pi / 2
        angle -= self.rotation_y
        c1, s1 = np.cos(angle), np.sin(angle)
        R1 = np.array([[c1, s1], [-s1, c1]], dtype=np.float32)
        self.ground_axis_aligned_box_coords = np.dot(
            R1, ground_corners.transpose(1, 0)
        ).transpose(1, 0)

    def get_height_intersection(self, other):
        min_y = max(other.min_y, self.min_y)
        max_y = min(other.max_y, self.max_y)

        return max(0, max_y - min_y)

    def get_area_intersection(self, other):
        result = self.ground_bbox_polygon.intersection(
            other.ground_bbox_polygon
        ).area
        assert result <= self.width * self.length

        return result

    def get_intersection(self, other):
        height_intersection = self.get_height_intersection(other)
        area_intersection = self.ground_bbox_polygon.intersection(
            other.ground_bbox_polygon
        ).area

        return height_intersection * area_intersection

    def get_iou(self, other):
        intersection = self.get_intersection(other)
        union = self.volume + other.volume - intersection
        iou = np.clip(intersection / union, 0, 1)
        return iou

    def get_axis_aligned_iou(self, other):
        xmin = max(
            np.amin(self.ground_axis_aligned_box_coords[:, 0]),
            np.amin(other.ground_axis_aligned_box_coords[:, 0]),
        )
        ymin = max(
            np.amin(self.ground_axis_aligned_box_coords[:, 1]),
            np.amin(other.ground_axis_aligned_box_coords[:, 1]),
        )
        xmax = min(
            np.amax(self.ground_axis_aligned_box_coords[:, 0]),
            np.amax(other.ground_axis_aligned_box_coords[:, 0]),
        )
        ymax = min(
            np.amax(self.ground_axis_aligned_box_coords[:, 1]),
            np.amax(other.ground_axis_aligned_box_coords[:, 1]),
        )
        intersection = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
        area1 = self.width * self.length
        area2 = other.width * other.length
        iou2d = intersection / (float(area1 + area2 - intersection + 1e-8))
        height_intersection = self.get_height_intersection(other)

        return iou2d * height_intersection


class NuscEvaluator:
    def __init__(
        self,
        iou_threshold,
        target_precisions,
        target_recalls,
        enable_ignore,
        num_workers,
    ):
        self._iou_threshold = iou_threshold
        self._target_precisions = target_precisions
        self._target_recalls = target_recalls
        self._num_workers = num_workers
        self._pool = multiprocessing.Pool(num_workers)
        self._enable_ignore = enable_ignore

    def __del__(self):
        self._pool.terminate()

    def __call__(self, gt, pred):

        gt_by_img = defaultdict(lambda: [])
        pred_by_img = defaultdict(lambda: [])
        gt_count = 0
        gt_objs, pred_objs = [], []

        for _gt in gt:
            image_key = _gt["image_key"]
            if self._enable_ignore:
                if not _gt["ignore"]:
                    gt_count += 1
            else:
                gt_count += 1
            gt_by_img[image_key] += [_gt]
            gt_objs += [_gt]

        for _pred in pred:
            image_key = _pred["image_key"]
            pred_by_img[image_key] += [_pred]
            pred_objs += [_pred]

        group_by_img = {
            k: {"gt": gt_by_img[k], "pred": pred_by_img[k]}
            for k in gt_by_img.keys()
        }

        res_by_img = []
        for v in group_by_img.values():
            res_by_img += [
                self._pool.apply_async(
                    self.get_tp_fp,
                    [
                        v["gt"],
                        v["pred"],
                        self._iou_threshold,
                        self._enable_ignore,
                    ],
                )
            ]

        res = self.summarize(res_by_img, gt_count)
        return res

    @staticmethod
    def get_tp_fp(gt_list, pred_list, iou_threshold, enable_ignore=False):
        gt = [
            Box3D(is_gt=True, enable_ignore=enable_ignore, **j)
            for j in gt_list
        ]
        pred = [Box3D(is_gt=False, **j) for j in pred_list]
        pred = sorted(pred, key=lambda x: x.score, reverse=True)

        num_pred = len(pred)
        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        conf = np.zeros(num_pred)
        gt_checked = np.zeros(len(gt))
        ignored_mask = np.zeros(num_pred)

        for pidx, det in enumerate(pred):
            conf[pidx] = det.score
            max_overlap = -np.inf
            jmax = -1

            if len(gt) > 0:
                overlaps = [det.get_iou(g) for g in gt]
                max_overlap = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if max_overlap > iou_threshold:
                if enable_ignore and gt[jmax].ignore:
                    ignored_mask[pidx] = 1
                    continue
                if gt_checked[jmax] == 0:
                    tp[pidx] = 1.0
                    gt_checked[jmax] = 1
                else:
                    fp[pidx] = 1.0
            else:
                fp[pidx] = 1.0

        if enable_ignore:
            valid_mask = (1 - ignored_mask).astype(bool)
            tp = tp[valid_mask]
            fp = fp[valid_mask]
            conf = conf[valid_mask]
        ret = {"tp": tp, "fp": fp, "conf": conf}

        return ret

    def summarize(self, res_by_img, gt_count):
        res_all = defaultdict(lambda: [])
        num_image = len(res_by_img)
        for res in res_by_img:
            res = res.get()
            res_all["tp"] += [res["tp"]]
            res_all["fp"] += [res["fp"]]
            res_all["conf"] += [res["conf"]]

        tp = np.concatenate(res_all["tp"])
        fp = np.concatenate(res_all["fp"])
        conf = np.concatenate(res_all["conf"])

        argsort = np.argsort(-conf)
        conf = conf[argsort]
        tp = tp[argsort]
        fp = fp[argsort]

        fp = np.cumsum(fp, axis=0)
        tp = np.cumsum(tp, axis=0)

        if gt_count == 0:
            recalls = np.zeros(tp.shape)
        else:
            recalls = tp / float(gt_count)
        assert np.all(0 <= recalls) & np.all(recalls <= 1)

        precisions = tp / (tp + fp + 1e-6)
        assert np.all(0 <= precisions) & np.all(precisions <= 1)
        ap, recall, precision = calap(recalls, precisions)
        recall = np.array(recall)
        precision = np.array(precision)

        fppi = fp / float(num_image)
        fppi += 1e-6
        detection_rate = (tp - fp) / (gt_count + 1e-6)
        ar = calar(fppi, recalls)

        results = {}
        results["aps"] = {
            "ap": ap,
            "rec": recall[-1] if len(recall) > 0 else np.inf,
            "detection_rate": detection_rate[-1]
            if len(detection_rate) > 0
            else np.inf,
            "precision": precision[-1] if len(precision) > 0 else np.inf,
            "ar": ar,
            "num_image": num_image,
            "num_gt": gt_count,
            "num_tp": tp[-1] if len(tp) else 0,
            "num_fp": fp[-1] if len(fp) else 0,
        }
        results["recall"] = recall.tolist()
        results["precision"] = precision.tolist()
        results["fppi"] = fppi.tolist()
        results["conf"] = conf.tolist()
        results["detection_rate"] = detection_rate.tolist()
        results["fp"] = fp.tolist()

        results["scores_lut"] = {}
        thres_rec_pre_tp_fp_gt = []
        for thres in np.unique(conf):
            select = conf >= thres
            sub_tp = np.max(tp[select])
            sub_fp = np.max(fp[select])
            rec = sub_tp / (gt_count + 1e-6)
            pre = sub_tp / (sub_tp + sub_fp + 1e-6)
            thres_rec_pre_tp_fp_gt.append(
                [thres, rec, pre, sub_tp, sub_fp, gt_count]
            )
            results["scores_lut"][float(thres)] = {
                "threshold": thres,
                "recall": rec,
                "precision": pre,
                "num_tp": sub_tp,
                "num_fp": sub_fp,
                "num_gt": gt_count,
            }
        thres_rec_pre_tp_fp_gt = np.array(thres_rec_pre_tp_fp_gt)

        results["target_recalls"] = {}
        if len(thres_rec_pre_tp_fp_gt) == 0:
            pass
        else:
            for target_recall in self._target_recalls:
                valid_idx = np.where(
                    thres_rec_pre_tp_fp_gt[:, 1] > target_recall
                )[0]
                if valid_idx.shape[0] > 0:
                    idx = valid_idx[
                        thres_rec_pre_tp_fp_gt[valid_idx, 1].argmin()
                    ]
                else:
                    idx = np.argmin(
                        target_recall - thres_rec_pre_tp_fp_gt[:, 1]
                    )
                results["target_recalls"][target_recall] = {
                    "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                    "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                    "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                    "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                    "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                    "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
                }
            results["target_precisions"] = {}
            for target_precision in self._target_precisions:
                valid_idx = np.where(
                    thres_rec_pre_tp_fp_gt[:, 2] > target_precision
                )[0]
                if valid_idx.shape[0] > 0:
                    idx = valid_idx[
                        thres_rec_pre_tp_fp_gt[valid_idx, 2].argmin()
                    ]
                else:
                    idx = np.argmin(
                        target_precision - thres_rec_pre_tp_fp_gt[:, 2]
                    )
                results["target_precisions"][target_precision] = {
                    "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                    "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                    "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                    "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                    "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                    "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
                }

        summary = {"AP": ap, "AR": ar, "all_results": results}
        return summary


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


class NuScenes(object):
    def __init__(
        self,
        gt_annos,
        target_recalls,
        target_precisions,
        max_depth=300,
        iou_threshold=0.5,
        mioa_theshold=0.6,
        enable_ignore=False,
        num_workers=1,
    ):

        self.gt_annos = gt_annos
        self.max_depth = max_depth
        self.mioa_thresh = mioa_theshold
        self.evaluator = NuscEvaluator(
            iou_threshold=iou_threshold,
            target_precisions=target_precisions,
            target_recalls=target_recalls,
            enable_ignore=enable_ignore,
            num_workers=num_workers,
        )

    def __call__(self, pred_list):
        # filter gt
        all_gt = []
        key_ignore_mask = {}
        for gt in self.gt_annos:
            for det in gt["data"]:
                if gt["image_key"] not in key_ignore_mask:
                    key_ignore_mask[gt["image_key"]] = (
                        gt["ignore_mask"] if "ignore_mask" in gt else None
                    )  # noqa

                det["ignore"] = det["depth"] >= self.max_depth
                det["image_key"] = gt["image_key"]
                all_gt.append(det)

        all_pred = []
        for pred in pred_list:
            for det in pred["data"]:
                bbox2d = (
                    det["bbox_2d"] if "bbox_2d" in det.keys() else det["bbox"]
                )  # noqa
                if bbox2d is None:
                    continue
                if (
                    np.all(np.array(det["dimensions"]) > 0)
                    and det["depth"] < self.max_depth
                ):  # noqa
                    if pred[
                        "image_key"
                    ] in key_ignore_mask.keys() and mioa_ignore(
                        bbox2d,
                        key_ignore_mask[pred["image_key"]],
                        self.mioa_thresh,
                    ):  # noqa
                        continue

                    det["image_key"] = pred["image_key"]
                    all_pred.append(det)
        res = self.evaluator(all_gt, all_pred)

        # write results json
        results = res["all_results"]

        return results


@OBJECT_REGISTRY.register
class Detection3DMetric(EvalMetric):
    """Image 3d detection metric."""

    def __init__(
        self,
        max_depth: float,
        iou_threshold: float,
        target_recalls: Sequence[float] = (0.5, 0.6, 0.7),
        target_precisions: Sequence[float] = (0.7, 0.8, 0.9),
        name: str = "Detection3DMetric",
    ):
        self._index = 0
        self.max_depth = max_depth
        self.iou_threshold = iou_threshold
        self.target_recalls = target_recalls
        self.target_precisions = target_precisions
        super().__init__(name)

    def _init_states(self):
        self.predictions = []
        self.gt_annos = []

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
        gt_boxes = convert_numpy(batch_data["gt_boxes"])
        batch_size = gt_boxes.shape[0]

        # predictions
        preds_parents = preds["detection"]
        preds_3d = preds["det_3d"]

        assert len(preds_parents) == len(preds_3d) == batch_size

        for idx in range(batch_size):
            # 1. convert groundtruth
            gt_boxes_i = gt_boxes[idx]
            gt_boxes_num_i = gt_boxes_num[idx]

            gt_boxes_i = gt_boxes_i[:gt_boxes_num_i, :]
            gt_boxes_lst = []
            for box in gt_boxes_i:
                gt_boxes_lst.append(
                    {
                        "bbox": box[:4],
                        "depth": box[-2],
                        "dimensions": box[-7:-4],
                        "location": box[-4:-1],
                        "rotation_y": box[-1],
                    }
                )

            ignore_mask = {
                "size": (
                    batch_data["ignore_mask"]["size"][0][idx].item(),
                    batch_data["ignore_mask"]["size"][1][idx].item(),
                ),
                "counts": batch_data["ignore_mask"]["counts"][idx],
            }

            self.gt_annos.append(
                {
                    "data": gt_boxes_lst,
                    "image_key": self._index + idx,
                    "ignore_mask": ignore_mask,
                }
            )

            # 2. convert pred
            pred_parents = preds_parents[idx]
            pred_3d = preds_3d[idx]
            pred_bbox_lst = []

            boxes = convert_numpy(pred_parents.boxes)
            locations = convert_numpy(pred_3d.locations)
            dimensions = convert_numpy(pred_3d.dimensions)
            yaws = convert_numpy(pred_3d.yaw)
            scores = convert_numpy(pred_3d.scores * pred_parents.scores)

            for box, loc, dim, rot, score in zip(
                boxes, locations, dimensions, yaws, scores
            ):
                pred_bbox_lst.append(
                    {
                        "bbox": box,
                        "depth": loc[-1],
                        "dimensions": dim,
                        "location": loc,
                        "rotation_y": rot,
                        "score": score,
                    }
                )

            self.predictions.append(
                {
                    "data": pred_bbox_lst,
                    "image_key": self._index + idx,
                }
            )

        self._index += batch_size

    def get(self):
        try:
            det3d_eval = NuScenes(
                self.gt_annos,
                self.target_recalls,
                self.target_precisions,
                max_depth=self.max_depth,
                iou_threshold=self.iou_threshold,
                enable_ignore=True,
            )
        except IndexError:
            return [], []

        results = det3d_eval(self.predictions)

        # names = ["ap", "ar"]
        # values = [ap, ar]

        # for k in ["num_gt", "num_tp", "num_fp"]:
        #     names.append(k)
        #     values.append(pr_results["aps"][k])

        # log_info = "\n"
        # for k, v in zip(names, values):
        #     if isinstance(v, (int, float)):
        #         log_info += "%s: [%.4f] \n" % (k, v)
        #     else:
        #         log_info += "%s: [%s] \n" % (str(k), str(v))
        logger.info(results)

        return ["AP"], [results["aps"]["ap"]]
