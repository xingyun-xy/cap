# Copyright (c) Changan Auto. All rights reserved.
import enum
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import torch

from cap.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["RecallPrecision"]


@enum.unique
class PredBboxState(enum.Enum):
    FP = 0
    TP = 1
    SUB_TP = 2
    IGNORE = -1


@enum.unique
class GtBboxState(enum.Enum):
    UN_MATCHED = 0
    MATCHED = 1


@OBJECT_REGISTRY.register
class RecallPrecision(EvalMetric):
    """Calculate recall / precision on score threshold.

    Args:
        score_thresh: Score threshold for calculate recall/precision.
        num_classes: Number of classes, class index is start from 0,
            which means there should has no background in predict
            result.
        class_names: if provided, will print out recall/precision
            for each class.
        iou_thresh: IOU overlap threshold for TP,
            default to 0.5;
        keep_all_matched_bbox_as_tp: Usually, only one predict bbox
            matched one gt is True Positive, other predict bboxes matched
            to this same gt bbox will be assigned to False Positive
            (keep_all_matched_bbox_as_tp=False),
            but when evaluation rpn, we need treat all matched gt predict
            bboxes as TP, only no matched with gt bboxes are treated as
            FP (keep_all_matched_bbox_as_tp=True).
            Default to False.
    """

    def __init__(
        self,
        score_thresh: float,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_thresh: float = 0.5,
        keep_all_matched_bbox_as_tp: bool = False,
    ):
        self.num_classes = num_classes
        if class_names is None:
            self.num = None
            name = [
                "Mean_" + str(score_thresh) + "_R",
                "Mean_" + str(score_thresh) + "_P",
            ]
        else:
            assert isinstance(class_names, (list, tuple))
            assert len(class_names) == num_classes
            for cls_name in class_names:
                assert isinstance(
                    cls_name, str
                ), "must provide class names as str"
            num = len(class_names)
            self.num = num + 1
            name = []
            for cls_name in class_names:
                name.append(str(cls_name) + "_" + str(score_thresh) + "_R")
                name.append(str(cls_name) + "_" + str(score_thresh) + "_P")
            name.append("Mean_" + str(score_thresh) + "_R")
            name.append("Mean_" + str(score_thresh) + "_P")

        super(RecallPrecision, self).__init__(name)
        self.reset()

        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.class_names = class_names
        self.keep_all_matched_bbox_as_tp = keep_all_matched_bbox_as_tp

    def _init_states(self):
        self.add_state(
            "_n_pos",
            default=torch.zeros(self.num_classes),
            dist_reduce_fx="sum",
        )
        for cls in range(self.num_classes):
            self.add_state(
                "_%d_match" % (cls),
                default=[],
                dist_reduce_fx="cap",
            )

    def reset(self):
        """Clear the internal statistics to initial state."""
        if self.num is None:
            # mean recall, precision
            self.num_inst = [0] * 2
            self.sum_metric = [0.0] * 2
        else:
            # num == (num class + mean)
            self.num_inst = [0] * self.num * 2
            self.sum_metric = [0.0] * self.num * 2
        super().reset()

    def compute(self):
        self._update()  # update metric at this time
        values = [
            x / y if y != 0 else float("nan")
            for x, y in zip(self.sum_metric, self.num_inst)
        ]
        return values

    def update(self, model_outs: Dict):
        """model_outs is a dict, the meaning of it's key is as following.

        pred_bboxes(List): Each element of pred_bboxes is the predict result
            of an image. It's shape is (N, 6).
            6 means (x1,y1,x2,y2,label,score)
        gt_bboxes(List): Each element of gt_bboxes is the bboxes' coordinates
            of an image. It's shape is (N, 4).
            4 means (x1,y1,x2,y2)
        gt_classes(List): Each element of gt_classes is the bboxes' classes.
            It's shape is (N), which equal to the number of gt_bboxes.
        gt_difficult(List): Each element of gt_difficult is the bboxes'
            difficult flag. It's shape is (N), which equal to the number
            of gt_bboxes.
        """

        def as_numpy(a):
            """Convert a (list of) torch.Tensor into numpy.ndarray."""
            if isinstance(a, (list, tuple)):
                out = [
                    x.cpu().detach().numpy()
                    if isinstance(x, torch.Tensor)
                    else x
                    for x in a
                ]
                return out
            elif isinstance(a, torch.Tensor):
                a = a.cpu().detach().numpy()
            return a

        outputs = model_outs["pred_bboxes"]
        gt_bboxes = model_outs["gt_bboxes"]
        gt_labels = model_outs["gt_classes"]
        gt_difficults = model_outs["gt_difficult"]

        outputs = [pred[pred[:, 5] >= self.score_thresh] for pred in outputs]
        pred_bboxes = [pred[:, :4] for pred in outputs]
        pred_labels = [pred[:, 4] for pred in outputs]
        pred_scores = [pred[:, 5] for pred in outputs]

        if gt_difficults is None:
            gt_difficults = [None for _ in as_numpy(gt_labels)]

        for (
            pred_bbox,
            pred_label,
            pred_score,
            gt_bbox,
            gt_label,
            gt_difficult,
        ) in zip(
            *[
                as_numpy(x)
                for x in [
                    pred_bboxes,
                    pred_labels,
                    pred_scores,
                    gt_bboxes,
                    gt_labels,
                    gt_difficults,
                ]
            ]
        ):
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]
            valid_gt = np.where(gt_label.flat >= 0)[0]
            gt_bbox = gt_bbox[valid_gt, :]
            gt_label = gt_label.flat[valid_gt].astype(int)

            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])
            else:
                gt_difficult = gt_difficult.flat[valid_gt]

            for ln in np.unique(
                np.concatenate((pred_label, gt_label)).astype(int)
            ):
                ln_match = []

                pred_mask_l = pred_label == ln
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == ln
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                self._n_pos[ln] += np.logical_not(gt_difficult_l).sum()
                device = self._n_pos.device

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    ln_match.extend(
                        (PredBboxState.FP.value,) * pred_bbox_l.shape[0]
                    )
                    pre_match = getattr(self, "_%d_match" % (ln))
                    cur_match = pre_match + [
                        torch.tensor(ln_match, device=device)
                    ]
                    setattr(self, "_%d_match" % (ln), cur_match)
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                bbox_a = pred_bbox_l.copy()
                bbox_a[:, 2:] += 1
                bbox_b = gt_bbox_l.copy()
                bbox_b[:, 2:] += 1

                tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
                br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

                area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
                area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2], axis=1)
                area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2], axis=1)
                iou = area_i / (area_a[:, None] + area_b - area_i)

                gt_status = GtBboxState.UN_MATCHED.value * np.ones(
                    gt_bbox_l.shape[0], dtype=np.int
                )

                for i in range(pred_bbox_l.shape[0]):
                    match_norm_gt_indx = -1
                    max_norm_overlap = -1
                    match_ignore_gt_indx = -1
                    max_ignore_overlap = -1
                    for j in range(gt_bbox_l.shape[0]):
                        if gt_status[j] == GtBboxState.MATCHED.value:
                            continue

                        if iou[i, j] < self.iou_thresh:
                            continue

                        ov_ij = iou[i, j]
                        if gt_difficult_l[j]:
                            if ov_ij > max_ignore_overlap:
                                max_ignore_overlap = ov_ij
                                match_ignore_gt_indx = j
                        else:
                            if ov_ij > max_norm_overlap:
                                max_norm_overlap = ov_ij
                                match_norm_gt_indx = j

                    if match_norm_gt_indx >= 0:
                        gt_status[
                            match_norm_gt_indx
                        ] = GtBboxState.MATCHED.value
                        ln_match.append(PredBboxState.TP.value)
                    elif match_ignore_gt_indx >= 0:
                        gt_status[
                            match_ignore_gt_indx
                        ] = GtBboxState.MATCHED.value
                        ln_match.append(PredBboxState.IGNORE.value)
                    else:
                        ln_match.append(PredBboxState.FP.value)

                if self.keep_all_matched_bbox_as_tp:
                    for i in range(pred_bbox_l.shape[0]):
                        match_list_indx = i - pred_bbox_l.shape[0]
                        if ln_match[match_list_indx] == PredBboxState.TP.value:
                            continue

                        match_norm_gt_indx = -1
                        match_ignore_gt_indx = -1
                        for j in range(gt_bbox_l.shape[0]):
                            if iou[i, j] < self.iou_thresh:
                                continue

                            if gt_difficult_l[j]:
                                match_ignore_gt_indx = j
                            else:
                                match_norm_gt_indx = j
                                break

                        if match_norm_gt_indx >= 0:
                            ln_match[
                                match_list_indx
                            ] = PredBboxState.SUB_TP.value
                        elif (
                            ln_match[match_list_indx] == PredBboxState.FP.value
                        ) and (match_ignore_gt_indx >= 0):
                            ln_match[
                                match_list_indx
                            ] = PredBboxState.IGNORE.value

                pre_match = getattr(self, "_%d_match" % (ln))
                cur_match = pre_match + [torch.tensor(ln_match, device=device)]
                setattr(self, "_%d_match" % (ln), cur_match)

    def _update(self):
        """Update num_inst and sum_metric."""
        recall, precs = self._recall_prec()
        for lp, rec, prec in zip(range(len(precs)), recall, precs):
            if self.num is not None and lp < (self.num - 1):
                self.sum_metric[2 * lp + 0] = rec
                self.sum_metric[2 * lp + 1] = prec
                self.num_inst[2 * lp + 0] = 1
                self.num_inst[2 * lp + 1] = 1
        if self.num is None:
            self.num_inst = [1, 1]
            self.sum_metric[0] = np.nanmean(recall)
            self.sum_metric[1] = np.nanmean(precs)
        else:
            self.num_inst[-2] = 1
            self.sum_metric[-2] = np.nanmean(recall)
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(precs)

    def _recall_prec(self):
        """Get recall and precision from internal records."""
        n_fg_class = self.num_classes
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for lk in range(self.num_classes):
            match_lk = getattr(self, "_%d_match" % (lk))

            if isinstance(match_lk, list):
                match_lk = [
                    y.unsqueeze(0) if y.ndim == 0 else y for y in match_lk
                ]
                match_lk = reduce(lambda a, b: torch.cat([a, b]), match_lk)

            match_l = match_lk.cpu().numpy()
            match_l = match_l.astype(np.int32)

            tp = np.sum(match_l == PredBboxState.TP.value)
            fp = np.sum(match_l == PredBboxState.FP.value)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            with np.errstate(divide="ignore", invalid="ignore"):
                if self.keep_all_matched_bbox_as_tp:
                    opt_tp = np.sum(match_l == PredBboxState.SUB_TP.value)
                    prec[lk] = (tp + opt_tp) / (fp + tp + opt_tp + 1e-10)
                else:
                    prec[lk] = tp / (fp + tp + 1e-10)
            # If n_pos[l] is 0, rec[l] is None.
            n_pos = self._n_pos[lk].cpu().numpy()
            if n_pos > 0:
                rec[lk] = tp / n_pos

        return rec, prec
