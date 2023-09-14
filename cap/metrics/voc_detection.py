# Copyright (c) Changan Auto. All rights reserved.
from functools import reduce
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from changan_plugin_pytorch.nn.functional import batched_nms

from cap.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["VOCMApMetric", "VOC07MApMetric"]


@OBJECT_REGISTRY.register
class VOCMApMetric(EvalMetric):
    """Calculate mean AP for object detection task.

    Args:
        num_classes: Num classs.
        iou_thresh: IOU overlap threshold for TP
        class_names: if provided, will print out AP for each class
    """

    def __init__(
        self,
        num_classes: int,
        cls_idx_mapping: Optional[Mapping] = None,
        iou_thresh: float = 0.5,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        if cls_idx_mapping is None:
            self.cls_idx_mapping = {i: i for i in range(num_classes)}
        else:
            self.cls_idx_mapping = cls_idx_mapping
        if class_names is None:
            self.num = None
            name = "MeanAP"
        else:
            assert isinstance(class_names, (list, tuple))
            assert len(class_names) == num_classes
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.num = num + 1
            name = list(class_names) + ["mAP"]

        super(VOCMApMetric, self).__init__(name)

        self.reset()
        self.iou_thresh = iou_thresh
        self.class_names = class_names

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

            self.add_state(
                "_%d_score" % (cls),
                default=[],
                dist_reduce_fx="cap",
            )

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, "num", None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        super().reset()

    def compute(self):
        self._update()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return float("nan")
            else:
                return self.sum_metric / self.num_inst
        else:
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
        gt_classes(List): Each element of gt_classes is the bboxes' classes
            of an image. It's shape is (N).
        gt_difficult(List):Each element of gt_difficult is the bboxes'
            difficult flag of an image. It's shape is (N).
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

        outputs = [
            gt_bboxes[0].new_zeros((0, 6)) if len(pred) == 0 else pred
            for pred in outputs
        ]
        pred_bboxes = [pred[:, :4] for pred in outputs]
        pred_labels = [pred[:, 4] for pred in outputs]
        pred_scores = [pred[:, 5] for pred in outputs]

        # dedup gt
        for i, _ in enumerate(gt_bboxes):
            keep_idx = batched_nms(
                gt_bboxes[i],
                torch.zeros_like(gt_labels[i]),
                gt_labels[i],
                0.99,
            )
            gt_bboxes[i] = gt_bboxes[i][keep_idx]
            gt_labels[i] = gt_labels[i][keep_idx]

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

            for c, idx in self.cls_idx_mapping.items():
                pred_label[pred_label == c] = idx
                gt_label[gt_label == c] = idx

            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])
            else:
                gt_difficult = gt_difficult.flat[valid_gt]

            for ln in np.unique(
                np.concatenate((pred_label, gt_label)).astype(int)
            ):
                ln_match = []
                ln_score = []

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
                ln_score.extend(pred_score_l)
                pre_score = getattr(self, "_%d_score" % (ln))
                cur_score = pre_score + [torch.tensor(ln_score, device=device)]
                setattr(self, "_%d_score" % (ln), cur_score)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    ln_match.extend((0,) * pred_bbox_l.shape[0])
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

                gt_index = iou.argmax(axis=1)
                gt_index[iou.max(axis=1) < self.iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            ln_match.append(-1)
                        else:
                            if not selec[gt_idx]:
                                ln_match.append(1)
                            else:
                                ln_match.append(0)
                        selec[gt_idx] = True
                    else:
                        ln_match.append(0)

                pre_match = getattr(self, "_%d_match" % (ln))
                cur_match = pre_match + [torch.tensor(ln_match, device=device)]
                setattr(self, "_%d_match" % (ln), cur_match)

    def _update(self):
        """Update num_inst and sum_metric."""
        aps = []
        recall, precs = self._recall_prec()
        for lp, rec, prec in zip(range(len(precs)), recall, precs):
            ap = self._average_precision(rec, prec)
            aps.append(ap)
            if self.num is not None and lp < (self.num - 1):
                self.sum_metric[lp] = ap
                self.num_inst[lp] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)

    def _recall_prec(self):
        """Get recall and precision from internal records."""
        n_fg_class = self.num_classes
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for lk in range(self.num_classes):
            score_lk = getattr(self, "_%d_score" % (lk))
            match_lk = getattr(self, "_%d_match" % (lk))

            if isinstance(score_lk, list):
                score_lk = [
                    y.unsqueeze(0) if y.ndim == 0 else y for y in score_lk
                ]
                score_lk = reduce(lambda a, b: torch.cat([a, b]), score_lk)
            if isinstance(match_lk, list):
                match_lk = [
                    y.unsqueeze(0) if y.ndim == 0 else y for y in match_lk
                ]
                match_lk = reduce(lambda a, b: torch.cat([a, b]), match_lk)
            score_l = score_lk.cpu().numpy()
            match_l = match_lk.cpu().numpy()
            match_l = match_l.astype(np.int32)

            order = score_l.argsort()[::-1]
            match_l = match_l[order]

            tp = np.cumsum(match_l == 1)
            fp = np.cumsum(match_l == 0)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            with np.errstate(divide="ignore", invalid="ignore"):
                prec[lk] = tp / (fp + tp)
            # If n_pos[l] is 0, rec[l] is None.
            n_pos = self._n_pos[lk].cpu().numpy()
            if n_pos > 0:
                rec[lk] = tp / n_pos

        return rec, prec

    def _average_precision(self, rec, prec):
        """Calculate average precision.

        Args:
            rec (numpy.array): cumulated recall
            prec (numpy.array): cumulated precision
        Returns:
            ap as float
        """

        if rec is None or prec is None:
            return np.nan

        # append sentinel values at both ends
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], np.nan_to_num(prec), [0.0]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


@OBJECT_REGISTRY.register
class VOC07MApMetric(VOCMApMetric):
    """Mean average precision metric for PASCAL V0C 07 dataset.

    Args:
        num_classes: Num classs.
        iou_thresh: IOU overlap threshold for TP
        class_names: if provided, will print out AP for each class
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresh: float = 0.5,
        class_names: Optional[List[str]] = None,
    ):
        super(VOC07MApMetric, self).__init__(
            num_classes=num_classes,
            iou_thresh=iou_thresh,
            class_names=class_names,
        )

    def _average_precision(self, rec, prec):
        """Calculate average precision, override the default one.

           special 11-point metric

        Args:
            rec (numpy.array): cumulated recall
            prec (numpy.array): cumulated precision
        Returns:
            ap as float
        """

        if rec is None or prec is None:
            return np.nan
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / 11.0
        return ap
