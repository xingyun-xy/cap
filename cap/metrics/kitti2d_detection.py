# Copyright (c) Changan Auto. All rights reserved.
# This code is modified from the official c++ code, you can refer to
# http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
import json
import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from cap.data.transforms.functional_bbox import bbox_overlaps
from cap.metrics.metric import EvalMetric
from cap.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Kitti2DMetric(EvalMetric):
    """Kitti2D detection metric.

    For details, you can refer to
    http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d.

    Args:
        anno_file (str): validation data annotation json file path.
        name: name of this metric instance for display.
        is_plot: whether to plot the PR curve.
    """

    CLASS_NAMES = [
        "Car",
        "Van",
        "Pedestrian",
        "Person_sitting",
        "Cyclist",
        "DontCare",
    ]
    CLASSES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    MIN_OVERLAP = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
    N_SAMPLE_PTS = 41

    def __init__(
        self,
        anno_file: str,
        name: str = "kittiAP",
        is_plot: bool = True,
    ):
        super().__init__(name)
        self.gt_dict = {}
        self.pred_dict = {}
        self.anno_file = anno_file
        self.is_plot = is_plot
        with open(self.anno_file, "r") as f:
            for line in f.readlines():
                img_info = json.loads(line)
                record_list = []
                img_name = img_info["image_key"]
                for element in img_info["kitti_vehicle"]:
                    record = {}
                    class_name = element["attrs"]["type"]
                    if class_name not in self.CLASS_NAMES:
                        continue
                    record["class"] = class_name
                    record["trunc"] = float(element["attrs"]["truncated"])
                    record["occ"] = float(element["attrs"]["occluded"])
                    record["box"] = [float(c) for c in element["data"]]
                    record_list.append(record)
                self.gt_dict[img_name] = record_list

    def _init_states(self):
        self.pred_dict = {}

    def reset(self):
        self.pred_dict = {}

    def update(self, output: Dict):  # noqa: D205,D400
        """
        Args:
            output: A dict of model output which includes det results and
                image infos. Support batch_size >= 1
            output['pred_bboxes'] (List[torch.Tensor]): Network output
                for each input.
            output['img_name'] (List(str)): image name for each input.
        """
        dets = output["pred_bboxes"]
        img_names = output["img_name"]
        for idx, det in enumerate(dets):
            img_name = img_names[idx]
            det = det.cpu().numpy()
            pred_label = det[:, -1]
            pred_score = det[:, -2]
            pred_bbox = det[:, 0:4]
            record_list = []
            for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
                record = {}
                record["class"] = self.CLASSES[int(label)]
                record["score"] = float(score)
                record["box"] = [float(c) for c in bbox]
                record_list.append(record)
            self.pred_dict[img_name] = record_list

    def get_thresholds(self, scores: List[float], n_gt: int):
        # sort scores in descending order
        # (highest score is assumed to give best/most confident detections)
        scores = np.array(scores)
        sort_ind_desc = np.argsort(scores * -1)
        scores = scores[sort_ind_desc]

        # holds scores needed to compute N_SAMPLE_PTS recall values
        t = []

        # get scores for linearly spaced recall
        current_recall = 0

        for i in range(scores.shape[0]):
            # check if right-hand-side recall with respect to current
            # recall is close than left-hand-side one
            # in this case, skip the current detection score
            l_recall = (i + 1) / n_gt

            if i < scores.shape[0] - 1:
                r_recall = (i + 2) / n_gt
            else:
                r_recall = l_recall

            if (r_recall - current_recall) < (
                current_recall - l_recall
            ) and i < (scores.shape[0] - 1):
                continue

            # left recall is the best approximation, so use this and goto
            # next recall step for approximation
            # recall = l_recall

            # the next recall step was reached
            t.append(scores[i])
            current_recall += 1.0 / (self.N_SAMPLE_PTS - 1.0)
        return t

    def clean_data(
        self,
        gts: List[Dict],
        preds: List[Dict],
        cls: str,
        diff: int,
    ):
        ignored_gt = []
        ignored_pred = []
        dontcare = []

        n_gt = 0

        # extract ground truth bounding boxes for current evaluation class
        for gt in gts:
            # 1. neighboring classes are ignored("van" for "car"
            # and "person_sitting" for "pedestrian")
            # 2. (lower/upper cases are ignored)
            if cls == gt["class"]:
                # all classes without a neighboring class
                valid_class = 1
            else:
                # classes with a neighboring class
                if gt["class"] == "Van" and cls == "Car":
                    valid_class = 0
                elif gt["class"] == "Person_sitting" and cls == "Pedestrian":
                    valid_class = 0
                else:
                    # classes not used for evaluation
                    valid_class = -1
            # only bounding boxes with a minimum height are used for evaluation
            height = gt["box"][3] - gt["box"][1]

            # ground truth is ignored, if occlusion, truncation exceeds the
            # difficulty or ground truth is too small(doesn't count as
            # flase_negative nor true_positive, although detections
            # may be assigned)
            if (
                gt["occ"] > self.MAX_OCCLUSION[diff]
                or gt["trunc"] > self.MAX_TRUNCATION[diff]
                or height < self.MIN_HEIGHT[diff]
            ):
                ignore = True
            else:
                ignore = False

            # set ignored vector for ground truth
            # current class and not ignored (total no. of ground truth
            # is detected for recall denominator)
            if valid_class == 1 and not ignore:
                n_gt += 1
                ignored_gt.append(0)
            # neighboring class, or current class but ignored
            elif valid_class == 0 or (ignore and valid_class == 1):
                ignored_gt.append(1)
            # all other classes which are flase_negative in the evaluation
            else:
                ignored_gt.append(-1)

            # extract dontcare areas
            if gt["class"] == "DontCare":
                dontcare.append(True)
            else:
                dontcare.append(False)

        # extract detections bounding boxes of the current class
        for pred in preds:
            # neighboring classes are not evaluated
            if pred["class"] == cls:
                valid_class = 1
            else:
                valid_class = -1
            height = pred["box"][3] - pred["box"][1]

            # set ignored vector for detections
            if height < self.MIN_HEIGHT[diff]:
                ignored_pred.append(1)
            elif valid_class == 1:
                ignored_pred.append(0)
            else:
                ignored_pred.append(-1)
        return ignored_gt, dontcare, ignored_pred, n_gt

    def compute_statistics(
        self,
        gts: List[Dict],
        preds: List[Dict],
        dontcare: List[bool],
        ignored_gt: List[int],
        ignored_pred: List[int],
        compute_fp: bool,
        threshold: float,
        cls: str,
        diff: int,
    ):
        n_gt = len(gts)
        n_pred = len(preds)
        assert len(preds) == len(
            ignored_pred
        ), f"{len(preds)} vs {len(ignored_pred)}"

        # holds wether a detection was assigned to a valid
        # or ignored ground truth
        assigned_detection = [False for _ in range(n_pred)]
        true_positive, false_positive, flase_negative = 0, 0, 0
        det_scores = []
        # holds detections with a threshold lower
        # than thresh if false_positive are computed
        ignore_threshold = []

        # detections with a low score are ignored for
        # computing precision(needs false_positive)
        if compute_fp:
            for pred in preds:
                if pred["score"] < threshold:
                    ignore_threshold.append(True)
                else:
                    ignore_threshold.append(False)
        else:
            for _pred in preds:
                ignore_threshold.append(False)

        # evaluate all ground truth boxes
        for i in range(n_gt):
            # this ground truth is not of the current or a
            # neighboring class and therefore ignored
            if ignored_gt[i] == -1:
                continue
            # find candidates (overlap with ground truth > 0.5)
            # (logical len(det))
            det_idx = -1
            valid_detection = -1
            max_iou = 0.0
            # search for a possible detection
            assigned_ignored_det = False
            for j in range(n_pred):
                # detections not of the current class, already assigned
                # or with a low threshold are ignored
                if ignored_pred[j] == -1:
                    continue
                if assigned_detection[j]:
                    continue
                if ignore_threshold[j]:
                    continue

                # find the maximum score for the candidates and
                # get idx of respective detection
                iou = bbox_overlaps(
                    np.array([gts[i]["box"]]),
                    np.array([preds[i]["box"]]),
                )[0][0]
                # for computing recall thresholds, the candidate
                # with highest score is considered
                if (
                    not compute_fp
                    and iou > self.MIN_OVERLAP[cls]
                    and preds[j]["score"] > threshold
                ):
                    det_idx = j
                    valid_detection = preds[j]["score"]
                # for computing pr curve values, the candidate with the
                # greatest overlap is considered if the greatest overlap is
                # an ignored detection (min_height), the overlapping
                # detection is used
                elif (
                    compute_fp
                    and iou > self.MIN_OVERLAP[cls]
                    and (iou > max_iou or assigned_ignored_det)
                    and ignored_pred[j] == 0
                ):
                    max_iou = iou
                    det_idx = j
                    valid_detection = 1
                    assigned_ignored_det = False
                elif (
                    compute_fp
                    and iou > self.MIN_OVERLAP[cls]
                    and valid_detection == -1.0
                    and ignored_pred[j] == 1
                ):
                    det_idx = j
                    valid_detection = 1
                    assigned_ignored_det = True

            # compute true_positive, false_positive and flase_negative
            if valid_detection == -1 and ignored_gt[i] == 0:
                flase_negative += 1
            # only evaluate valid ground truth <=> detection
            # assignments (considering difficulty level)
            elif valid_detection != -1 and (
                ignored_gt[i] == 1 or ignored_pred[det_idx] == 1
            ):
                assigned_detection[det_idx] = True
            # found a valid true positive
            elif valid_detection != -1:
                # write highest score to threshold vector
                true_positive += 1
                det_scores.append(preds[det_idx]["score"])
                # clean up
                assigned_detection[det_idx] = True
        # if false_positive are requested, consider stuff area
        if compute_fp:
            for i in range(n_pred):
                # count false positives if required (height smaller than
                # required is ignored (ignored_det==1)
                if not (
                    assigned_detection[i]
                    or ignored_pred[i] == -1
                    or ignored_pred[i] == 1
                    or ignore_threshold[i]
                ):
                    false_positive += 1

            # do not consider detections overlapping with stuff area
            n_stuff = 0
            for i in range(n_gt):
                if not dontcare[i]:
                    continue
                for j in range(n_pred):
                    # detections not of the current class, already assigned,
                    # with a low threshold or a low minimum height are ignored
                    if assigned_detection[j]:
                        continue
                    if ignored_pred[j] == -1 or ignored_pred[j] == 1:
                        continue
                    if ignore_threshold[j]:
                        continue
                    # compute overlap and assign to stuff area,
                    # if overlap exceeds class specific value
                    iou = bbox_overlaps(
                        np.array([preds[i]["box"]]),
                        np.array([gts[i]["box"]]),
                        mode="iof",
                    )
                    if iou > self.MIN_OVERLAP[cls]:
                        assigned_detection[j] = True
                        n_stuff += 1

            # false_positive = no. of all not to ground truth assigned
            # detections - detections assigned to stuff areas
            false_positive -= n_stuff

        return true_positive, false_positive, flase_negative, det_scores

    def eval_class(
        self,
        gt_dict,
        pred_dict,
        cls,
        diff,
    ):
        ignored_gt_dict = {}
        ignored_pred_dict = {}
        dontcare_list = {}
        total_gt_num = 0  # total no. of gt (denominator of recall)

        # clean data
        det_scores = (
            []
        )  # detection scores, evaluated for recall discretization
        for key in gt_dict.keys():
            # holds ignored ground truth, ignored detections and
            # dontcare areas for current frame
            ignored_gt, dontcare, ignored_pred, n_gt_ = self.clean_data(
                gt_dict[key], pred_dict[key], cls, diff
            )
            ignored_gt_dict[key] = ignored_gt
            ignored_pred_dict[key] = ignored_pred
            dontcare_list[key] = dontcare
            total_gt_num += n_gt_
            # compute statistics to get recall values
            _, _, _, det_scores_ = self.compute_statistics(
                gt_dict[key],
                pred_dict[key],
                dontcare,
                ignored_gt,
                ignored_pred,
                False,
                0,
                cls,
                diff,
            )
            # add detection scores to vector over all images
            det_scores = det_scores + det_scores_
        # get scores that must be evaluated for recall discretization
        thresholds = self.get_thresholds(det_scores, total_gt_num)
        len_th = len(thresholds)
        true_positives = [0.0] * len_th
        false_positives = [0.0] * len_th
        flase_negatives = [0.0] * len_th
        # compute true_positive,false_positive,flase_negative
        # for relevant scores
        for key in gt_dict.keys():
            # for all scores/recall thresholds do
            for t, th in enumerate(thresholds):
                (
                    true_positive,
                    false_positive,
                    flase_negative,
                    _,
                ) = self.compute_statistics(
                    gt_dict[key],
                    pred_dict[key],
                    dontcare_list[key],
                    ignored_gt_dict[key],
                    ignored_pred_dict[key],
                    True,
                    th,
                    cls,
                    diff,
                )
                true_positives[t] += true_positive
                false_positives[t] += false_positive
                flase_negatives[t] += flase_negative
        # compute recall, precision
        precisions = [0.0] * self.N_SAMPLE_PTS
        recalls = []

        for t, _th in enumerate(thresholds):
            r = true_positives[t] / (true_positives[t] + flase_negatives[t])
            recalls.append(r)
            precisions[t] = true_positives[t] / (
                true_positives[t] + false_positives[t]
            )

        # filter precision using max_{i..end}(precision)
        for t, _th in enumerate(thresholds):
            precisions[t] = np.max(precisions[t:])

        return precisions, recalls

    def plot_and_compute(self, precisions, cls):
        if self.is_plot:
            xs = np.arange(0.0, 1.0, 1.0 / len(precisions[0]))

            l_easy = plt.plot(xs, precisions[0], c="green")[0]
            l_moderate = plt.plot(xs, precisions[1], c="blue")[0]
            l_hard = plt.plot(xs, precisions[2], c="red")[0]

            labels = ["Easy", "Moderate", "Hard"]
            plt.legend(
                handles=[l_easy, l_moderate, l_hard], labels=labels, loc="best"
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(cls)
            plt.ylim((0, 1.0))
            plt.grid()
            plt.savefig("2d_result.png")
            plt.show()
            plt.close()
        val_easy, val_moderate, val_hard = 0.0, 0.0, 0.0
        for i in range(0, self.N_SAMPLE_PTS, 4):
            val_easy += precisions[0][i]
            val_moderate += precisions[1][i]
            val_hard += precisions[2][i]

        ap_easy = 100.0 * val_easy / 11.0
        ap_moderate = 100.0 * val_moderate / 11.0
        ap_hard = 100.0 * val_hard / 11.0
        name = ([f"2D Detection AP for {cls}: ", "Easy", "Moderate", "Hard"],)
        value = (["", ap_easy, ap_moderate, ap_hard],)
        logging.info("2D Detection AP for %s\n" % cls)
        logging.info("Easy: %f" % ap_easy)
        logger.info("Moderate: %f" % ap_moderate)
        logger.info("Hard: %f" % ap_hard)
        return name, value

    def get(self):
        names = []
        values = []
        for _, cls in self.CLASSES.items():
            recall_all_diff = []
            precision_all_diff = []
            for diff in range(3):
                precisions, recalls = self.eval_class(
                    self.gt_dict, self.pred_dict, cls, diff
                )
                precision_all_diff.append(precisions)
                recall_all_diff.append(recalls)
            name, value = self.plot_and_compute(precision_all_diff, cls)
            names.extend(name)
            values.extend(value)
        return names, values
