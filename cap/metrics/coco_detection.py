# Copyright (c) Changan Auto. All rights reserved.
import datetime
import json
import logging
import os
import sys
from os import path as osp
from typing import Dict, Optional

import numpy as np

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info
from .metric import EvalMetric

__all__ = ["COCODetectionMetric"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class COCODetectionMetric(EvalMetric):
    """Evaluation in COCO protocol.

    Args:
        ann_file: validation data annotation json file path.
        val_interval: evaluation interval.
        name: name of this metric instance for display.
        save_prefix: path to save result.
        adas_eval_task: task name for adas-eval, such as 'vehicle', 'person'
            and so on.
        use_time: whether to use time for name.
        cleanup: whether to clean up the saved results when the process ends.

    Raises:
        RuntimeError: fail to write json to disk.

    """

    def __init__(
        self,
        ann_file: str,
        val_interval: int = 1,
        name: str = "COCOMeanAP",
        save_prefix: str = "./WORKSPACE/results",
        adas_eval_task: Optional[str] = None,
        use_time: bool = True,
        cleanup: bool = False,
    ):
        super().__init__(name)
        self.cleanup = cleanup

        self.coco = COCO(ann_file)
        self._img_ids = sorted(self.coco.getImgIds())
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.categories = sorted(self.categories, key=lambda x: x["id"])
        self.iter = 0
        self.val_interval = val_interval
        rank, world_size = get_dist_info()
        self.save_prefix = save_prefix + str(rank) + "_" + str(world_size)
        self.use_time = use_time
        self.adas_eval_task = adas_eval_task

        try:
            os.makedirs(osp.expanduser(self.save_prefix))
        except Exception:
            pass

        if use_time:
            t = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        else:
            t = ""
        self._filename = osp.abspath(
            osp.join(osp.expanduser(self.save_prefix), t + ".json")
        )
        try:
            f = open(self._filename, "w")
        except IOError as e:
            raise RuntimeError(
                "Unable to open json file to dump. What(): {}".format(str(e))
            )
        else:
            f.close()

    def _init_states(self):
        self._results = []

    def __del__(self):
        if self.cleanup:
            try:
                os.remove(self._filename)
            except IOError as err:
                logger.error(str(err))

    def reset(self):
        self._results = []

    def _update(self):
        """Use coco to get real scores."""
        if not self._results:
            # in case of empty results, push a dummy result
            self._results.append(
                {
                    "image_name": "fake_img",
                    "image_id": self._img_ids[0],
                    "category_id": 0,
                    "bbox": [0, 0, 0, 0],
                    "score": 0,
                }
            )
        try:
            # update self._filename
            if self.use_time:
                t = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            else:
                t = ""
            self._filename = osp.abspath(
                osp.join(osp.expanduser(self.save_prefix), t + ".json")
            )

            with open(self._filename, "w") as f:
                json.dump(self._results, f)
        except IOError as e:
            raise RuntimeError(
                "Unable to dump json file, ignored. What(): {}".format(str(e))
            )

        pred = self.coco.loadRes(self._filename)
        gt = self.coco
        coco_eval = COCOeval(gt, pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._coco_eval = coco_eval
        return coco_eval

    def get(self):
        """Get evaluation metrics."""

        # Metric printing adapted from detectron/json_dataset_evaluator.
        def _get_thr_ind(coco_eval, thr):
            ind = np.where(
                (coco_eval.params.iouThrs > thr - 1e-5)
                & (coco_eval.params.iouThrs < thr + 1e-5)
            )[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        # call real update
        try:
            coco_eval = self._update()
        except IndexError:
            # invalid model may result in empty JSON results, skip it
            return ["mAP"], ["0.0"]

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = coco_eval.eval["precision"][
            ind_lo : (ind_hi + 1), :, :, 0, 2
        ]
        ap_default = np.mean(precision[precision > -1])
        names, values = [], []
        names.append("~~~~ Summary metrics ~~~~\n")
        # catch coco print string, don't want directly print here
        _stdout = sys.stdout
        sys.stdout = StringIO()
        coco_eval.summarize()
        coco_summary = sys.stdout.getvalue()
        sys.stdout = _stdout
        values.append(str(coco_summary).strip())
        # per-category AP
        for cls_ind, cls_name in enumerate(self.categories):
            precision = coco_eval.eval["precision"][
                ind_lo : (ind_hi + 1), :, cls_ind, 0, 2
            ]
            ap = np.mean(precision[precision > -1])
            names.append(cls_name["name"])
            values.append("{:.1f}".format(100 * ap))
        # put mean AP at last, for comparing perf
        names.append(
            "~~~~ MeanAP @ IoU=[{:.2f},{:.2f}] ~~~~\n".format(
                IoU_lo_thresh, IoU_hi_thresh
            )
        )
        values.append("{:.1f}".format(100 * ap_default))
        names.append("mAP")
        values.append(100 * ap_default)
        if self.adas_eval_task is not None:
            assert isinstance(self.adas_eval_task, str)
            self.iter += self.val_interval
            self.save_adas_eval(save_iter=self.iter)

        log_info = ""
        for k, v in zip(names, values):
            if isinstance(v, (int, float)):
                log_info += "%s[%.4f] " % (k, v)
            else:
                log_info += "%s[%s] " % (str(k), str(v))
        logger.info(log_info)
        return names[-1], values[-1]

    def save_adas_eval(self, save_iter):
        adas_eval_results = []
        unique_image_id = []
        coco_results = json.load(open(self._filename, "r"))
        for line in coco_results:
            if line["image_id"] not in unique_image_id:
                cur_result = {}
                cur_result["image_key"] = line["image_name"]
                cur_result[self.adas_eval_task] = []
                bbox_dict = {}
                bbox_dict["bbox"] = line["bbox"]
                bbox_dict["bbox_score"] = line["score"]
                cur_result[self.adas_eval_task].append(bbox_dict)
                adas_eval_results.append(cur_result)
                unique_image_id.append(line["image_id"])
            else:
                bbox_dict = {}
                bbox_dict["bbox"] = line["bbox"]
                bbox_dict["bbox_score"] = line["score"]
                adas_eval_results[line["image_id"]][
                    self.adas_eval_task
                ].append(bbox_dict)

        save_path = os.path.split(self._filename)[0]
        save_path = os.path.join(
            save_path, self.adas_eval_task + "_" + str(save_iter) + ".json"
        )
        save_file = open(save_path, "w")
        for line in adas_eval_results:
            save_file.write(json.dumps(line) + "\n")
        save_file.close()

    def update(self, output: Dict):
        """Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            output: A dict of model output which includes det results and
                image infos.

        """
        dets = output["pred_bboxes"]
        for idx, det in enumerate(dets):
            det = det.cpu().numpy()
            pred_label = det[:, -1]
            pred_score = det[:, -2]
            pred_bbox = det[:, 0:4]
            # convert [xmin, ymin, xmax, ymax] to original coordinates
            if "scale_factor" in output:
                pred_bbox = (
                    pred_bbox / output["scale_factor"][idx].cpu().numpy()
                )
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :].astype(np.float)
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred].astype(np.float)
            # for each bbox detection in each image
            for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
                category_id = self.coco.getCatIds()[label]  # label is 0-based
                # convert [xmin, ymin, xmax, ymax] to [xmin, ymin, w, h]
                bbox[2:4] -= bbox[:2]
                self._results.append(
                    {
                        "image_name": "%012d" % output["img_id"][idx][0].item()
                        + ".jpg",
                        "image_id": output["img_id"][idx][0].item(),
                        "category_id": category_id,
                        "bbox": bbox.tolist(),
                        "score": score,
                    }
                )
