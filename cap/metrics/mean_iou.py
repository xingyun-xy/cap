# Copyright (c) Changan Auto. All rights reserved.

import logging
from typing import List, Sequence, Union

import numpy as np
import torch

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list

from .metric import EvalMetric

__all__ = ["MeanIOU"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class MeanIOU(EvalMetric):
    """Evaluation segmentation results.

    Args:
        seg_class(list(str)): A list of classes the segmentation dataset
            includes，the order should be the same as the label.
        name(str): Name of this metric instance for display, also used as
            monitor params for Checkpoint.
        ignore_index(int): The label index that will be ignored in evaluation.
        global_ignore_index(list,int): The label index that will be ignored in
            global evaluation,such as:mIoU,mAcc,aAcc.Supporting list of label
            index.
        verbose(bool):  Whether to return verbose value for sda eval, default
            is False.

    """

    def __init__(
        self,
        seg_class: List[str],
        name: str = "MeanIOU",
        ignore_index: int = 255,
        global_ignore_index: Union[Sequence, int] = 255,
        verbose: bool = False,
        dist_sync_on_step: bool = False,
    ):
        self.num_classes = len(seg_class)
        self.seg_class = seg_class
        self.name = name
        super(MeanIOU, self).__init__(name)
        self.ignore_index = ignore_index
        self.global_save_index_list = [
            index
            for index in range(len(seg_class))
            if index not in _as_list(global_ignore_index)
        ]
        self.verbose = verbose
        self.dist_sync_on_step = dist_sync_on_step

    def _init_states(self):

        self.add_state(
            "intersect",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "union",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "label",
            default=torch.zeros((self.num_classes,)),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        label: torch.Tensor,
        preds: Union[Sequence[torch.Tensor], torch.Tensor],
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            preds: model output.
            label: gt.

        """
        # only one pred and one gt used in MeanIOU calculation.
        pred_label = _as_list(preds)[0].detach()

        mask = label != self.ignore_index
        pred_label = pred_label[mask].float()
        label = label[mask].float()

        intersect = pred_label[pred_label == label]

        area_intersect = torch.histc(
            intersect, bins=self.num_classes, max=self.num_classes - 1
        )
        area_pred_label = torch.histc(
            pred_label, bins=self.num_classes, max=self.num_classes - 1
        )
        area_label = torch.histc(
            label, bins=self.num_classes, max=self.num_classes - 1
        )
        area_union = area_pred_label + area_label - area_intersect

        self.intersect += area_intersect
        self.union += area_union
        self.pred_label += area_pred_label
        self.label += area_label

    def compute(self):
        """Get evaluation metrics."""

        def _save_tensor_ele(x: torch.Tensor, save_index_list: list):
            _save_tensor = [x[index : index + 1] for index in save_index_list]
            return torch.cat(_save_tensor, dim=0)

        def _tensor_nan_mean(x: torch.Tensor):
            """Compute the mean value, ignoring NaNs."""

            tmp_value = x.cpu().numpy()
            _mean_iou = np.nanmean(tmp_value)
            mean_iou = x.new_tensor(_mean_iou)
            return mean_iou

        all_acc = (
            _save_tensor_ele(self.intersect, self.global_save_index_list).sum()
            / _save_tensor_ele(self.label, self.global_save_index_list).sum()
        )
        acc = self.intersect / self.label
        iou = self.intersect / self.union

        summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
        summary_str += "Summary:\n"
        line_format = "{:<15} {:>10} {:>10} {:>10}\n"
        summary_str += line_format.format("Scope", "mIoU", "mAcc", "aAcc")

        miou = _tensor_nan_mean(
            _save_tensor_ele(iou, self.global_save_index_list)
        )
        macc = _tensor_nan_mean(
            _save_tensor_ele(acc, self.global_save_index_list)
        )
        iou_str = "{:.2f}".format(miou.cpu().item() * 100)
        acc_str = "{:.2f}".format(macc.cpu().item() * 100)
        all_acc_str = "{:.2f}".format(all_acc * 100)
        summary_str += line_format.format(
            "global", iou_str, acc_str, all_acc_str
        )

        summary_str += "Per Class Results:\n"
        line_format = "{:<15} {:>10} {:>10}\n"
        summary_str += line_format.format("Class", "IoU", "Acc")

        for i in range(self.num_classes):
            iou_str = "{:.2f}".format(iou[i].cpu().item() * 100)
            acc_str = "{:.2f}".format(acc[i].cpu().item() * 100)
            summary_str += line_format.format(
                self.seg_class[i], iou_str, acc_str
            )
        logger.info(summary_str)

        if self.verbose:
            return miou, macc, all_acc, iou, acc, self.seg_class
        else:
            return miou

    def compute_2(self):
        """Get evaluation metrics."""

        def _save_tensor_ele(x: torch.Tensor, save_index_list: list):
            _save_tensor = [x[index : index + 1] for index in save_index_list]
            return torch.cat(_save_tensor, dim=0)

        def _tensor_nan_mean(x: torch.Tensor):
            """Compute the mean value, ignoring NaNs."""

            tmp_value = x.cpu().numpy()
            _mean_iou = np.nanmean(tmp_value)
            mean_iou = x.new_tensor(_mean_iou)
            return mean_iou

        all_acc = (
            _save_tensor_ele(self.intersect, self.global_save_index_list).sum()
            / _save_tensor_ele(self.label, self.global_save_index_list).sum()
        )
        acc = self.intersect / self.label
        iou = self.intersect / self.union

        summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
        summary_str += "Summary:\n"
        line_format = "{:<15} {:>10} {:>10} {:>10}\n"
        summary_str += line_format.format("Scope", "mIoU", "mAcc", "aAcc")

        miou = _tensor_nan_mean(
            _save_tensor_ele(iou, self.global_save_index_list)
        )
        macc = _tensor_nan_mean(
            _save_tensor_ele(acc, self.global_save_index_list)
        )
        iou_str = "{:.2f}".format(miou.cpu().item() * 100)
        acc_str = "{:.2f}".format(macc.cpu().item() * 100)
        all_acc_str = "{:.2f}".format(all_acc * 100)
        summary_str += line_format.format(
            "global", iou_str, acc_str, all_acc_str
        )

        summary_str += "Per Class Results:\n"
        line_format = "{:<15} {:>10} {:>10}\n"
        summary_str += line_format.format("Class", "IoU", "Acc")

        for i in range(self.num_classes):
            iou_str = "{:.2f}".format(iou[i].cpu().item() * 100)
            acc_str = "{:.2f}".format(acc[i].cpu().item() * 100)
            summary_str += line_format.format(
                self.seg_class[i], iou_str, acc_str
            )
        logger.info(summary_str)

        if self.verbose:
            return miou, macc, all_acc, iou, acc, self.seg_class
        else:
            return miou

    def get(self):
        """Get current evaluation result.

        Returns:
           names: Name of the metrics.
           values: Value of the evaluations.
        """

        values = self.compute_2()
        if isinstance(values, list):
            assert isinstance(self.name, list) and len(self.name) == len(
                values
            )

        return self.name, values
