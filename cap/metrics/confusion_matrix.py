# Copyright (c) Changan Auto. All rights reserved.

import logging
from typing import List, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list

from .metric import EvalMetric

__all__ = ["ConfusionMatrix"]

logger = logging.getLogger(__name__)


def gen_confusion_matrix(img_predict, img_label, num_class):
    mask = (img_label >= 0) & (img_label < num_class)
    label = num_class * img_label[mask] + img_predict[mask]
    label = label.astype(np.int64)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


@OBJECT_REGISTRY.register
class ConfusionMatrix(EvalMetric):
    """Evaluation segmentation results.

    Args:
        seg_class: a list of classes the segmentation dataset includes，
            the order should be the same as the label.
        name: name of this metric instance for display, also used as
            monitor params for Checkpoint.
        ignore_index: the label index that will be ignored in evaluation.

    """

    def __init__(
        self,
        seg_class: List[str],
        name: str = "ConfusionMatrix",
        ignore_index: int = 255,
        dist_sync_on_step: bool = False,
    ):
        self.num_classes = len(seg_class)
        self.label_ids = range(self.num_classes)
        self.seg_class = seg_class
        super(ConfusionMatrix, self).__init__(name)
        self.ignore_index = ignore_index
        self.dist_sync_on_step = dist_sync_on_step

    def _init_states(self):
        self.add_state(
            "confusion_matrix",
            default=torch.zeros((self.num_classes, self.num_classes)),
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

        cls_confusion_matrix = gen_confusion_matrix(
            label.numpy(), pred_label.numpy(), self.num_classes
        )

        # cls_confusion_matrix = confusion_matrix(label.numpy(),
        #                                         pred_label.numpy(),
        #                                         labels=self.label_ids)
        area_pred_label = torch.histc(
            pred_label, bins=self.num_classes, max=self.num_classes - 1
        )
        area_label = torch.histc(
            label, bins=self.num_classes, max=self.num_classes - 1
        )

        self.confusion_matrix += cls_confusion_matrix
        self.label += area_label
        self.pred_label += area_pred_label

    def compute(self):
        """Get evaluation metrics."""

        for i in range(self.num_classes):
            self.confusion_matrix[i] = self.confusion_matrix[i] / max(
                self.label[i], 1
            )

        return (self.confusion_matrix, self.label, self.pred_label)

    def compute_2(self):
        """Get evaluation metrics."""

        for i in range(self.num_classes):
            self.confusion_matrix[i] = self.confusion_matrix[i] / max(
                self.label[i], 1
            )

        return (self.confusion_matrix, self.label, self.pred_label)

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
