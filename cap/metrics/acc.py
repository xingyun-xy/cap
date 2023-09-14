# Copyright (c) Changan Auto. All rights reserved.

import numpy
import torch

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .metric import EvalMetric

__all__ = ["Accuracy", "TopKAccuracy", "AccuracySeg"]


@OBJECT_REGISTRY.register
class Accuracy(EvalMetric):
    """Computes accuracy classification score.

    Args:
        axis (int): The axis that represents classes
        name (str):  Name of this metric instance for display.
    """

    def __init__(
        self,
        axis=1,
        name="accuracy",
    ):
        self.axis = axis
        super().__init__(name)

    def update(self, labels, preds):

        labels = _as_list(labels)
        preds = _as_list(preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = torch.argmax(pred_label, self.axis)

            # flatten before checking shapes to avoid shape miss match
            label = label.flatten()
            pred_label = pred_label.flatten()

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            pred_label_num = (label >= 0).sum()
            self.num_inst += pred_label_num


@OBJECT_REGISTRY.register
class AccuracySeg(EvalMetric):
    """# TODO(xuefang.wang, 0.5): merged with Accuracy #."""

    def __init__(
        self,
        name="accuracy",
        axis=1,
    ):
        super().__init__(name)
        self.axis = axis

    def update(self, output):
        labels = output["gt_seg"]
        preds = output["pred_seg"]

        labels = _as_list(labels)
        preds = _as_list(preds)

        for label, pred_label in zip(labels, preds):
            # different with Accuracy
            if pred_label.shape != label.shape:
                pred_label = torch.argmax(pred_label, self.axis)

            # flatten before checking shapes to avoid shape miss match
            label = label.flatten()
            pred_label = pred_label.flatten()

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            pred_label_num = (label >= 0).sum()
            self.num_inst += pred_label_num


@OBJECT_REGISTRY.register
class TopKAccuracy(EvalMetric):
    """Computes top k predictions accuracy.

    `TopKAccuracy` differs from Accuracy in that it considers the prediction
    to be ``True`` as long as the ground truth label is in the top K
    predicated labels.

    If `top_k` = ``1``, then `TopKAccuracy` is identical to `Accuracy`.

    Args:
        top_k (int): Whether targets are in top k predictions.
        name (str):  Name of this metric instance for display.
    """

    def __init__(self, top_k, name="top_k_accuracy"):
        super().__init__(name)
        self.top_k = top_k
        assert self.top_k > 1, "Please use Accuracy if top_k=1"
        self.name += "_%d" % self.top_k

    def update(self, labels, preds):

        labels = _as_list(labels)
        preds = _as_list(preds)

        for label, pred_label in zip(labels, preds):
            assert len(pred_label.shape) == 2, "Predictions should be 2 dims"
            # Using argpartition here instead of argsort is safe because
            # we do not care about the order of top k elements. It is
            # much faster, which is important since that computation is
            # single-threaded due to Python GIL.
            pred_label = numpy.argpartition(
                pred_label.detach().cpu().numpy().astype("float32"),
                -self.top_k,
            )
            label = label.detach().cpu().numpy().astype("int32")

            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    num_correct = (pred_label[:, num_classes - 1 -
                                              j].flat == label.flat).sum()
                    self.sum_metric += num_correct
            self.num_inst += num_samples
