# Copyright (c) Changan Auto. All rights reserved.
import torch

from cap.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["EndPointError"]


@OBJECT_REGISTRY.register
class EndPointError(EvalMetric):
    """Metric for OpticalFlow task, endpoint error (EPE).

    The endpoint error measures the distance between the
    endpoints of two optical flow vectors (u0, v0) and (u1, v1)
    and is defined as sqrt((u0 - u1) ** 2 + (v0 - v1) ** 2).

    Args:
        name: metric name.
    Refs:
        https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/model_pwcnet.py
    """

    def __init__(
        self,
        name="EPE",
    ):
        super(EndPointError, self).__init__(name)
        self.name = name

    def update(self, labels, preds):
        diff = preds - labels
        bs = preds.shape[0]
        epe = torch.norm(diff, p=2, dim=1).mean((1, 2)).sum().item()
        self.sum_metric += epe
        self.num_inst += bs
