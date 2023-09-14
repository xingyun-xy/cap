from typing import List

import torch.nn.functional as F

from cap.registry import OBJECT_REGISTRY

__all__ = ["IPMHeadParser"]


@OBJECT_REGISTRY.register
class IPMHeadParser(object):
    """Reisze head predictions for IPM-Seg task.

    Args:
        out_strides: Output strides of head prediction.
    """

    def __init__(self, out_strides: List[int]):
        self.out_strides = out_strides

    def __call__(self, preds):
        assert len(preds) == len(self.out_strides)
        resize_preds = []
        for i in range(len(self.out_strides)):
            resize_preds.append(
                F.interpolate(
                    preds[i], scale_factor=self.out_strides[i], mode="bilinear"
                )
            )
        if len(self.out_strides) == 1:
            resize_preds = resize_preds[0]
        return resize_preds
