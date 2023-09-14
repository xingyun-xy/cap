# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cap.registry import OBJECT_REGISTRY

__all__ = ["FCNTarget"]


@OBJECT_REGISTRY.register
class FCNTarget(nn.Module):
    """Generate Target for FCN.

    Args:
        num_classes: Number of classes. Defualt: 19.
    """

    def __init__(
        self,
        num_classes: Optional[int] = 19,
    ):
        super(FCNTarget, self).__init__()
        self.num_classes = num_classes

    def forward(
        self, label: torch.Tensor, pred: torch.Tensor
    ) -> dict:  # noqa: D205,D400
        """

        Args:
            label: data Tenser.(n, h, w)
            pred: Output Tenser. (n, c, h, w).

        Returns:
            dict: Loss inputs.

        """
        if label.ndim == 4:
            size = label.shape[2:]
        else:
            size = label.shape[1:]
        scaled_pred = F.interpolate(
            input=pred, size=size, mode="bilinear", align_corners=None
        )
        seg_target = {
            "pred": scaled_pred,
            "target": label,
        }
        return seg_target
