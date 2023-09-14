# Copyright (c) Changan Auto. All rights reserved.

from typing import Sequence

import torch

from cap.models.base_modules.resize_parser import resize
from cap.registry import OBJECT_REGISTRY

__all__ = ["SegTarget"]


@OBJECT_REGISTRY.register
class SegTarget(object):
    """Generate training targets for Seg task.

    Args:
        ignore_index (int, optional): Index of ignore class.
        label_name (str, optional): The key corresponding to the gt seg in
            label.
    """

    def __init__(self, ignore_index=255, label_name="gt_seg"):
        self.ignore_index = ignore_index
        self.label_name = label_name

    def __call__(
        self, label: dict, pred: Sequence[torch.Tensor], *args
    ) -> Sequence[dict]:  # noqa: D205,D400
        """

        Args:
            label: Meta data dict.
            pred: Output corresponding to multiple strides, the
                shape of each element is NHWC, HW is different for each stride.
            *args: Receive extra parameters, not used.

        Returns:
            Sequence[dict]: Loss inputs list.

        """
        # loss need 3D input
        if label[self.label_name].shape[0] != 1:
            gt_seg = label[self.label_name].squeeze()
        else:
            gt_seg = label[self.label_name].squeeze(-1)
        gt_seg = gt_seg.type(torch.int64)

        loss_inputs = []
        for _ind, out in enumerate(pred):
            loss_input = {}
            assert len(out.shape) == 4, "bilinear need 4D input"
            out = resize(
                input=out,
                size=gt_seg.shape[1:3],
                mode="bilinear",
                align_corners=False,
                warning=False,
            )
            loss_input["pred"] = out
            loss_input["target"] = gt_seg
            avg_factor = (gt_seg != self.ignore_index).sum()
            loss_input["avg_factor"] = avg_factor.clamp(min=1.0)
            loss_inputs.append(loss_input)
        return loss_inputs
