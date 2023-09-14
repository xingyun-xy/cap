# Copyright (c) Changan Auto. All rights reserved.
from typing import Dict, Tuple

import torch

from cap.registry import OBJECT_REGISTRY

__all__ = ["FCOSLoss"]


@OBJECT_REGISTRY.register
class FCOSLoss(torch.nn.Module):
    """
    FCOS loss wrapper.

    Args:
        losses (list): loss configs.

    Note:
        This class is not universe. Make sure you know this class limit before
        using it.

    """

    def __init__(
        self,
        cls_loss: torch.nn.Module,
        reg_loss: torch.nn.Module,
        centerness_loss: torch.nn.Module,
    ):
        super(FCOSLoss, self).__init__()
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.centerness_loss = centerness_loss

    def forward(self, pred: Tuple, target: Tuple[Dict]) -> Dict:
        assert len(target) == 3
        res = {}
        # `pred` is in target tuple, so we get pred from target.
        # assume fcos target is in cls/reg/centerness order.
        cls_res = self.cls_loss(**target[0])
        reg_res = self.reg_loss(**target[1])
        ctr_res = self.centerness_loss(**target[2])
        res.update(cls_res)
        res.update(reg_res)
        res.update(ctr_res)
        # Three dict shouldn't contain same key.
        assert len(res) == len(cls_res) + len(reg_res) + len(ctr_res), (
            "`cls_res`, `reg_cls` and `ctr_res` have same name keys, "
            "this may cause bugs."
        )
        return res
