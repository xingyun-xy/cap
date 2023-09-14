# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Dict, Union

import torch

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .metric import EvalMetric

__all__ = ["LossShow"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class LossShow(EvalMetric):
    """Show loss.

    # TODO(min.du, 0.1): a better class name is required #

    Args:
        name: Name of this metric instance for display.
        norm: Whether norm loss when loss size bigger than 1.
            If True, calculate mean loss, else calculate loss sum.
            Default True.
    """

    def __init__(self, name: str = "loss", norm: bool = True):
        super(LossShow, self).__init__(name, warn_without_compute=False)
        self.norm = norm

    def _init_states(self):
        self.num_inst = 0
        self.loss = 0.0
        self.first_update = True

    def update(self, loss: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if loss is None:
            return

        if isinstance(loss, torch.Tensor):
            if loss.size().numel() > 1:
                loss = loss.mean() if self.norm else loss.sum()

            self.loss += loss.item()
            self.num_inst += 1
        else:
            assert isinstance(loss, dict)
            if self.first_update:
                self.name = list(loss.keys())
                self.loss = [ll.item() for ll in loss.values()]
                self.first_update = False
                return
            else:
                losses = [ll.item() for ll in loss.values()]
                self.loss = [
                    loss1 + loss2
                    for (loss1, loss2) in zip(*(self.loss, losses))
                ]
            self.num_inst += 1

    def get(self):
        if not self.num_inst:
            logger.error("self.num_inst is 0, please check.")
            return _as_list(self.name), [0.0] * len(_as_list(self.name))

        return _as_list(self.name), [
            loss / self.num_inst for loss in _as_list(self.loss)
        ]

    def reset(self):
        self.num_inst = 0
        self.loss = 0.0
        self.first_update = True
