# Copyright (c) Changan Auto. All rights reserved.
import logging
from collections import defaultdict
from typing import List, Optional

import torch

from cap.registry import OBJECT_REGISTRY
from .callbacks import CallbackMixin

__all__ = ["GradScale"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class GradScale(CallbackMixin):
    """Set gradient scale for different module.

    When training multitask, gradient of each task might be different.
    Comparing to changing loss weight, another more efficient method is to
    adjust gradient.

    Example:
        >>> grad_scale_callback = dict(
        ...    type="GradScale",
        ...    module_and_scale=[
        ...        ("backbone", 0.1, "real3d_fov120"),
        ...        ("bifpn", 0.1, "real3d_fov120"),
        ...        ("real3d_fov120", 1.0, "real3d_fov120"),
        ...    ],
        ...    clip_grad_norm=None,
        ...)

    Args:
        module_and_scale: module name, gradient scale and task name. Task name
            can be none if you don't need.
        clip_grad_norm: Max norm for `torch.nn.utils.clip_grad_norm_`.
        clip_norm_type: Norm type for `torch.nn.utils.clip_grad_norm_`.
    """

    def __init__(
        self,
        module_and_scale: List,
        clip_grad_norm: Optional[float] = None,
        clip_norm_type: Optional[int] = 2,
    ):
        self.module_and_scale = module_and_scale
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm_type = clip_norm_type
        self._grad_cache = defaultdict(float)

    def on_loop_begin(self, **kwargs):
        logger.info(f"[GradScale] {self.module_and_scale}")

    def on_backward_end(self, model, batch, optimizer, **kwargs):
        """Task-wise backward_end."""
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            for item in self.module_and_scale:
                # get task from item
                if len(item) == 3:
                    module, scale, task = item
                elif len(item) == 2:
                    module, scale = item
                    task = None
                else:
                    raise ValueError(f"Unvalid args: {item}")
                # do scale
                if (module in name) and (not task or task in batch):
                    param.grad *= scale
            # cache grad
            self._grad_cache[name] += param.grad.detach()
        optimizer.zero_grad()

    def on_optimizer_step_begin(self, model, **kwargs):
        # move grad from cache to param.grad
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            param.grad = self._grad_cache[name]
        # do clip grad norm
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.clip_grad_norm,
                norm_type=self.clip_norm_type,
            )
        # empty grad cache
        self._grad_cache.clear()
