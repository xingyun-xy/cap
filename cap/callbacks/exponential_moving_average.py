# Copyright (c) Changan Auto. All rights reserved.

import itertools
import logging
import math
from copy import deepcopy
from typing import Optional

import torch

from cap.registry import OBJECT_REGISTRY
from cap.utils.model_helpers import get_binding_module
from .callbacks import CallbackMixin

__all__ = ["ExponentialMovingAverage"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ExponentialMovingAverage(CallbackMixin):  # noqa: D400
    def __init__(
        self,
        decay: Optional[float] = 0.9999,
        decay_base: Optional[float] = 2000,
    ):
        """Callbacks of ExponentialMovingAverage.

        Some training algorithms, such as GradientDescent and Momentum
        often benefit from maintaining a moving average of variables
        during optimization. Using the moving averages for evaluations
        often improve results significantly.
        https://www.tensorflow.org/versions/master/api_docs/python/train/moving_averages

        Args:
            decay: Decay ratio for ema.
            decay_base: Decay base for ema.

        """
        self.decay = lambda x: decay * (1 - math.exp(-x / decay_base))

    def update(self, global_step_id, model, ema_model):
        with torch.no_grad():
            d = self.decay(global_step_id + 1)
            msd = get_binding_module(model).state_dict()
            for k, v in itertools.chain(
                ema_model.named_parameters(), ema_model.named_buffers()
            ):
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def on_step_end(
        self,
        global_step_id,
        model=None,
        ema_model=None,
        **kwargs,
    ):
        if ema_model is None:
            return
        self.update(global_step_id, model, ema_model)

    def on_loop_begin(
        self,
        loop,
        model,
        **kwargs,
    ):
        if not hasattr(loop, "ema_model"):
            logger.warning("loop has no ema_model, ema is disabled.")
            return

        if loop.ema_model is None:
            loop.ema_model = deepcopy(get_binding_module(model)).eval()
            for p in loop.ema_model.parameters():
                p.requires_grad_(False)
