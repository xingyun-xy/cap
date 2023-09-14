# Copyright (c) Changan Auto. All rights reserved.
import logging
import os
from typing import Callable, Optional

from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info
from ..callbacks import CallbackMixin

__all__ = ["SaveEvalResult"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SaveEvalResult(CallbackMixin):
    """
    Save result of predictor for evaluation.

    Args:
        output_dir: Directory for saving result.
        match_task_output: Function to match current task outputs.

    """

    def __init__(
        self,
        output_dir: str,
        match_task_output: Optional[Callable] = None,
    ):
        self.output_dir = output_dir
        self.match_task_output = match_task_output
        self.rank, self.world_size = get_dist_info()
        if self.rank == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def save_result(self, global_id, batch, result):
        """Save method."""
        raise NotImplementedError

    def on_batch_end(self, global_step_id, batch, model_outs, **kwargs):
        if self.match_task_output:
            batch_output, match_output = self.match_task_output(
                batch, model_outs
            )
        else:
            batch_output, match_output = batch, model_outs

        self.save_result(global_step_id, batch_output, match_output)

    def __repr__(self):
        return "SaveEvalResult"
