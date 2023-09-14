# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Callable, Optional

from cap.registry import OBJECT_REGISTRY
from ..callbacks import CallbackMixin

__all__ = ["BaseVisualize"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BaseVisualize(CallbackMixin):
    """
    BaseVisualize Callback is used for visualize results on specific tasks.

    All results are first written on a picture and saved,
    and then drawn according to the needs of users.

    Args:
        match_task_output: Function to match current task outputs.
    """

    def __init__(
        self,
        match_task_output: Optional[Callable] = None,
    ):
        self.match_task_output = match_task_output

    def visualize(self, global_id, batch, result):
        """Visualize method."""
        raise NotImplementedError

    def on_batch_end(self, global_step_id, batch, model_outs, **kwargs):
        if self.match_task_output:
            batch_output, match_output = self.match_task_output(
                batch, model_outs
            )
        else:
            batch_output, match_output = batch, model_outs

        self.visualize(global_step_id, batch_output, match_output)

    def __repr__(self):
        return "BaseVisualize"
