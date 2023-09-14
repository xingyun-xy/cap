# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Sequence

from cap.registry import OBJECT_REGISTRY
from .visualize import BaseVisualize
from projects.panorama.configs.resize.common import infer_save_prefix

__all__ = ["ComposeVisualize"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ComposeVisualize(BaseVisualize):
    """
    ComposeVisualize Callback is used for compose visualize results.

    All results are first written on a picture and saved,
    and then drawn according to the needs of users.

    Args:
        callbacks: All of callbacks for visualize.
    """

    def __init__(
        self,
        callbacks: Sequence[BaseVisualize],
        bev_eval: bool = False
    ):
        for callback in callbacks:
            assert isinstance(callback, BaseVisualize)
        self.callbacks = callbacks
        self.bev_eval = bev_eval
        super().__init__()

    def on_epoch_begin(self, epoch_id, **kwargs):
        self.bev_res = []


    def on_batch_end(self, global_step_id, batch, model_outs, **kwargs):
        for callback in self.callbacks:
            if callback.match_task_output:
                batch_output, match_output = callback.match_task_output(
                    batch, model_outs
                )
            else:
                batch_output, match_output = batch, model_outs

            global_step_id, batch_output, match_output = callback.visualize(
                global_step_id, batch_output, match_output
            )
            # add some results bev eval needed     add by zmj
            if match_output.get("singletask_bev"):
                self.bev_res.append(match_output["singletask_bev"][2])


    def on_epoch_end(self, **kwargs):
            if self.bev_res and self.bev_eval:
                import numpy as np
                import os
                save_path = os.path.join(infer_save_prefix, 'bev_res_all.npy')
                np.save(save_path, np.array(self.bev_res))

        


    def __repr__(self):
        return "ComposeVisualize"
