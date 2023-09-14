# Copyright (c) Changan Auto. All rights reserved.
import logging
import os
import pickle
from typing import Callable

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import convert_numpy
from cap.utils.distributed import all_gather_object
from .save_eval_result import SaveEvalResult

__all__ = ["SaveDet2dResult"]

logger = logging.getLogger(__name__)


def default_det2d_match(batch, model_outs):
    if "structure" in batch:
        objects = batch["structure"]
    else:
        raise KeyError("Cannot find img structure in batch data!")

    preds, _ = model_outs

    return objects, preds


@OBJECT_REGISTRY.register
class SaveDet2dResult(SaveEvalResult):
    """
    Save det2d result is used for saving results on 2d-detection tasks.

    Args:
        output_dir: Output dir for saving results.
        match_task_output: Function to match current task outputs.
    """

    def __init__(
        self,
        output_dir: str,
        match_task_output: Callable = default_det2d_match,
    ):
        super().__init__(output_dir, match_task_output)
        if self.rank == 0:
            self.output_viz_dir = os.path.join(self.output_dir, "viz_imgs")
            if not os.path.exists(self.output_viz_dir):
                os.makedirs(self.output_viz_dir)

    def save_result(self, global_id, batch, result):
        output = {}
        output["batch"] = batch
        output["result"] = result
        np_output = convert_numpy(output)
        out_pkl = pickle.dumps(np_output)
        global_out_pkl = [None for _ in range(self.world_size)]
        all_gather_object(global_out_pkl, out_pkl)
        if self.rank == 0:
            for out_pkl_rank in global_out_pkl:
                with open(
                    "{}/{}.pkl".format(self.output_dir, "det2d"), "ab"
                ) as f:
                    f.write(out_pkl_rank)

    def __repr__(self):
        return "SaveDet2dResult"
