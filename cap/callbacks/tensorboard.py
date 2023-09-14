# Copyright (c) Changan Auto. All rights reserved.

import logging
import os
import re
import shutil
from typing import Callable, Optional, Sequence

import torch
from timeout_decorator import TimeoutError, timeout
from torch.utils import tensorboard

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import to_flat_ordered_dict
from cap.utils.distributed import rank_zero_only
from .callbacks import CallbackMixin

__all__ = ["TensorBoard"]

logger = logging.getLogger(__name__)

# TODO(?) support recording metric in tensorboard.


class SummaryWriter(tensorboard.SummaryWriter):
    """The use of this class is exactly the same as tensorboard.SummaryWriter.

    The reason for overriding each `add_` method is to add a time limit
    in case of spending too much time in writing.
    """

    def __init__(self, *args, **kwargs):
        super(SummaryWriter, self).__init__(*args, **kwargs)

    @timeout(seconds=60)
    def add_hparams(self, *args, **kwargs):
        super(SummaryWriter, self).add_hparams(*args, **kwargs)

    @timeout(seconds=60)
    def add_scalar(self, *args, **kwargs):
        super(SummaryWriter, self).add_scalar(*args, **kwargs)

    @timeout(seconds=60)
    def add_scalars(self, *args, **kwargs):
        super(SummaryWriter, self).add_scalars(*args, **kwargs)

    @timeout(seconds=60)
    def add_histogram(self, *args, **kwargs):
        super(SummaryWriter, self).add_histogram(*args, **kwargs)

    @timeout(seconds=60)
    def add_histogram_raw(self, *args, **kwargs):
        super(SummaryWriter, self).add_histogram_raw(*args, **kwargs)

    @timeout(seconds=60)
    def add_image(self, *args, **kwargs):
        super(SummaryWriter, self).add_image(*args, **kwargs)

    @timeout(seconds=60)
    def add_images(self, *args, **kwargs):
        super(SummaryWriter, self).add_images(*args, **kwargs)

    @timeout(seconds=60)
    def add_image_with_boxes(self, *args, **kwargs):
        super(SummaryWriter, self).add_image_with_boxes(*args, **kwargs)

    @timeout(seconds=60)
    def add_figure(self, *args, **kwargs):
        super(SummaryWriter, self).add_figure(*args, **kwargs)

    @timeout(seconds=60)
    def add_video(self, *args, **kwargs):
        super(SummaryWriter, self).add_video(*args, **kwargs)

    @timeout(seconds=60)
    def add_audio(self, *args, **kwargs):
        super(SummaryWriter, self).add_audio(*args, **kwargs)

    @timeout(seconds=60)
    def add_text(self, *args, **kwargs):
        super(SummaryWriter, self).add_text(*args, **kwargs)

    @timeout(seconds=60)
    def add_onnx_graph(self, *args, **kwargs):
        super(SummaryWriter, self).add_onnx_graph(*args, **kwargs)

    @timeout(seconds=60)
    def add_graph(self, *args, **kwargs):
        super(SummaryWriter, self).add_graph(*args, **kwargs)

    @timeout(seconds=60)
    def add_embedding(self, *args, **kwargs):
        super(SummaryWriter, self).add_embedding(*args, **kwargs)

    @timeout(seconds=60)
    def add_pr_curve(self, *args, **kwargs):
        super(SummaryWriter, self).add_pr_curve(*args, **kwargs)

    @timeout(seconds=60)
    def add_pr_curve_raw(self, *args, **kwargs):
        super(SummaryWriter, self).add_pr_curve_raw(*args, **kwargs)

    @timeout(seconds=60)
    def add_custom_scalars_multilinechart(self, *args, **kwargs):
        super(SummaryWriter, self).add_custom_scalars_multilinechart(
            *args, **kwargs
        )

    @timeout(seconds=60)
    def add_custom_scalars_marginchart(self, *args, **kwargs):
        super(SummaryWriter, self).add_custom_scalars_marginchart(
            *args, **kwargs
        )

    @timeout(seconds=60)
    def add_custom_scalars(self, *args, **kwargs):
        super(SummaryWriter, self).add_custom_scalars(*args, **kwargs)

    @timeout(seconds=60)
    def add_mesh(self, *args, **kwargs):
        super(SummaryWriter, self).add_mesh(*args, **kwargs)


@OBJECT_REGISTRY.register
class TensorBoard(CallbackMixin):  # noqa: D205,D400
    """
    TensorBoard Callback is used for recording somethings durning training,
    such as loss, image and other visualization.

    Args:
        save_dir (str): Directory to save tensorboard.
        overwrite (bool): Whether overwriting existed save_dir.
        loss_name_reg (str): Specific loss pattern.
        update_freq (int): Freq of tensorboard whether to output to file.
        update_by: Set `update_freq` unit to step or epoch.
            Default is step.
        tb_update_funcs (list(callable)): list of function with
            `writer`, `model_outs`, `step_id` as input and then
            update tensorboard.

    """

    @rank_zero_only
    def __init__(
        self,
        model,
        save_dir: str,
        overwrite: bool = False,
        loss_name_reg: str = "^.*loss.*",
        update_freq: int = 1,
        update_by: str = "step",
        tb_update_funcs: Optional[Sequence[Callable]] = None,
        **tb_writer_kwargs,
    ):

        if overwrite:
            shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)
        self.writer = SummaryWriter(save_dir, **tb_writer_kwargs)
        self.model = model
        self.update_freq = update_freq
        self.update_by = update_by
        self.tb_update_funcs = tb_update_funcs
        self.loss_name_pattern = re.compile(loss_name_reg, re.IGNORECASE)

    @rank_zero_only
    def on_loop_end(self, **kwargs):
        if self.writer is not None:
            self.writer.close()

    def _record_loss(self, model_outs, global_step_id):
        for k, v in model_outs.items():
            if self.loss_name_pattern.match(k):
                if v is not None and isinstance(v, torch.Tensor):
                    try:
                        self.writer.add_scalar(
                            k, v, global_step=global_step_id
                        )
                    except TimeoutError as exception:
                        logger.exception(str(exception))
                        continue

    def _record_lr(self, optimizer: torch.optim.Optimizer, global_step_id):
        # print (optimizer)
        assert isinstance(optimizer, torch.optim.Optimizer), type(optimizer)
        last_lr = optimizer.state_dict()['param_groups'][-1]["lr"]
        self.writer.add_scalar( 'lr', last_lr, global_step=global_step_id)

    @rank_zero_only
    def on_batch_end(self, model_outs=None, global_step_id=None, **kwargs):
        model = self.model
        if self.update_by == "step":
            # print (batch)
            assert model_outs is not None and global_step_id is not None, (
                "model_outs and global_step_id must be "
                + "provided when update by step"
            )
            if global_step_id % self.update_freq == 0:
                model_outs = to_flat_ordered_dict(model_outs) 
                self._record_loss(model_outs, global_step_id)
                if model is not None:
                    for name, param in model.named_parameters():
                        self.writer.add_histogram(name, param.grad.clone().cpu().data.numpy(), global_step_id)
                if self.tb_update_funcs is not None:
                    for func in self.tb_update_funcs:
                        try:
                            func(
                                self.writer,
                                model_outs=model_outs,
                                global_step_id=global_step_id,
                                **kwargs,
                            )
                        except TimeoutError as exception:
                            logger.exception(str(exception))
                            continue

    @rank_zero_only
    def on_step_end(self, optimizer=None, global_step_id=None, **kwargs):
        if self.update_by == "step":
            # print (batch)
            if global_step_id % self.update_freq == 0:
                self._record_lr(optimizer, global_step_id)
                if self.tb_update_funcs is not None:
                    for func in self.tb_update_funcs:
                        try:
                            func(
                                self.writer,
                                model_outs=model_outs,
                                global_step_id=global_step_id,
                                **kwargs,
                            )
                        except TimeoutError as exception:
                            logger.exception(str(exception))
                            continue

    @rank_zero_only
    def on_epoch_end(self, epoch_id=None, **kwargs):
        if self.update_by == "epoch":
            assert (
                epoch_id is not None
            ), "epoch_id must be provided when update by epoch"
            if epoch_id % self.update_freq == 0:
                if self.tb_update_funcs is not None:
                    for func in self.tb_update_funcs:
                        func(self.writer, epoch_id=epoch_id, **kwargs)
