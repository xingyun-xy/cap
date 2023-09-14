# Copyright (c) Changan Auto. All rights reserved.

import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from cap.callbacks import CallbackMixin
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from cap.utils.logger import MSGColor, format_msg
from .launcher import register_launcher
from .loop_base import LoopBase

__all__ = ["Trainer"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class Trainer(LoopBase):  # noqa: D205,D400
    """Trainer is a tool for train, which include all pipeline for training.

    Args:
        model: Model config or a `nn.Module` instance.
        data_loader: Training data loader config or a instantiated data loader.
        optimizer: Optimizer config or a optimizer instance.
        batch_processor: Batch processor config or a `BatchProcessorMixin`
            instance.
        device: Int gpu id or None.
            If int, do `model.cuda(device)` and `data.cuda(device)`.
            If None, no-op.
        model_convert_pipeline: Define the process of model convert.
            e.g. convert float model to qat model, convert qat model
            to quantize model.
        resume_optimizer: whether load optimizer dict when resume checkpoint.
        resume_epoch_or_step: whether need to resume epoch_or_step
            when resume checkpoint.
        stop_by: Stop training by counting epoch or step.
            If equal to 'epoch', stop training when `epoch_id == num_epochs - 1`.
            If equal to 'step', stop training when `global_step_id == num_steps - 1`.
            Default 'epoch'.
        num_epochs: Num of training epochs, should be non-negative integer.
            If stop_by != 'epoch', no-op.
            Set 0 to skip training and run `self.on_loop_begin/end` only.
        start_epoch: Training start epoch, should be non-negative integer.
        num_steps: Num of training steps, should be non-negative integer.
            If stop_by != 'step', no-op.
            Set 0 to skip training and run `self.on_loop_begin/end` only.
        start_step: Training start step, should be non-negative integer.
        callbacks: Callback configs or instances.
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        log_interval: Logging output frequency.
        val_model: Model for validation. Passed to Validation Callback as
            a parameter.
        deploy_model: Model for tracing and export to pt file. Passed to
            Checkpoint Callback as a parameter.
        deploy_inputs: Example inputs for tracing deploy_model. Passed to
            Checkpoint Callback as a parameter.
    """  # noqa

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        batch_processor,
        device: Union[int, None],
        model_convert_pipeline: Optional[Union[Dict, List]] = None,
        resume_optimizer: bool = False,
        resume_epoch_or_step: bool = False,
        stop_by: Optional[str] = "epoch",
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = 0,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = 0,
        callbacks: Optional[Sequence[Union[dict, CallbackMixin]]] = None,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        log_interval: int = 0,
        val_model: Optional[nn.Module] = None,
        deploy_model: Optional[nn.Module] = None,
        deploy_inputs: Optional[Dict] = None,
        val_only: bool = False,
        export_ckpt_only: bool = False,
    ):
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.callbacks = callbacks
        self.val_only = val_only
        self.export_ckpt_only = export_ckpt_only

        super(Trainer, self).__init__(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            batch_processor=batch_processor,
            device=device,
            model_convert_pipeline=model_convert_pipeline,
            resume_optimizer=resume_optimizer,
            resume_epoch_or_step=resume_epoch_or_step,
            stop_by=stop_by,
            num_epochs=self.num_epochs,
            start_epoch=start_epoch,
            num_steps=self.num_steps,
            start_step=start_step,
            callbacks=self.callbacks,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            profiler=profiler,
            log_interval=log_interval,
        )

        self.profiler.setup(stage="train", local_rank=device)
        self.model.train()

    def on_loop_begin(self, **kwargs):
        self.model.train()
        super(Trainer, self).on_loop_begin(**kwargs)


def launch(
    main_func, device_ids=None, dist_url=None, dist_launcher=None, args=()
):
    if device_ids is None:
        current_device = None
    else:
        device_ids = _as_list(device_ids)
        if len(device_ids) > 1:
            msg = (
                "`Trainer` only support one device, but get %s, now only "
                "use device %d. Use `distributed_data_parallel_trainer` "
                "or `data_parallel_trainer` instead if you want to train "
                "on multiple devices." % (device_ids, device_ids[0])
            )
            logger.info(format_msg(msg, MSGColor.GREEN))
            device_ids = device_ids[:1]

        # Note: if device_ids=[3], then after setting
        # `CUDA_VISIBLE_DEVICES`, index of gpu 3 becomes 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])
        current_device = 0
        torch.cuda.set_device(current_device)

    main_func(current_device, *args)


register_launcher("Trainer", launch)
