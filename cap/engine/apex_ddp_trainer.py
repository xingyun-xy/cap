# Copyright (c) Changan Auto. All rights reserved.

import logging
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from cap.callbacks import CallbackMixin
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import to_cuda
from .ddp_trainer import launch
from .launcher import register_launcher
from .processors import BatchProcessorMixin
from .trainer import Trainer

try:
    import apex
except ImportError:
    apex = None

__all__ = ["ApexDistributedDataParallelTrainer"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("apex_distributed_data_parallel_trainer")
class ApexDistributedDataParallelTrainer(Trainer):
    """ApexDistributedDataParallelTrainer tool.

    ApexDistributedDataParallelTrainer is a tool function to new a
    `Trainer` instance, which training with
    `apex.distributed.DistributedDataParallel` method, and running
    on one of the GPU devices.

    The launch function is the same as `distributed_data_parallel_trainer`.

    By setting `stop_by`, you are able to stop training by counting epoch
    (default) or step.

    Args:
        model: Model config or a `nn.Module` instance.
        data_loader: Training data loader config or a instantiated data loader.
        optimizer: Optimizer config or a optimizer instance.
        batch_processor: Batch processor config or a `BatchProcessorMixin`
            instance.
        device: GPU id.
        stop_by: Stop training by counting epoch or step.
            If equal to 'epoch', stop training when
            `epoch_id == num_epochs - 1`.
            If equal to 'step', stop training when
            `global_step_id == num_steps - 1`.
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
        sync_bn: Whether to convert bn to sync_bn.
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        opt_level: The opt level of apex.
        keep_batchnorm_fp32: Whether to keep batchnorm fp32 of apex.
        loss_scale: The loss scale of apex.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        batch_processor: BatchProcessorMixin,
        device: int,
        stop_by: Optional[str] = "epoch",
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = 0,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = 0,
        callbacks: Optional[Sequence[CallbackMixin]] = None,
        sync_bn: Optional[bool] = False,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        opt_level: Optional[str] = "O0",
        keep_batchnorm_fp32: Optional[str] = None,
        loss_scale: Optional[str] = None,
        **kwargs
    ):
        super(ApexDistributedDataParallelTrainer, self).__init__(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            batch_processor=batch_processor,
            device=device,
            stop_by=stop_by,
            num_epochs=num_epochs,
            start_epoch=start_epoch,
            num_steps=num_steps,
            start_step=start_step,
            callbacks=callbacks,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            profiler=profiler,
            **kwargs
        )
        if apex is None:
            raise ModuleNotFoundError(
                "Apex is required by ApexDistributedDataParallelTrainer.",
                "Please refer to Quick Start in "
                "https://github.com/NVIDIA/apex.",
            )

        current_device = torch.cuda.current_device()
        assert current_device == self.device, "%d vs. %d" % (
            current_device,
            self.device,
        )

        self.model.cuda(self.device)
        # move optimizer to cuda
        if isinstance(self.optimizer, torch.optim.Optimizer):
            to_cuda(self.optimizer, self.device, inplace=True)
        if sync_bn:
            self.model = apex.parallel.convert_syncbn_model(self.model)

        self.model, self.optimizer = apex.amp.initialize(
            self.model,
            self.optimizer,
            opt_level=opt_level,
            keep_batchnorm_fp32=keep_batchnorm_fp32,
            loss_scale=loss_scale,
        )

        batch_processor.enable_apex = True
        self.model = apex.parallel.DistributedDataParallel(
            self.model,
            delay_allreduce=True,
        )


register_launcher("ApexDistributedDataParallelTrainer", launch)
register_launcher("apex_distributed_data_parallel_trainer", launch)
