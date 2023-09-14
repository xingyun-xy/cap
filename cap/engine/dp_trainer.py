# Copyright (c) Changan Auto. All rights reserved.

import logging
import os
from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn

from cap.callbacks import CallbackMixin
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, to_cuda
from .launcher import register_launcher
from .processors import BatchProcessorMixin
from .trainer import Trainer

__all__ = ["DataParallelTrainer"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("data_parallel_trainer")
class DataParallelTrainer(Trainer):
    """DataParallelTrainer is a tool function to new a `Trainer` instance.

    which training with `DataParallel` method, and running on multiple gpu
    devices.

    It can be launched by launch function below.

    By setting `stop_by`, you are able to stop training by counting epoch
    (default) or step.

    Args:
        model: Model config or a `nn.Module` instance.
        data_loader: Training data loader config or a instantiated data loader.
        optimizer: Optimizer config or a optimizer instance.
        batch_processor: Batch processor config or a `BatchProcessorMixin`
            instance.
        device: GPU ids.
        stop_by: Stop training by counting epoch or step. If equal to
            'epoch', stop training when `epoch_id == num_epochs - 1`. If
            equal to 'step', stop training
            when `global_step_id == num_steps - 1`.
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
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        batch_processor: BatchProcessorMixin,
        device: Union[int, Sequence[int]],
        stop_by: Optional[str] = "epoch",
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = 0,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = 0,
        callbacks: Optional[Sequence[Union[dict, CallbackMixin]]] = None,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        **kwargs
    ):

        device_ids = _as_list(device)
        assert len(device_ids) > 0
        current_device = device_ids[0]
        torch.cuda.set_device(current_device)
        super(DataParallelTrainer, self).__init__(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            batch_processor=batch_processor,
            device=current_device,  # copy model, data to current device
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

        self.model.cuda(current_device)
        # move optimizer to cuda
        if isinstance(self.optimizer, torch.optim.Optimizer):
            to_cuda(self.optimizer, current_device, inplace=True)
        if len(device_ids) > 1:
            assert not isinstance(
                self.model, nn.parallel.DataParallel
            ), "is already a `DataParallel` instance"
            self.model = nn.DataParallel(self.model, device_ids=device_ids)


def launch(main_func, device_ids, dist_url=None, dist_launcher=None, args=()):
    device_ids = _as_list(device_ids)
    num_devices = len(device_ids)
    assert num_devices > 0

    # Note: if device_ids=[1, 3], then after setting `CUDA_VISIBLE_DEVICES`,
    # new device_ids=[0, 1].
    str_ids = list(map(str, device_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
    device_ids = list(range(num_devices))

    main_func(device_ids, *args)


register_launcher("data_parallel_trainer", launch)
register_launcher("DataParallelTrainer", launch)
