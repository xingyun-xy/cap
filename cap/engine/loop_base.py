# Copyright (c) Changan Auto. All rights reserved.

import logging
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Dict, Iterable, List, Optional, Sequence, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler

from cap.callbacks import CallbackMixin
from cap.core import EventStorage
from cap.data.dataloaders.utils import get_len
from cap.profiler import PassThroughProfiler
from cap.utils.apply_func import _as_list
from cap.utils.distributed import get_device_count
from cap.utils.generator import prefetch_iterator
from cap.utils.global_var import get_value
from cap.utils.logger import MSGColor, format_msg
from .processors import BatchProcessorMixin

__all__ = ["PipeBase", "LoopBase"]

logger = logging.getLogger(__name__)


class PipeBase(ABC):
    """Base class for callbacks pipeline."""

    def __init__(
        self,
        callbacks: Optional[Sequence[CallbackMixin]] = None,
        profiler: Optional[dict] = None,
    ):

        self.name = self.__class__.__name__
        if profiler is None:
            profiler = PassThroughProfiler()
        self.profiler = profiler

        self.callbacks = []
        if callbacks is not None:
            self.callbacks = _as_list(callbacks)

    def set_callbacks(self, callbacks: Sequence[CallbackMixin] = None):
        self.callbacks = []
        if callbacks is None:
            return
        for cb in _as_list(callbacks):
            assert isinstance(cb, CallbackMixin), type(cb)
            self.callbacks.append(cb)

    def on_loop_begin(self, **kwargs):
        # Note: Not support *args, because different callback may have
        # different positional arguments.
        with self.profiler.profile(f"on_{self.name}_loop_begin"):
            for cb in self.callbacks:
                cb.on_loop_begin(**kwargs)

    def on_loop_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_loop_end"):
            for cb in self.callbacks:
                cb.on_loop_end(**kwargs)

    def on_epoch_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_epoch_begin"):
            for cb in self.callbacks:
                cb.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_epoch_end"):
            for cb in self.callbacks:
                cb.on_epoch_end(**kwargs)

    def on_step_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_step_begin"):
            for cb in self.callbacks:
                cb.on_step_begin(**kwargs)

    def on_step_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_step_end"):
            for cb in self.callbacks:
                cb.on_step_end(**kwargs)

    def on_batch_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_batch_begin"):
            for cb in self.callbacks:
                cb.on_batch_begin(**kwargs)

    def on_batch_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_batch_end"):
            for cb in self.callbacks:
                cb.on_batch_end(**kwargs)

    def on_backward_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_backward_begin"):
            for cb in self.callbacks:
                cb.on_backward_begin(**kwargs)

    def on_backward_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_backward_end"):
            for cb in self.callbacks:
                cb.on_backward_end(**kwargs)

    def on_optimizer_step_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_optimizer_step_begin"):
            for cb in self.callbacks:
                cb.on_optimizer_step_begin(**kwargs)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass


class LoopBase(PipeBase):  # noqa: D205,D400
    """LoopBase controls the data flow from `data_loader` to `model`, including
    model forward, loss backward and parameters update.

    It is hardware independent, run on cpu (device is None) or gpu (device is
    int gpu id).

    By setting `stop_by`, you are able to stop loop by counting epoch
    (default) or step.

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
        stop_by: Stop loop by counting epoch or step.
            If equal to 'epoch', stop loop when `epoch_id == num_epochs - 1`.
            If equal to 'step', stop loop when `global_step_id == num_steps - 1`.
            Default 'epoch'.
        num_epochs: Num of loop epochs, should be non-negative integer.
            If stop_by != 'epoch', no-op.
            Set 0 to skip loop epochs and run `self.on_*_loop_begin/end` only.
        start_epoch: Training start epoch, should be non-negative integer.
        num_steps: Num of loop steps, should be non-negative integer.
            If stop_by != 'step', no-op.
            Set 0 to skip loop steps and run `self.on_*_loop_begin/end` only.
        start_step: Training start step, should be non-negative integer.
        callbacks: Callback configs or instances.
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during loop and
            assist in identifying bottlenecks.
        log_interval: Logging output frequency.
    """  # noqa

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        batch_processor: BatchProcessorMixin,
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
    ):
        super(LoopBase, self).__init__(callbacks=callbacks, profiler=profiler)
        assert isinstance(device, int) or device is None, (
            "device should be int (gpu id) or None (run on cpu), but get %s"
            % type(device)
        )

        assert start_epoch >= 0, (
            f"{self.name} loop start epoch should be "
            f"non-negative integer, but get {start_epoch}"
        )
        assert start_step >= 0, (
            f"{self.name} loop start step should be "
            f"non-negative integer, but get {start_step}"
        )

        stop_by = stop_by.lower()
        if stop_by == "epoch":
            assert num_epochs is not None and num_epochs >= 0, (
                f"if stop {self.name} loop by counting epoch, num_epochs "
                f"should be non-negative integer, but get {num_epochs}"
            )
            assert num_epochs >= start_epoch, f"{num_epochs} vs. {start_epoch}"
            _skip_loop = num_epochs == 0

        elif stop_by == "step":
            assert num_steps is not None and num_steps >= 0, (
                f"if stop {self.name} loop by counting step, num_steps "
                f"should be non-negative integer, but get {num_steps}"
            )
            assert num_steps >= start_step, f"{num_steps} vs. {start_step}"
            _skip_loop = num_steps == 0

        else:
            raise ValueError(
                f"stop_by should be `epoch` or `step`, but get {stop_by}"
            )

        self.model = model
        self.ema_model = None
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self.train_metrics = _as_list(train_metrics)
        self.val_metrics = _as_list(val_metrics)
        self.device = device
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.num_steps = num_steps
        self.start_step = start_step
        self.log_interval = log_interval

        self._skip_loop = _skip_loop
        self._stop_by_epoch = stop_by == "epoch"
        self._stop_by_step = stop_by == "step"

        if model_convert_pipeline is not None:
            self.model = model_convert_pipeline(self.model)
        if callable(self.optimizer):
            self.optimizer = self.optimizer(self.model)
        self.checkpoint = get_value("model_checkpoint")

        if resume_optimizer:
            assert (
                self.checkpoint is not None
            ), "No checkpoint found, double check"
            self.resume_optimizer(resume_epoch_or_step)

            if resume_epoch_or_step:
                self.resume_epoch_or_step()

        self.storage = EventStorage()
        if self.device is not None:
            self.set_device(self.device)

    def resume_optimizer(self, resume_epoch_or_step):
        previous_gpu_num = self.checkpoint.get("devices", None)
        device_num = get_device_count()
        if previous_gpu_num is None:
            logger.warning(
                format_msg(
                    "The number of devices is not found in checkpoint."
                    " please ensure that the number of gpu devices is "
                    "the same as before when resuming",
                    MSGColor.RED,
                )
            )
        # else:
        #     assert previous_gpu_num == device_num, (
        #         f"The number of gpu devices should be "
        #         f"{previous_gpu_num}, but get {device_num}."
        #     )
        self.optimizer.load_state_dict(self.checkpoint["optimizer"])
        if not resume_epoch_or_step:
            for group in self.optimizer.param_groups:
                assert (
                    "lr" in group
                ), "Not found `lr` in a optimizer.param_groups"
                group["initial_lr"] = group["lr"]

    def resume_epoch_or_step(self):
        self.start_epoch = self.checkpoint["epoch"] + 1
        self.start_step = (
            self.checkpoint["step"] + 1
            if self._stop_by_step
            else self.checkpoint["step"]
        )
        logger.info(
            format_msg(
                "reset training `start_epoch` to %d and `start_step` to %d"
                % (self.start_epoch, self.start_step),
                MSGColor.GREEN,
            )
        )

    def set_device(self, device):
        self.device = device
        self.model.cuda(device)

        for m in chain(self.train_metrics, self.val_metrics):
            if m is not None:
                m.to(device)

    def on_epoch_begin(self, epoch_id, **kwargs):
        if hasattr(self.data_loader, "sampler"):
            sampler = self.data_loader.sampler
            if isinstance(sampler, dict):
                sampler = sampler.values()
            else:
                sampler = _as_list(sampler)

            for sa in sampler:
                if isinstance(sa, DistributedSampler):
                    sa.set_epoch(epoch_id)

        # need to set device again in case device has been changed before epoch begins # noqa
        if self.device is not None:
            self.set_device(self.device)
        super(LoopBase, self).on_epoch_begin(epoch_id=epoch_id, **kwargs)

    def get_callback(self, type_name):
        for cb in self.callbacks:
            if cb.__class__.__name__ == type_name:
                return cb
        return None

    def fit(self):
        """Do model fitting on data from data_loader.

        `self.batch_processor` responsible for model forward, loss backward and
        parameters update.

        `self.callbacks` responsible for metric update, checkpoint, logging and
        so on.
        """
        if self._skip_loop:
            msg = (
                f"Skip {self.name} loop and only run `on_*_loop_begin` and "
                f"`on_*_loop_end` as num_epochs={self.num_epochs}, "
                f"num_steps={self.num_steps}, one of them is 0"
            )
        else:
            if self._stop_by_epoch:
                msg = (
                    f"Start {self.name} loop from epoch {self.start_epoch}, "
                    f"num_epochs={self.num_epochs}"
                )
            elif self._stop_by_step:
                msg = (
                    f"Start {self.name} loop from step {self.start_step}, "
                    f"num_steps={self.num_steps}"
                )
            else:
                raise NotImplementedError

        logger.info(format_msg(msg, MSGColor.GREEN))

        # local vars
        epoch_id = self.start_epoch
        global_step_id = self.start_step
        end_loop_flag = self._skip_loop

        # TODO(linkai.liang, 0.5), not pass LoopBase to callback, not do resume
        #  in `Checkpoint` callback
        self.on_loop_begin(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            num_epochs=self.num_epochs,
            num_steps=self.num_steps,
            loop=self,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            storage=self.storage,
        )

        while not end_loop_flag:
            self.data_loader_pr = self.profiler.profile_iterable(
                enumerate(prefetch_iterator(self.data_loader)),
                f"get_{self.name}_batch_data",
            )
            self.on_epoch_begin(
                model=self.model,
                epoch_id=epoch_id,
                optimizer=self.optimizer,
                global_step_id=global_step_id,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
            )

            for step_id, (batch, _is_last_batch) in tqdm(self.data_loader_pr):
                if self.log_interval > 0 and step_id % self.log_interval == 0:
                    logger.info(f"{step_id} / {get_len(self.data_loader)}")
                self.on_step_begin(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    data_loader=self.data_loader,
                    start_epoch=self.start_epoch,
                    start_step=self.start_step,
                    global_step_id=global_step_id,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                )

                self.batch_processor(
                    batch,
                    self.model,
                    self.device,
                    optimizer=self.optimizer,
                    batch_begin_callback=partial(
                        self.on_batch_begin,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                        train_metrics=self.train_metrics,
                        val_metrics=self.val_metrics,
                        storage=self.storage,
                    ),
                    batch_end_callback=partial(
                        self.on_batch_end,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                        train_metrics=self.train_metrics,
                        val_metrics=self.val_metrics,
                        storage=self.storage,
                    ),
                    backward_begin_callback=partial(
                        self.on_backward_begin,
                        model=self.model,
                        optimizer=self.optimizer,
                    ),
                    backward_end_callback=partial(
                        self.on_backward_end,
                        model=self.model,
                        optimizer=self.optimizer,
                    ),
                    optimizer_step_begin_callback=partial(
                        self.on_optimizer_step_begin,
                        model=self.model,
                        optimizer=self.optimizer,
                    ),
                    profiler=self.profiler,
                )

                self.on_step_end(
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                    data_loader=self.data_loader,
                    model=self.model,
                    ema_model=self.ema_model,
                    optimizer=self.optimizer,
                    num_steps=self.num_steps,
                    device=self.device,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                )

                global_step_id += 1
                if self._stop_by_step and global_step_id >= self.num_steps:
                    end_loop_flag = True
                    break

            self.on_epoch_end(
                epoch_id=epoch_id,
                global_step_id=global_step_id,
                model=self.model,
                ema_model=self.ema_model,
                optimizer=self.optimizer,
                num_epochs=self.num_epochs,
                device=self.device,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
            )

            epoch_id += 1
            if self._stop_by_epoch and epoch_id >= self.num_epochs:
                end_loop_flag = True

        self.on_loop_end(
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            epoch_id=epoch_id,
            global_step_id=global_step_id,
            device=self.device,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            callbacks=self.callbacks,
            storage=self.storage,
        )
        self.profiler.describe()
