# Copyright (c) Changan Auto. All rights reserved.

from abc import ABC, abstractmethod
from decimal import localcontext
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torchvision
from torch.cuda.amp import GradScaler, autocast

from cap.profiler import BaseProfiler, PassThroughProfiler
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, to_cuda

try:
    import apex
except ImportError:
    apex = None

__all__ = ["BatchProcessorMixin", "BasicBatchProcessor", "MultiBatchProcessor"]


class BatchProcessorMixin(ABC):
    """Batch Processor Interface."""

    @abstractmethod
    def __call__(
        self,
        batch: Union[Tuple[Any], List[Any], object],
        model: torch.nn.Module,
        device: Union[int, None],
        optimizer=None,
        batch_begin_callback: Callable = None,
        batch_end_callback: Callable = None,
        backward_begin_callback: Callable = None,
        backward_end_callback: Callable = None,
        optimizer_step_begin_callback: Callable = None,
    ):
        pass


@OBJECT_REGISTRY.register
class BasicBatchProcessor(BatchProcessorMixin):  # noqa: D205,D400
    """
    Processor dealing with `(inputs, target)` batch, and the model output is a
    `(losses, preds)` pair.

    It is suitable for training (need_grad_update) or validation
    (not need_grad_update).

    Args:
        need_grad_update: Whether need gradient update, True for training,
            False for Validation.
        batch_transforms: Config of batch transforms.
        loss_collector: A callable object used to collect loss Tensors in model
            outputs.
        enable_amp: Whether training with `Automatic Mixed Precision`.
        enabel_apex: Whether training with `Apex`.
        grad_scaler: An instance ``scaler`` of :class:`GradScaler`
            helps perform the steps of gradient scaling conveniently.
    """

    def __init__(
        self,
        need_grad_update: bool,
        batch_transforms: Optional[List] = None,
        loss_collector: Callable = None,
        enable_amp: bool = False,
        enable_apex: bool = False,
        grad_scaler: torch.cuda.amp.GradScaler = None,
    ):
        if need_grad_update:
            assert (
                loss_collector is not None
            ), "Provide `loss_collector` when need_grad_update"
            assert callable(loss_collector)
        if enable_amp and enable_apex:
            raise RuntimeError(
                "enable_amp and enable_apex cannot be true together."
            )
        if enable_apex and apex is None:
            raise ModuleNotFoundError("Apex is required.")

        self.need_grad_update = need_grad_update
        self.enable_apex = enable_apex
        self.loss_collector = loss_collector
        if grad_scaler is not None:
            self.grad_scaler = grad_scaler
        else:
            self.grad_scaler = GradScaler(enabled=enable_amp)
        self.enable_amp = self.grad_scaler.is_enabled()
        if enable_amp:
            assert self.enable_amp, (
                "When grad_scaler is not None, enable_amp does not work."
                "You set enable_amp is {}, but the enable_amp of "
                "grad_scaler is {}. Please check your config!!"
            ).format(enable_amp, self.enable_amp)
        if batch_transforms:
            if isinstance(batch_transforms, (list, tuple)):
                batch_transforms = torchvision.transforms.Compose(
                    batch_transforms
                )  # noqa
            self.transforms = batch_transforms
        else:
            self.transforms = None

    def __call__(
        self,
        batch: Union[Tuple[Any], List[Any], object],
        model: torch.nn.Module,
        device: Union[int, None],
        optimizer=None,
        batch_begin_callback: Callable = None,
        batch_end_callback: Callable = None,
        backward_begin_callback: Callable = None,
        backward_end_callback: Callable = None,
        optimizer_step_begin_callback: Callable = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
    ):
        assert self.need_grad_update == model.training, (
            "%s vs. %s, set model to training/eval mode by "
            "model.train()/model.eval() when need_grad_update or not"
            % (self.need_grad_update, model.training)
        )

        if batch_begin_callback is not None:
            batch_begin_callback(batch=batch)

        if profiler is None:
            profiler = PassThroughProfiler()

        # 0. reset grad
        if self.need_grad_update:
            with profiler.profile("optimizer_zero_grad"):
                optimizer.zero_grad()

        if device is not None:
            batch = to_cuda(batch, device, non_blocking=True)
        else:
            # run on cpu
            pass

        if self.transforms is not None:
            with profiler.profile("gpu_transforms"):
                batch = self.transforms(batch)

        # 1. forward
        grad_decorator = (
            torch.enable_grad if self.need_grad_update else torch.no_grad
        )
        if not self.enable_apex:
            auto_cast = autocast(enabled=self.enable_amp)
        else:
            auto_cast = localcontext()
        with profiler.profile("model_forward"):
            with auto_cast:
                with grad_decorator():
                    # model outputs can be in any format
                    model_outs = model(*_as_list(batch))

        # 2. filter out loss Tensors in model outputs
        if self.loss_collector is not None:
            losses = self.loss_collector(model_outs)
        else:
            losses = None

        # 2. backward & step
        if self.need_grad_update:
            # Not allow to backward each loss independently, so sum them
            loss = sum([loss for loss in _as_list(losses) if loss is not None])
            assert isinstance(loss, torch.Tensor), type(loss)
            loss_scalar = loss.sum()

            # when grad_scaler is not enable, equivalent to loss.backward()
            with profiler.profile("model_backward"):
                if backward_begin_callback:
                    backward_begin_callback()
                if self.enable_apex:
                    with apex.amp.scale_loss(loss_scalar, optimizer) as loss_s:
                        loss_s.backward()
                else:
                    self.grad_scaler.scale(loss_scalar).backward()
                if backward_end_callback:
                    backward_end_callback()

            # when grad_scaler is not enable, equivalent to optimizer.step()
            with profiler.profile("optimizer_step"):
                if optimizer_step_begin_callback is not None:
                    optimizer_step_begin_callback(grad_scaler=self.grad_scaler)
                if self.enable_apex:
                    optimizer.step()
                else:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()

        if batch_end_callback is not None:
            batch_end_callback(
                batch=batch,
                losses=losses,
                model_outs=model_outs,
            )


@OBJECT_REGISTRY.register
class MultiBatchProcessor(BatchProcessorMixin):
    """
    Processor can forward backward multiple batches within a training step (before `optimizer.step()`).

    It is useful for:

    (1) Training a multitask model on single task annotation samples, of which
    each task forward backward its batch sequentially within a multitask training step

    (2) Training on a memory shortage GPU and want to increase batch size,
    you are able to forward backward multiple batches within a training step

    .. note::

        Example multitask: vehicle, person and traffic light detection.
        Single task annotation means only annotate vehicle bounding boxes on an image with vehicle,
        person, and traffic light objects.

    .. note::

        Multiple batches should be organized in tuple format, e.g.

        * `batch = (batch1, batch2, ...)`

        If not, it will be treated as a single batch, e.g.

        * `batch = dict(inputs=xx, target=xx)`

        * `batch = [inputs, target]`

        See code below for extra explanation.

    It is much general in usage than `BasicBatchProcessor` , batch and model
    outputs can be in any format, but note that if batch is a tuple means it contains multiple batches.

    It is Hardware independent, run on cpu (device None) or gpu
    (device is gpu id).

    It is suitable for training (need_grad_update) and validation
    (not need_grad_update).

    Args:
        need_grad_update: Whether need gradient update, True for training,
            False for Validation.
        batch_transforms: Config of batch transforms.
        loss_collector: A callable object used to collect loss Tensors in model
            outputs.
        enable_amp: Whether training with `Automatic Mixed Precision`.
        enabel_apex: Whether training with `Apex`.
        delay_sync: Whther delay sync grad when train on DDP.
            Refer to: DDP.no_sync() API
        grad_scaler: An instance ``scaler`` of :class:`GradScaler`
            helps perform the steps of gradient scaling conveniently.
    """  # noqa

    def __init__(
        self,
        need_grad_update: bool,
        batch_transforms: Optional[List] = None,
        loss_collector: Callable = None,
        enable_amp: bool = False,
        enable_apex: bool = False,
        delay_sync: bool = False,
        grad_scaler: torch.cuda.amp.GradScaler = None,
    ):
        if need_grad_update:
            assert (
                loss_collector is not None
            ), "Provide `loss_collector` when need_grad_update"
            assert callable(loss_collector)
        if enable_amp and enable_apex:
            raise RuntimeError(
                "enable_amp and enable_apex cannot be true together."
            )
        if enable_apex and apex is None:
            raise ModuleNotFoundError("Apex is required.")
        if delay_sync:
            assert apex is not None, "Delay sync only support env with apex."

        self.need_grad_update = need_grad_update
        self.loss_collector = loss_collector
        self.enable_apex = enable_apex
        self.delay_sync = delay_sync
        if grad_scaler is not None:
            self.grad_scaler = grad_scaler
        else:
            self.grad_scaler = GradScaler(enabled=enable_amp)
        self.enable_amp = self.grad_scaler.is_enabled()
        if enable_amp:
            assert self.enable_amp, (
                "When grad_scaler is not None, enable_amp does not work."
                "You set enable_amp is {}, but the enable_amp of "
                "grad_scaler is {}. Please check your config!!"
            ).format(enable_amp, self.enable_amp)
        if batch_transforms:
            if isinstance(batch_transforms, (list, tuple)):
                batch_transforms = torchvision.transforms.Compose(
                    batch_transforms
                )  # noqa
            self.transforms = batch_transforms
        else:
            self.transforms = None

    def __call__(
        self,
        batch: Union[Tuple[Any], List[Any], object],
        model: torch.nn.Module,
        device: Union[int, None],
        optimizer=None,
        batch_begin_callback: Callable = None,
        batch_end_callback: Callable = None,
        backward_begin_callback: Callable = None,
        backward_end_callback: Callable = None,
        optimizer_step_begin_callback: Callable = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
    ):
        assert self.need_grad_update == model.training, (
            "%s vs. %s, set model to training/eval mode by "
            "model.train()/model.eval() when need_grad_update or not"
            % (self.need_grad_update, model.training)
        )

        if profiler is None:
            profiler = PassThroughProfiler()

        # 0. reset grad
        if self.need_grad_update:
            with profiler.profile("optimizer_zero_grad"):
                optimizer.zero_grad()

        if isinstance(batch, tuple):
            # Means that `batch_data` contains multiple batches, e.g.
            # (1) contains task specific batches of a `multitask model`
            # batch_data = (
            #    [task1_data1, task1_data2, ...],   # task1 batch
            #    [task2_data1, task2_data2, ...],   # task2 batch
            #    [task3_data1, task3_data2, ...],   # can be list/tuple of objs
            #    task4_data                         # or just a single obj
            #    ...
            # )
            #
            # (2) contains multiple batches for a single task model
            # batch_data = (
            #    [batch1_data1, batch1_data2, ...],
            #    [batch2_data1, batch2_data2, ...], # can be list/tuple of objs
            #    data1                              # or just a single obj
            #    ...
            # )
            batches = batch
        else:
            # Means that `data` just contains a single batch, e.g.
            # (1) is a single obj
            # batch_data = task_data  # e.g. a dict(inputs=xx, target=xx)
            #
            # (2) is a list (NOT A TUPLE) of objs
            # batch_data = [task_data1, task_data2, ...]
            #
            # convert to tuple
            batches = (batch,)

        # for each batch in multiple batches
        last_batch_idx = len(batches) - 1
        for idx, batch_i in enumerate(batches):
            if batch_begin_callback is not None:
                batch_begin_callback(batch=batch_i)

            if device is not None:
                batch_i = to_cuda(batch_i, device, non_blocking=True)
            else:
                # run on cpu
                pass

            if self.transforms is not None:
                with profiler.profile("gpu_transforms"):
                    batch_i = (self.transforms(batch_i[0]), batch_i[1])

            # 1. forward
            grad_decorator = (
                torch.enable_grad if self.need_grad_update else torch.no_grad
            )
            if not self.enable_apex:
                auto_cast = autocast(enabled=self.enable_amp)
            else:
                auto_cast = localcontext()
            with profiler.profile("model_forward"):
                with auto_cast:
                    with grad_decorator():
                        if (
                            apex is not None
                            and self.delay_sync
                            and idx != last_batch_idx
                            and isinstance(
                                model, apex.parallel.DistributedDataParallel
                            )
                        ):
                            # delay sync grad until last batch
                            model.disable_allreduce()
                            model_outs = model(*_as_list(batch_i))
                        else:
                            # only support enable_allreduce in apex
                            if hasattr(model, "enable_allreduce"):
                                model.enable_allreduce()
                            # model outputs can be in any format
                            model_outs = model(*_as_list(batch_i))

            # 2. filter out loss Tensors in model outputs
            if self.loss_collector is not None:
                losses = self.loss_collector(model_outs)
            else:
                losses = None

            torch.cuda.empty_cache()

            # 3. backward
            if self.need_grad_update:
                # Not allow to backward each loss independently, so sum them
                loss = sum(
                    [loss for loss in _as_list(losses) if loss is not None]
                )
                assert isinstance(loss, torch.Tensor), type(loss)
                loss_scalar = loss.sum()
                if backward_begin_callback:
                    backward_begin_callback(batch=batch_i)
                # when grad_scaler is not enable, equivalent to loss.backward()
                with profiler.profile("model_backward"):
                    if self.enable_apex:
                        with apex.amp.scale_loss(
                            loss_scalar, optimizer
                        ) as loss_s:
                            loss_s.backward()
                    else:
                        self.grad_scaler.scale(loss_scalar).backward()
                if backward_end_callback:
                    backward_end_callback(batch=batch_i)

            if batch_end_callback is not None:
                batch_end_callback(
                    batch=batch_i, losses=losses, model_outs=model_outs
                )

        # 4. update grad
        if self.need_grad_update:
            if optimizer_step_begin_callback is not None:
                optimizer_step_begin_callback(grad_scaler=self.grad_scaler)
            # when grad_scaler is not enable, equivalent to optimizer.step()
            with profiler.profile("optimizer_step"):
                if self.enable_apex:
                    optimizer.step()
                else:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
