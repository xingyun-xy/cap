# Copyright (c) Changan Auto. All rights reserved.
# type: ignore

import logging
import os
import signal
from typing import Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cap.callbacks import CallbackMixin
from cap.core.task_sampler import TaskSampler
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, to_cuda
from cap.utils.distributed import (
    find_free_port,
    get_local_process_group,
    split_process_group_by_host,
)
from .launcher import register_launcher
from .processors import BatchProcessorMixin
from .trainer import Trainer

__all__ = ["DistributedDataParallelTrainer", "launch"]

logger = logging.getLogger(__name__)


def convert_sync_bn(model, process_group=None, local_sync=True):
    if local_sync:
        process_group, change = split_process_group_by_host(process_group)
        if change:
            logger.info("SyncBatcnNorm has been set to use local host sync.")

    return nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("distributed_data_parallel_trainer")
class DistributedDataParallelTrainer(Trainer):
    """DistributedDataParallelTrainer tool.

    DistributedDataParallelTrainer is a tool function to new a `Trainer`
    instance, which training with `DistributedDataParallel` method,
    and running on one of the GPU devices.

    It can be launched by launch function below, which spawns multiple
    processes and each of it owns an independent Trainer.

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
        sync_bn_by_host: Whether sync bn within host node
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        task_sampler: TaskSampler config for multitask training.
        convert_submodule_list: List of submodule for converting DDP.
        find_unused_parameters: Args of DistributedDataParallel module.
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
        sync_bn_by_host: Optional[bool] = False,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        task_sampler: Optional[TaskSampler] = None,
        convert_submodule_list: Optional[List[str]] = None,
        find_unused_parameters: Optional[bool] = True,
        **kwargs
    ):  # noqa: D205,D400
        super(DistributedDataParallelTrainer, self).__init__(
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
        assert isinstance(self.device, int), (
            "%s, run `DistributedDataParallel` model"
            " only on one gpu" % type(self.device)
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
        assert not isinstance(
            self.model, nn.parallel.DistributedDataParallel
        ), "is already a `DistributedDataParallel` instance"
        if sync_bn:
            self.model = convert_sync_bn(
                self.model,
                process_group=get_local_process_group(),
                local_sync=sync_bn_by_host,
            )

        if bool(int(os.environ.get("CAP_USE_CHECKPOINT", "0"))):
            # Set `False` when using checkpoint to enlarge batch size.
            # Or would raise bug.
            assert (
                not find_unused_parameters
            ), "Cannot find unused parameter while using checkpoint."

        broadcast_buffers = True
        if task_sampler is not None and task_sampler.is_parallel():
            # task parallel must set find unused parameters to True
            find_unused_parameters = True
            broadcast_buffers = False

        if convert_submodule_list:
            # support to convert DDP with submodule
            convert_submodule_list = _as_list(convert_submodule_list)
            for submodule in convert_submodule_list:
                module = getattr(self.model, submodule)
                module = nn.parallel.DistributedDataParallel(
                    module,
                    find_unused_parameters=find_unused_parameters,
                    broadcast_buffers=broadcast_buffers,
                    device_ids=[device],
                )
                setattr(self.model, submodule, module)
        else:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=broadcast_buffers,
                device_ids=[self.device],
            )


def launch(
    main_func,
    device_ids,
    dist_url="auto",
    dist_launcher=None,
    num_processes=None,
    backend="NCCL",
    args=(),
):
    device_ids = _as_list(device_ids)
    num_devices = len(device_ids)
    assert num_devices > 0

    num_processes = num_processes if num_processes else num_devices
    assert num_processes > 0
    if num_processes == num_devices and backend != "NCCL":
        logger.warning(
            "NCCL is the best choice in case of single process on single gpu."
        )

    # Note: if device_ids=[1, 3], then after setting `CUDA_VISIBLE_DEVICES`,
    # new device_ids=[0, 1].
    str_ids = list(map(str, device_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)

    if dist_url == "auto":
        port = find_free_port()  #找到空闲端口使用（good）
        dist_url = "tcp://localhost:%s" % port

    if dist_launcher is not None:
        if dist_launcher == "mpi":
            _main_mpi(
                main_func, dist_url, backend, num_devices, num_processes, args
            )
        else:
            raise TypeError("unknown dist_launcher: %s" % dist_launcher)
    else:
        try:
            mp.spawn(
                _main_func,
                nprocs=num_processes,
                args=(
                    main_func,
                    dist_url,
                    backend,
                    num_devices,
                    num_processes,
                    args,
                ),
            )
        # when press Ctrl+c, all sub processes will exits too.
        except KeyboardInterrupt as exception:
            logger.exception(str(exception))
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


register_launcher("DistributedDataParallelTrainer", launch)
register_launcher("distributed_data_parallel_trainer", launch)


def _main_func(
    local_rank, main_func, dist_url, backend, num_devices, num_processes, args
):
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=num_processes,
            rank=local_rank,
        )
    except Exception as e:
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    torch.cuda.set_device(local_rank % num_devices)
    main_func(local_rank, *args)


def _main_mpi(main_func, dist_url, backend, num_devices, num_processes, args):
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    world_size = comm.Get_size()
    logger.info("MPI local_rank=%d, world_size=%d" % (local_rank, world_size))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=local_rank,
        )
    except Exception as e:
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    current_device = local_rank % num_devices
    torch.cuda.set_device(current_device)
    main_func(current_device, *args)
