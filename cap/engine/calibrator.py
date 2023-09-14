# Copyright (c) Changan Auto. All rights reserved.
# type: ignore

import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Union

import changan_plugin_pytorch as changan
import torch
import torch.nn as nn

from cap.callbacks import CallbackMixin
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from cap.utils.logger import MSGColor, format_msg
from .launcher import register_launcher
from .loop_base import LoopBase
from .processors import BatchProcessorMixin

__all__ = ["Calibrator"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class Calibrator(LoopBase):
    """Calibrator is a tool for calibration.

    The abundant callbacks in trainer is also supported.

    Args:
        model: `nn.Module` instance.
        data_loader: Validation data loader.
        batch_processor: Batch processor config.
        device: Int gpu id or None.
        model_convert_pipeline: Define the process of model convert.
            e.g. convert float model to qat model, convert qat model
            to quantize model.
        num_steps: Num of calibration steps, should be non-negative integer.
        callbacks: Callbacks.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        log_interval: Logging output frequency.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        batch_processor: BatchProcessorMixin,
        device: Optional[int] = None,
        model_convert_pipeline: Optional[Union[Dict, List]] = None,
        num_steps: Optional[int] = None,
        callbacks: Optional[Sequence[Union[dict, CallbackMixin]]] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        log_interval: int = 0,
        **kwargs
    ):
        deploy_model = kwargs.get("deploy_model", None)
        deploy_inputs = kwargs.get("deploy_inputs", None)
        val_model = kwargs.get("val_model", None)
        self.callbacks = _as_list(callbacks)
        val_callback = self.get_callback("Validation")
        if val_callback is not None:
            val_callback.val_model = val_model
        ckpt_callback = self.get_callback("Checkpoint")
        if ckpt_callback is not None:
            ckpt_callback.deploy_model = deploy_model
            ckpt_callback.deploy_inputs = deploy_inputs
        super(Calibrator, self).__init__(
            model=model,
            data_loader=data_loader,
            optimizer=None,
            batch_processor=batch_processor,
            device=device,
            model_convert_pipeline=model_convert_pipeline,
            num_steps=num_steps,
            stop_by="step",
            callbacks=self.callbacks,
            val_metrics=val_metrics,
            profiler=profiler,
            log_interval=log_interval,
        )

        if len(self.callbacks) == 0:
            logger.info(
                format_msg(
                    "`callbacks` is empty, make sure you want this",
                    MSGColor.RED,
                )
            )

        if batch_processor.need_grad_update:
            batch_processor.need_grad_update = False
        assert batch_processor.need_grad_update is False

    def on_loop_begin(self, **kwargs):
        self.model.eval()
        return super().on_loop_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        self.model.set_qconfig()
        changan.quantization.prepare_qat(self.model, inplace=True)
        return super().on_epoch_end(**kwargs)


def launch(
    main_func, device_ids=None, dist_url=None, dist_launcher=None, args=()
):
    if device_ids is None:
        current_device = None
    else:
        device_ids = _as_list(device_ids)
        if len(device_ids) > 1:
            msg = (
                "`Calibrator` only support one device, but get %s, now "
                "only use device %d." % (device_ids, device_ids[0])
            )
            logger.info(format_msg(msg, MSGColor.GREEN))
            device_ids = device_ids[:1]

        # Note: if device_ids=[3], then after setting
        # `CUDA_VISIBLE_DEVICES`, index of gpu 3 becomes 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])
        current_device = 0
        torch.cuda.set_device(current_device)

    # only support run on cpu0-gpu0 while using mpi
    if dist_launcher == "mpi":
        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        local_rank = comm.Get_rank()
        if local_rank > 0:
            return

    main_func(current_device, *args)


register_launcher("Calibrator", launch)
