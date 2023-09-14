# Copyright (c) Changan Auto. All rights reserved.

import copy
import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch.nn as nn

from cap.engine.predictor import Predictor
from cap.registry import OBJECT_REGISTRY
from cap.utils.logger import MSGColor, format_msg
from .callbacks import CallbackMixin
from .checkpoint import get_valid_state_dict

__all__ = ["Validation"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Validation(CallbackMixin):  # noqa: D400
    """Callbacks of validation.

    Validation Callback does the following: \

    (1) use train model to forward batches from validation data loader.
    (2) use callbacks like `MetricUpdater` to record validation results. \

    You are suppose to think about two questions then provide \
    below arguments: \

    (1) Whether use train `model` as val model or provide one by `val_model`?
    (2) whether init val model with train model or not (you can init it by
    yourself)?

    Args:
        data_loader: Validation data loader.
        batch_processor: Validation batch processor (not need_grad_update).
        callbacks: Callbacks you want to run when doing validation, commonly
            it should contain callbacks like `MetricUpdater`, which is used to
            reset, update, logging validation metrics.
            If is empty list, no-op.
        val_interval: Validation interval.
        interval_by: Set `val_interval` unit to step or epoch.
            Default is epoch.
        val_model: Model used for validation.
            If None, use train model as val model.
        model_convert_pipeline: Define the process of model convert.
            e.g. convert float model to qat model, convert qat model
            to quantize model.
        init_with_train_model: Whether init val model with train model.
            If val_model is None, no-op.
        strict_match: Whether to strictly enforce that the keys in
            `model.state_dict()` (train model) match the keys in
            `val_model.state_dict()`. Default: ``False``
        val_on_train_end: Whether do validation when `on_loop_end` is
            triggered. Default is True.
        profiler: To profile individual steps during validation and
            assist in identifying bottlenecks.
        log_interval: Logging output frequency.
    """

    def __init__(
        self,
        data_loader: Iterable,
        batch_processor: Callable,
        callbacks: Sequence[Union[dict, CallbackMixin]],
        val_interval: Optional[int] = 1,
        interval_by: Optional[str] = "epoch",
        val_model: Union[Dict, nn.Module] = None,
        model_convert_pipeline: Optional[Union[Dict, List]] = None,
        init_with_train_model: Optional[bool] = True,
        strict_match: Optional[bool] = False,
        val_on_train_end: Optional[bool] = True,
        profiler: Optional[dict] = None,
        log_interval: Optional[int] = 10,
    ):
        self.predictor = Predictor(
            model=val_model,
            model_convert_pipeline=model_convert_pipeline,
            data_loader=data_loader,
            batch_processor=batch_processor,
            callbacks=callbacks,
            profiler=profiler,
            log_interval=log_interval,
        )

        self.val_interval = val_interval
        self.interval_by = interval_by
        assert self.interval_by in ("epoch", "step")

        self.val_model = None
        if val_model is not None:
            self.val_model = val_model

        self.init_with_train_model = init_with_train_model
        self.strict_match = strict_match
        self.val_on_train_end = val_on_train_end
        self.log_interval = log_interval

    def _select_and_init_val_model(self, train_model):
        # Question 1: use self.val_model or train_model as val model?
        if self.val_model is not None:

            logger.info(
                format_msg(
                    "Use `self.val_model` instead of train `model` as val model.",  # noqa: E501
                    MSGColor.GREEN,
                )
            )
            val_model = self.val_model

            # Question 2: whether init val model with train_model?
            if self.init_with_train_model:
                assert train_model is not None, (
                    "Try to init val model with train model, "
                    "but train model is None"
                )
                logger.info(
                    format_msg(
                        "Initialize val model with train `model`",
                        MSGColor.GREEN,
                    )
                )

                state = get_valid_state_dict(train_model)
                miss_key, unexpect_key = val_model.load_state_dict(
                    state, strict=self.strict_match
                )
                logger.info("miss_key: %s" % (" ".join(miss_key)))
                logger.info("unexpect_key: %s" % (" ".join(unexpect_key)))

            else:
                logger.info(
                    format_msg(
                        "Not init val model with train model, "
                        "make sure you want this.",
                        MSGColor.RED,
                    )
                )

        else:
            logger.info(
                format_msg("Use train `model` as val model.", MSGColor.GREEN)
            )
            # train_model's training status cant affected by val_model
            val_model = copy.deepcopy(train_model)

        return val_model

    def _do_val(self, epoch_id, model, ema_model, device, metrics):
        # save current model.training status
        if model.training:
            model_training = True
        else:
            model_training = False

        # select val model and init
        eval_model = model
        if ema_model is not None:
            eval_model = ema_model

        val_model = self._select_and_init_val_model(train_model=eval_model)

        val_model.eval()
        self.predictor.model = val_model
        self.predictor.train_metrics = metrics
        self.predictor.set_device(device)
        self.predictor.start_epoch = epoch_id
        self.predictor.fit()

        # resume model.training status before validation
        if model_training:
            model.train()

    def on_step_end(
        self,
        epoch_id,
        step_id,
        global_step_id,
        device,
        val_metrics,
        model=None,
        ema_model=None,
        optimizer=None,
        num_steps=None,
        **kwargs,
    ):
        if self.interval_by == "step" and (
            (global_step_id + 1) % self.val_interval == 0
        ):
            self._do_val(epoch_id, model, ema_model, device, val_metrics)

    def on_epoch_end(
        self,
        epoch_id,
        model,
        device,
        num_epochs,
        val_metrics,
        ema_model=None,
        **kwargs,
    ):
        if self.interval_by == "epoch" and (
            (epoch_id + 1) % self.val_interval == 0
        ):
            self._do_val(epoch_id, model, ema_model, device, val_metrics)

    def on_loop_end(
        self, model, device, epoch_id, val_metrics, ema_model=None, **kwargs
    ):
        if self.val_on_train_end:
            self._do_val(epoch_id, model, ema_model, device, val_metrics)
