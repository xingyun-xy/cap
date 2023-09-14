# Copyright (c) Changan Auto. All rights reserved.

import logging
import re
from typing import Any, Callable, Dict, Optional, Sequence, Union

from cap.core.event import EventStorage
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import (
    _as_list,
    is_list_of_type,
    to_flat_ordered_dict,
)
from .callbacks import CallbackMixin
from .model_tracking import MetricsStorageHandler

logger = logging.getLogger(__name__)

__all__ = [
    "update_metric_using_index",
    "update_metric_using_regex",
    "MetricUpdater",
]


def update_metric_using_index(
    per_metric_idxs: Sequence[Dict[str, Union[int, Sequence[int]]]]
) -> Callable:  # noqa: D205,D400
    """
    Use index to filter out necessary predictions and labels, then update the
    metrics.

    Args:
        per_metric_idxs: Specific label indexes in `batch`, and prediction
            indexes in `model_outs` for each metric.

    Examples::

        >>> from cap.metrics import Accuracy
        >>> metrics = [Accuracy(), Accuracy()]

        >>> from torch import Tensor
        >>> batch = [Tensor([1.0]), Tensor([2.0])]
        >>> model_outs = [Tensor([1.0]), Tensor([3.0])]

        >>> per_metric_idxs = [
        ...     dict(label_idx=0, pred_idx=0),  # for metrics[0]
        ...     dict(label_idx=1, pred_idx=1),  # for metrics[1]
        ... ]

        >>> update_func = update_metric_using_index(per_metric_idxs)
        >>> update_func(metrics, batch, model_outs)
        >>> metrics[0].get()
        ('accuracy', 1.0)
        >>> metrics[1].get()
        ('accuracy', 0.0)

    """
    per_metric_lb_idxs = []
    per_metric_pred_idxs = []
    for idx_dict in _as_list(per_metric_idxs):
        assert isinstance(idx_dict, dict), type(idx_dict)

        # label_idxs can be None for loss metric
        l_idxs = _as_list(idx_dict["label_idx"])
        if l_idxs is not None:
            assert is_list_of_type(l_idxs, element_type=int), l_idxs
        per_metric_lb_idxs.append(l_idxs)

        p_idxs = _as_list(idx_dict["pred_idx"])
        assert is_list_of_type(p_idxs, element_type=int), p_idxs
        per_metric_pred_idxs.append(p_idxs)

    def _update(
        metrics: Sequence, batch: Sequence, model_outs: Sequence
    ):  # noqa: D205,D400
        """
        Args:
            metrics: Metrics to update.
            batch: Original model inputs.
            model_outs: Original model outputs.
        """
        assert len(metrics) == len(
            per_metric_pred_idxs
        ), f"{len(metrics)} vs. {len(per_metric_pred_idxs)}"
        assert isinstance(batch, (list, tuple)), type(batch)
        assert isinstance(model_outs, (list, tuple)), type(model_outs)

        def _filter(idxs, list_data):
            res = []
            for i in idxs:
                if i is not None:
                    assert i < len(list_data), f"{i} vs. {len(list_data)}"
                    res.append(list_data[i])

            return res

        for metric, label_idxs, pred_idxs in zip(
            metrics, per_metric_lb_idxs, per_metric_pred_idxs
        ):
            labels = _filter(label_idxs, batch)
            preds = _filter(pred_idxs, model_outs)

            if len(preds) > 0:
                # labels can be empty for loss metric, but not preds
                metric(*labels, *preds)

    return _update


def update_metric_using_regex(
    per_metric_patterns: Sequence[Dict[str, Union[str, Sequence[str]]]],
    flat_condition: Optional[Callable[[Any], bool]] = None,
) -> Callable:  # noqa: D205,D400
    """Use regular expression to filter out necessary predictions and
    labels, then update the metrics.

    Args:
        per_metric_patterns: Specific label pattern and prediction pattern for
            each metric.
        flat_condition: Function with (`key`, `value`) as input,
            return `True/False` means whether flat this `values` or not, used
            when flatting `batch` and `model_outs`. See `to_flat_ordered_dict`
            for more.

    Examples::

        >>> from cap.metrics import Accuracy
        >>> metrics = [Accuracy(), Accuracy()]

        >>> from torch import Tensor
        >>> batch = dict(data=Tensor([1.0]), target=Tensor([2.0]))
        >>> model_outs = dict(pred1=Tensor([2.0]), pred2=Tensor([3.0]))

        >>> per_metric_patterns = [
        ...     dict(label_pattern='.*target.*', pred_pattern='.*pred1.*'),  # for metrics[0]  # noqa
        ...     dict(label_pattern='.*target.*', pred_pattern='.*pred2.*'),  # for metrics[1]  # noqa
        ... ]

        >>> update_func = update_metric_using_regex(per_metric_patterns)
        >>> update_func(metrics, batch, model_outs)
        >>> metrics[0].get()
        ('accuracy', 1.0)
        >>> metrics[1].get()
        ('accuracy', 0.0)

    """
    per_metric_lb_pat = []
    per_metric_pred_pat = []
    for pat_dict in _as_list(per_metric_patterns):
        assert isinstance(pat_dict, dict), type(pat_dict)

        # label_idxs can be None for loss metric
        lb_pat = pat_dict["label_pattern"]
        if lb_pat is not None:
            lb_pat = [re.compile(i) for i in _as_list(lb_pat)]
        per_metric_lb_pat.append(lb_pat)

        per_metric_pred_pat.append(
            [re.compile(i) for i in _as_list(pat_dict["pred_pattern"])]
        )

    def _update(
        metrics: Sequence, batch: Any, model_outs: Any
    ):  # noqa: D205,D400
        """
        Args:
            metrics: Metrics to update.
            batch: Original model inputs.
            model_outs: Original model outputs.
        """
        assert len(metrics) == len(
            per_metric_pred_pat
        ), f"{len(metrics)} vs. {len(per_metric_pred_pat)}"

        batch = to_flat_ordered_dict(batch, flat_condition=flat_condition)
        model_outs = to_flat_ordered_dict(
            model_outs, flat_condition=flat_condition
        )

        def _filter(patterns, dict_data):
            res = []
            if patterns is not None:
                # to keep res in same order as patterns
                for pat in patterns:
                    for k, v in dict_data.items():
                        if pat.match(k):
                            res.append(v)
            return res

        for metric, label_pat, pred_pat in zip(
            metrics, per_metric_lb_pat, per_metric_pred_pat
        ):
            labels = _filter(label_pat, batch)
            preds = _filter(pred_pat, model_outs)

            if len(preds) > 0:
                # labels can be empty for loss metric, but not preds
                metric.update(*labels, *preds)

    return _update


@OBJECT_REGISTRY.register
class MetricUpdater(CallbackMixin):
    """Callback used to reset, update, logging metrics.

    Args:
        metric_update_func: Function with `metrics`, `batch` and `model_outs`
            as inputs, filter out labels, predictions then update corresponding
            metric.
        metrics: Metric configs or metric instances, for multi-task.
        filter_condition: Function to filter current task metric inputs
            on batch end, including `model_outs` and `batch`. Useful in
            multitask training.
        step_log_freq: Logging every `step_log_freq` steps. If < 1, disable
            step log output.
        reset_metrics_by: When are metrics reset during training, can be
            either one of 'step', 'log' and 'epoch'.
        epoch_log_freq: Logging every `epoch_log_freq` epochs. This argument
            works only when reset_metric_by == 'epoch'.
        log_prefix: Logging info prefix.
    """

    def __init__(
        self,
        metric_update_func: Callable,
        metrics: Optional[Sequence] = None,
        filter_condition: Optional[Callable] = None,
        step_log_freq: Optional[int] = 1,
        reset_metrics_by: Optional[str] = "epoch",
        epoch_log_freq: Optional[int] = 1,
        log_prefix: Optional[str] = "",
    ):
        assert callable(metric_update_func)

        if metrics is None:
            self.metrics = None
        else:
            self.metrics = metrics
            self._reset_metrics(self.metrics)

        self.metric_update_func = metric_update_func
        self.filter_condition = filter_condition
        self.step_log_freq = step_log_freq
        assert reset_metrics_by in ("step", "log", "epoch")
        self.reset_metrics_by = reset_metrics_by
        self.epoch_log_freq = epoch_log_freq
        self.log_prefix = log_prefix

    def _reset_metrics(self, metrics):
        assert metrics is not None
        for m in metrics:
            m.reset()

    def on_loop_begin(self, loop, storage: EventStorage, **kwargs):
        if self.metrics is not None:
            for m in self.metrics:
                m.to(loop.device)
        logger.debug("begin producer ack")
        MetricsStorageHandler.producer_ack(storage)

    def on_batch_end(self, batch, model_outs, train_metrics, **kwargs):
        # filter task by filter_condition
        if self.filter_condition is not None and (
            not self.filter_condition(batch)
        ):
            return

        metrics = train_metrics if self.metrics is None else self.metrics
        self.metric_update_func(metrics, batch, model_outs)

    def on_step_end(
        self,
        epoch_id,
        step_id,
        global_step_id,
        train_metrics,
        storage: EventStorage,
        **kwargs,
    ):
        metrics = train_metrics if self.metrics is None else self.metrics

        if self.step_log_freq > 0 and (step_id + 1) % self.step_log_freq == 0:
            prefix = "Epoch[%d] Step[%d] GlobalStep[%d] %s: " % (
                epoch_id,
                step_id,
                global_step_id,
                self.log_prefix,
            )
            self._log(metrics, prefix)
            logger.debug("producer begin simple produce")
            MetricsStorageHandler.simple_produce(storage, metrics)

            if self.reset_metrics_by == "log":
                self._reset_metrics(metrics)

        if self.reset_metrics_by == "step":
            self._reset_metrics(metrics)

    def on_epoch_begin(self, train_metrics, **kwargs):
        metrics = train_metrics if self.metrics is None else self.metrics
        self._reset_metrics(metrics)

    def on_epoch_end(self, epoch_id, train_metrics, **kwargs):
        metrics = train_metrics if self.metrics is None else self.metrics
        if (epoch_id + 1) % self.epoch_log_freq == 0:
            prefix = "Epoch[%d] %s: " % (epoch_id, self.log_prefix)
            self._log(metrics, prefix)

    def _log(self, train_metrics, prefix=""):
        log_info = prefix
        for m in train_metrics:
            name, value = m.get()
            for k, v in zip(_as_list(name), _as_list(value)):
                if isinstance(v, (int, float)):
                    log_info += "%s[%.4f] " % (k, v)
                else:
                    log_info += "%s[%s] " % (str(k), str(v))
        logger.info(log_info)
