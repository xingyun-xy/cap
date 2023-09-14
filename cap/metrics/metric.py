# Copyright (c) Changan Auto. All rights reserved.

import functools
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup
from torchmetrics.metric import Metric
from torchmetrics.utilities import apply_to_collection
from torchmetrics.utilities.data import _flatten, dim_zero_cat
from torchmetrics.utilities.distributed import gather_all_tensors

__all__ = ["EvalMetric"]

logger = logging.getLogger(__name__)


class EvalMetric(Metric, ABC):
    """Base class for all evaluation metrics.

    Built on top of `torchmetrics.metric.Metric`, this base class introduces
    the name attribute and a name-value format output (the get method). It
    also makes possible to syncnronize state tensors of different shapes in
    each device to support AP-like metrics.


    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but inherit it to create new
        metric classes instead.

    Args:
        name: Name of this metric instance for display.
        process_group:  Specify the process group on which synchronization is
            called. Default: None (which selects the entire world)
        warn_without_compute: Whether to output warning log if `self.compute`
            is not called in `self.get`. Since synchronization among devices
            is executed in `self.compute`, this value reflects if the metric
            will support distributed computation.
    """

    def __init__(
        self,
        name: Union[List[str], str],
        process_group: Optional[ProcessGroup] = None,
        warn_without_compute: bool = True,
    ):

        self.name = name
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
            process_group=process_group,
        )
        self._warn_without_compute = warn_without_compute
        self.get = self._wrap_get(self.get)
        self._init_states()

    def _init_states(self):
        """Initialize state variables.

        It is generally recommended to create state variables with
        add_state method, since in that case synchronization and reset
        can be handled automatically.

        State variables manually added with `self.xxx = yyy` cannot be
        synchronized and thus cannot be used in distributed case. Besides,
        they need to be manually reset by extending the reset method.
        """

        self.add_state(
            "sum_metric",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def reset(self) -> None:
        """Reset the metric state variables to their default value.

        If （and only if） there are state variables that are not registered
        with `self.add_state` need to be regularly set to default values,
        please extend this method in subclasses.
        """
        super().reset()

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Optional[Any] = None,
    ) -> None:
        input_dict = {
            attr: getattr(self, attr) for attr in self._reductions.keys()
        }

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists
            # to reduce number of all_gather operations
            input_dict[attr] = input_dict[attr].cuda()
            if (
                reduction_fn == dim_zero_cat
                and isinstance(input_dict[attr], list)
                and len(input_dict[attr]) > 1
            ):
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

        output_dict = apply_to_collection(
            input_dict,
            torch.Tensor,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)
            if isinstance(output_dict[attr][0], torch.Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            assert isinstance(reduction_fn, Callable) or reduction_fn is None
            reduced = (
                reduction_fn(output_dict[attr])
                if reduction_fn is not None
                else output_dict[attr]
            )
            setattr(self, attr, reduced)

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method to update the state variables."""

    def compute(self) -> Union[float, List[float]]:
        """Override this method to compute final results from metric states.

        All states variables registered with `self.add_state` are synchronized
        across devices before the execution of this method.
        """
        val = self.sum_metric / self.num_inst

        # scalar case
        if val.numel() == 1:
            val = val.item()
        else:
            val = val.cpu().numpy().tolist()
        return val

    def __getstate__(self) -> Dict[str, Any]:
        # ignore update and compute functions for pickling
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["update", "compute", "get", "_update_signature"]
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)
        self.compute: Callable = self._wrap_compute(self.compute)
        self.get: Callable = self._wrap_get(self.get)

    def get(self) -> Tuple[Union[str, List[str]], Union[float, List[float]]]:
        """Get current evaluation result.

        To skip the synchronization among devices, please override this method
        and calculate results without calling `self.compute()`.

        Returns:
           names: Name of the metrics.
           values: Value of the evaluations.
        """

        values = self.compute_2()
        if isinstance(values, list):
            assert isinstance(self.name, list) and len(self.name) == len(
                values
            )

        return self.name, values

    def _wrap_get(self, get: Callable):
        @functools.wraps(get)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            res = get(*args, **kwargs)
            if self._warn_without_compute and self._computed is None:
                logger.warning(
                    f"{self.__class__} not ready for distributed environment,"
                    + " should not be used together with DistributedSampler."
                    + "Might be slow in validation due to resource competition"
                )
            return res

        return wrapped_func

    def get_name_value(self):
        """
        Return zipped name and value pairs.

        Returns:
            List(tuples): A (name, value) tuple list.
        """

        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))
