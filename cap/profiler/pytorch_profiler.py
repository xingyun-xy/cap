# Copyright (c) Changan Auto. All rights reserved.
"""
This file is modified from pytorch-lightning.

checking if there are any bottlenecks in your code.
"""
import inspect
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import torch
from torch.autograd.profiler import EventList, record_function
from torch.profiler import (
    ProfilerAction,
    ProfilerActivity,
    tensorboard_trace_handler,
)

from cap.registry import OBJECT_REGISTRY
from cap.utils.logger import rank_zero_warn
from .profilers import BaseProfiler

_PROFILER = Union[
    torch.autograd.profiler.profile,
    torch.cuda.profiler.profile,
    torch.autograd.profiler.emit_nvtx,
]


class ScheduleWrapper:
    """ScheduleWrapper.

    This class is used to override the schedule logic from the profiler \
    and perform recording for `optimizer_zero_grad`, `model_forward`、 \
    `model_backward`、`optimizer_step`.
    """

    def __init__(self, schedule: Callable) -> None:
        self._schedule = schedule
        self.reset()

    def setup(self, start_action_name: str) -> None:
        self._start_action_name = start_action_name

    def pre_step(self, current_action: str) -> None:
        self._current_action = current_action

    def reset(self):
        self._num_optimizer_zero_grad = 0
        self._num_model_forward = 0
        self._num_model_backward = 0
        self._num_optimizer_step = 0
        self._optimizer_zero_grad_reached_end = False
        self._model_forward_reached_end = False
        self._model_backward_reached_end = False
        self._optimizer_step_reached_end = False
        # used to stop profiler when
        # `ProfilerAction.RECORD_AND_SAVE` is reached.
        self._current_action: Optional[str] = None
        self._start_action_name: Optional[str] = None

    @property
    def num_step(self) -> int:
        if self._current_action == "optimizer_zero_grad":
            return self._num_optimizer_zero_grad
        elif self._current_action == "model_forward":
            return self._num_model_forward
        elif self._current_action == "model_backward":
            return self._num_model_backward
        elif self._current_action == "optimizer_step":
            return self._num_optimizer_step
        return 0

    def _step(self) -> None:
        if self._current_action == "optimizer_zero_grad":
            self._num_optimizer_zero_grad += 1
        elif self._current_action == "model_forward":
            self._num_model_forward += 1
        elif self._current_action == "model_backward":
            self._num_model_backward += 1
        elif self._current_action == "optimizer_step":
            self._num_optimizer_step += 1

    @property
    def has_finished(self) -> bool:
        if self._current_action == "optimizer_zero_grad":
            return self._optimizer_zero_grad_reached_end
        elif self._current_action == "model_forward":
            return self._model_forward_reached_end
        elif self._current_action == "model_backward":
            return self._model_backward_reached_end
        elif self._current_action == "optimizer_step":
            return self._optimizer_step_reached_end
        return False

    def __call__(self, num_step: int) -> "ProfilerAction":
        # ignore the provided input. Keep internal state instead.
        if self.has_finished:
            return ProfilerAction.NONE
        self._step()
        action = self._schedule(max(self.num_step, 0))
        if action == ProfilerAction.RECORD_AND_SAVE:
            if self._current_action == "optimizer_zero_grad":
                self._optimizer_zero_grad_reached_end = True
            elif self._current_action == "model_forward":
                self._model_forward_reached_end = True
            elif self._current_action == "model_backward":
                self._model_backward_reached_end = True
            elif self._current_action == "optimizer_step":
                self._optimizer_step_reached_end = True
        return action


@OBJECT_REGISTRY.register
class PyTorchProfiler(BaseProfiler):
    """This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of.

    different operators inside your model - both on the CPU and GPU

    Args:
        dirpath: Directory path for the ``filename``.
        filename: If present, filename where the profiler results will be
            saved instead of printing to stdout. The ``.txt`` extension will
            be used automatically.

        group_by_input_shapes: Include operator input shapes and group calls by shape.

        emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
            Run::

                nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

            To visualize, you can either use::

                nvvp trace_name.prof
                torch.autograd.profiler.load_nvprof(path)

        export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
            It will generate a ``.json`` file which can be read by Chrome.

        row_limit: Limit the number of rows in a table, ``-1`` is a special value that
            removes the limit completely.

        sort_by_key: Attribute used to sort entries. By default
            they are printed in the same order as they were registered.
            Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
            ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
            ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.

        record_functions: Set of profiled functions which will create a context manager on.
            Any other will be pass through.

        record_module_names: Whether to add module names while recording autograd operation.

        profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

        Raises:
            Exception:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
                If arg ``schedule`` is not a ``Callable``.
                If arg ``schedule`` does not return a ``torch.profiler.ProfilerAction``.
    """  # noqa

    RECORD_FUNCTIONS = {
        "model_forward",
    }
    STEP_FUNCTIONS = {
        "model_forward",
    }
    AVAILABLE_SORT_KEYS = {
        "cpu_time",
        "cuda_time",
        "cpu_time_total",
        "cuda_time_total",
        "cpu_memory_usage",
        "cuda_memory_usage",
        "self_cpu_memory_usage",
        "self_cuda_memory_usage",
        "count",
    }

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_functions: Set[str] = None,
        record_module_names: bool = True,
        **profiler_kwargs: Any,
    ) -> None:

        super().__init__(dirpath=dirpath, filename=filename)

        self._group_by_input_shapes = (
            group_by_input_shapes
            and profiler_kwargs.get("record_shapes", False)
        )
        self._emit_nvtx = emit_nvtx
        self._export_to_chrome = export_to_chrome
        self._row_limit = row_limit
        self._sort_by_key = (
            sort_by_key
            or f"{'cuda' if profiler_kwargs.get('use_cuda', False) else 'cpu'}_time_total"  # noqa
        )
        self._user_record_functions = record_functions or set()
        self._record_functions = (
            self._user_record_functions | self.RECORD_FUNCTIONS
        )
        self._record_module_names = record_module_names
        self._profiler_kwargs = profiler_kwargs

        self.profiler: Optional[_PROFILER] = None
        self.function_events: Optional["EventList"] = None
        self._register = None
        self._parent_profiler: Optional[_PROFILER] = None
        self._recording_map: Dict[str, record_function] = {}
        self._start_action_name: Optional[str] = None
        self._schedule: Optional[ScheduleWrapper] = None

        self._init_kineto(profiler_kwargs)

        if self._sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise KeyError(
                f"Found sort_by_key: {self._sort_by_key}. "
                f"Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

    def _init_kineto(self, profiler_kwargs: Any) -> None:
        has_schedule = "schedule" in profiler_kwargs
        self._has_on_trace_ready = "on_trace_ready" in profiler_kwargs

        schedule = profiler_kwargs.get("schedule", None)
        if schedule is not None:
            if not isinstance(schedule, Callable):
                raise TypeError(
                    f"Schedule should be a callable. Found: {schedule}"
                )
            action = schedule(0)
            if not isinstance(action, ProfilerAction):
                raise TypeError(
                    f"Schedule should return "
                    f"a `torch.profiler.ProfilerAction`. Found: {action}"
                )
        self._default_schedule()
        schedule = schedule if has_schedule else self._default_schedule()
        self._schedule = (
            ScheduleWrapper(schedule) if schedule is not None else schedule
        )
        self._profiler_kwargs["schedule"] = self._schedule

        activities = profiler_kwargs.get("activities", None)
        self._profiler_kwargs["activities"] = (
            activities or self._default_activities()
        )
        self._export_to_flame_graph = profiler_kwargs.get(
            "export_to_flame_graph", False
        )
        self._metric = profiler_kwargs.get("metric", "self_cpu_time_total")
        with_stack = (
            profiler_kwargs.get("with_stack", False)
            or self._export_to_flame_graph
        )
        self._profiler_kwargs["with_stack"] = with_stack

    def _should_override_schedule(self) -> bool:
        return False

    @staticmethod
    @lru_cache(1)
    def _default_schedule() -> Optional[callable]:
        # Those schedule defaults allow the profiling
        # overhead to be negligible over training time.
        return torch.profiler.schedule(wait=1, warmup=1, active=3)

    def _default_activities(self) -> List["ProfilerActivity"]:
        activities = []
        if self._profiler_kwargs.get("use_cpu", True):
            activities.append(ProfilerActivity.CPU)
        if self._profiler_kwargs.get("use_cuda", torch.cuda.is_available()):
            activities.append(ProfilerActivity.CUDA)
        return activities

    def start(self, action_name: str) -> None:
        if self.profiler is None:
            # close profiler if it is already opened.
            # might happen if 2 profilers
            # are created and the first one did not call `describe`
            try:
                torch.autograd._disable_profiler()
            except (AttributeError, RuntimeError):
                pass

            if self._schedule is not None:
                self._schedule.setup(action_name)

            self._create_profilers()

            profiler = self.profiler.__enter__()
            if profiler is not None:
                self.profiler = profiler

            if self._parent_profiler is not None:
                self._parent_profiler.__enter__()

            if self._register is not None:
                self._register.__enter__()

        if (
            self.profiler is not None
            and (action_name in self._record_functions)
            and action_name not in self._recording_map
        ):

            recording = record_function(action_name)
            recording.__enter__()
            self._recording_map[action_name] = recording

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

        if self.profiler is not None and (action_name in self.STEP_FUNCTIONS):

            # the default schedule requires a minimum of 5
            # steps to properly work: `wait=1, warmup=1, active=3`.
            # otherwise, this will raise a `segmentation fault`.
            if self._should_override_schedule():
                warnings.warn(
                    "The PyTorch Profiler default schedule will be "
                    "overridden as there is not enough "
                    "steps to properly record traces."
                )
                self._schedule = None
                self.profiler.schedule = (
                    torch.profiler.profiler._default_schedule_fn
                )

            if self._schedule is not None:
                self._schedule.pre_step(action_name)

            def on_trace_ready(profiler):
                if self.dirpath is not None:
                    if self._export_to_chrome:
                        handler = tensorboard_trace_handler(
                            self.dirpath, self._prepare_filename(extension="")
                        )
                        handler(profiler)

                    if self._export_to_flame_graph:
                        path = os.path.join(
                            self.dirpath,
                            self._prepare_filename(extension=".stack"),
                        )
                        profiler.export_stacks(path, metric=self._metric)
                else:
                    rank_zero_warn(
                        "The PyTorchProfiler failed to export "
                        "trace as `dirpath` is None"
                    )

            if not self._has_on_trace_ready:
                self.profiler.on_trace_ready = on_trace_ready

            if self._schedule is not None:
                self.profiler.step_num = self._schedule.num_step
            self.profiler.step()

    def summary(self) -> str:
        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""

        self._delete_profilers()

        if not self.function_events:
            return ""
        data = self.function_events.key_averages(
            group_by_input_shapes=self._group_by_input_shapes
        )
        table = data.table(
            sort_by=self._sort_by_key, row_limit=self._row_limit
        )

        recorded_stats = {"records": table}
        return self._stats_to_str(recorded_stats)

    def _create_profilers(self) -> None:
        if self._emit_nvtx:
            self._parent_profiler = self._create_profiler(
                torch.cuda.profiler.profile
            )
            self.profiler = self._create_profiler(
                torch.autograd.profiler.emit_nvtx
            )
        else:
            self._parent_profiler = None
            self.profiler = self._create_profiler(torch.profiler.profile)

    def _create_profiler(self, profiler: Type[_PROFILER]) -> _PROFILER:
        init_parameters = inspect.signature(profiler.__init__).parameters
        kwargs = {
            k: v
            for k, v in self._profiler_kwargs.items()
            if k in init_parameters
        }
        return profiler(**kwargs)

    def _cache_functions_events(self) -> None:
        if self._emit_nvtx:
            return
        self.function_events = self.profiler.events()

    def _delete_profilers(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self._cache_functions_events()
            self.profiler = None

        if self._schedule is not None:
            self._schedule.reset()

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None

    def teardown(self, stage: Optional[str] = None) -> None:
        self._delete_profilers()

        for k in self._recording_map:
            self.stop(k)
        self._recording_map = {}

        super().teardown(stage=stage)
