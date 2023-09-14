# Copyright (c) Changan Auto. All rights reserved.
"""This file is modified from pytorch-lightning, add task sampler."""
import logging
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from torch.utils.data import Dataset

from cap.core.task_sampler import TaskSampler
from cap.registry import OBJECT_REGISTRY, build_from_registry
from cap.utils.apply_func import apply_to_collection
from cap.utils.distributed import all_gather_object, get_dist_info
from cap.utils.logger import MSGColor, format_msg, rank_zero_info
from .utils import get_len, has_len

if TYPE_CHECKING:
    from cap.engine.trainer import Trainer

__all__ = ["MultitaskLoader"]

logger = logging.getLogger(__name__)


class CycleIterator(object):
    """Iterator for restarting a dataloader if it runs out of samples."""

    def __init__(
        self,
        loader: Any,
        length: Optional[int] = None,
        fill_none: bool = False,
    ):  # noqa: D205,D400
        """

        Args:
            loader: the loader to restart for cyclic (and optionally infinite)
                sampling.
            length: the number of batches to sample (with restarted loaders
                if necessary) before raising StopIteration.
                if None: infinite
            fill_none: fill null data batch without repeat data in cycle.

        """
        if length is None:
            length = float("inf")

        self.length = length
        self.loader = loader
        self._loader_iter = None
        self.counter = 0
        if fill_none:
            self.none_loader = NoneIterator()
        if hasattr(loader, "batch_size"):
            self.batch_size = loader.batch_size

    def __iter__(self) -> Any:
        """
        Create the internal iterator and returns self.

        Returns:
            CycleIterator: self

        """
        self.counter = 0
        self._loader_iter = iter(self.loader)
        return self

    def __next__(self) -> Any:  # noqa: D205,D400
        """
        Fetch the next batch from internal dataloader and restarts
        it if necessary.

        Returns:
            Any: the resulting batch

        Raises:
            StopIteration: if more then :attr:`length` batches have been
                returned

        """
        # Note: if self.length is `inf`, then the iterator will never stop
        if self.counter >= self.__len__():
            # for torch.utils.data.DataLoader, we manual shutdown workers
            # to prevent call too many subprocess.
            if hasattr(self._loader_iter, "_shutdown_workers"):
                self._loader_iter._shutdown_workers()
            raise StopIteration

        try:
            return next(self._loader_iter)

        except StopIteration:
            if hasattr(self, "none_loader"):
                self._loader_iter = iter(self.none_loader)
            else:
                self._loader_iter = iter(self.loader)
            return next(self._loader_iter)

        finally:
            self.counter += 1

    def __len__(self) -> Union[int, float]:
        return self.length

    @property
    def sampler(self):
        return self.loader.sampler


class NoneIterator(object):
    """Data iterator returning None, used in CycleIterator."""

    def __iter__(self):
        return self

    def __next__(self):
        return None

    @property
    def sampler(self):
        return None


class MultitaskDataset(object):
    """Combine multiple datasets and compute their statistics."""

    COMPUTE_FUNCS = {"min_size": min, "max_size": max, "validation": max}

    def __init__(
        self,
        datasets: Union[Sequence, Mapping],
        mode: str = "max_size",
    ):  # noqa: D205,D400
        """
        Args:
            datasets: a sequence/mapping datasets. Can be a collections of
                torch.utils.Dataset,
                Iterable or even None.
            mode: whether to use the minimum number of batches in all samples
                or the maximum.
                number of batches in all samples.

        """
        self.datasets = datasets
        if mode not in self.COMPUTE_FUNCS.keys():
            raise ValueError(
                f'You have selected unsupported mode "{mode}",'
                f" please select one the: {list(self.COMPUTE_FUNCS.keys())}."
            )
        self.mode = mode

    @property
    def max_len(self) -> Union[int, float]:
        return self._calc_num_data(self.datasets, "max_size")

    @property
    def min_len(self) -> Union[int, float]:
        return self._calc_num_data(self.datasets, "min_size")

    @staticmethod
    def _calc_num_data(
        datasets: Union[Sequence, Mapping], mode: str
    ) -> Union[int, float]:
        """
        Compute the length of `MultitaskDataset` according to the `mode`.

        Args:
            datasets: a sequence/mapping datasets. Can be a collections of
                torch.utils.data.Dataset, Iterable or even None.
            mode: Determine `MultitaskDataset`'s length is the maximum or
                minimum of the datasets.

        Returns:
            length: the length of `MultitaskDataset`

        """
        if mode not in MultitaskDataset.COMPUTE_FUNCS.keys():
            raise ValueError(f"Invalid Mode: {mode}")

        # extract the lengths
        all_lengths = apply_to_collection(
            datasets,
            (Dataset, Iterable, type(None)),
            get_len,
            wrong_dtype=(Sequence, Mapping),
        )

        compute_func = MultitaskDataset.COMPUTE_FUNCS[mode]

        if isinstance(all_lengths, (int, float)):
            length = all_lengths
        else:
            length = _nested_calc_num_data(all_lengths, compute_func)

        return length

    def __len__(self) -> int:
        """Return the minimum length of the datasets."""
        return self._calc_num_data(self.datasets, self.mode)


@OBJECT_REGISTRY.register
class MultitaskLoader(object):  # noqa: D205,D400
    """
    Combines different dataloaders and allows sampling in parallel,
    supporting task sampler.

    Mainly used in multitask training.

    Supported modes are 'min_size', which raises StopIteration after the
    shortest loader (the one with the lowest number of batches) is done,
    and 'max_size' which raises StopIteration after the longest loader
    (the one with most batches) is done, while cycling through the shorter
    loaders. 'validation' raise StopIteration when all loaders finish,
    without cycle iterator.

    Examples:
        >>> loaders = {'a': torch.utils.data.DataLoader(range(6), batch_size=4),
        ...            'b': torch.utils.data.DataLoader(range(15), batch_size=5)}
        >>> multitask_loader = MultitaskLoader(loaders, mode='max_size')
        >>> for item in multitask_loader:
        ...     print(item)
        OrderedDict([('a', tensor([0, 1, 2, 3])), ('b', tensor([0, 1, 2, 3, 4]))])
        OrderedDict([('a', tensor([4, 5])), ('b', tensor([5, 6, 7, 8, 9]))])
        OrderedDict([('a', tensor([0, 1, 2, 3])), ('b', tensor([10, 11, 12, 13, 14]))])
        >>> multitask_loader = MultitaskLoader(loaders, mode='min_size')
        >>> for item in multitask_loader:
        ...     print(item)
        OrderedDict([('a', tensor([0, 1, 2, 3])), ('b', tensor([0, 1, 2, 3, 4]))])
        OrderedDict([('a', tensor([4, 5])), ('b', tensor([5, 6, 7, 8, 9]))])

    """  # noqa

    SUPPORTED_MODES = ("min_size", "max_size", "validation")

    def __init__(
        self,
        loaders: Dict,
        task_sampler: Optional[TaskSampler] = None,
        mode: str = "max_size",
        return_task: bool = False,
        custom_length: Optional[int] = None,
    ):
        """Initialize method.

        Args:
            loaders: the loaders to sample from. Should be a dict.
            task_sampler: task sampler. Used to sample task in each iter.
            mode: the mode. Supported are 'min_size' which stops if the\
                shortest loader is exhausted and 'max_size' which stops
                if the longest loader is exhausted and cycles through the
                smaller ones, and 'validation' which stops when all loaders
                finish without cycle iterator.
            return_task: return output as "batch, task" pairs, used by graph
                model input. Default return dict with task as key.
            custom_length : Custom length of `MultitaskLoaderIterator`,
                i.e. the batch num. If greater than real length, no-op.

        """

        self.task_sampler = task_sampler
        self.loaders = {}
        if task_sampler is not None:
            for t in task_sampler.tasks:
                self.loaders[t] = loaders[t]
        else:
            self.loaders = loaders

        datasets = apply_to_collection(
            self.loaders,
            Iterable,
            getattr,
            "dataset",
            None,
            wrong_dtype=(Sequence, Mapping),
        )
        # could be multiple datasets, but use self.dataset to follow the name
        # convention in DataLoader.
        self.dataset = MultitaskDataset(datasets, mode)

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Invalid Mode: {mode}")
        self.mode = mode
        if task_sampler is not None and task_sampler.is_parallel():
            self._wrap_sync_loaders_cycle()
        else:
            if self.mode in ("validation", "max_size"):
                self._wrap_loaders_max_size_cycle()

        self.return_task = return_task
        if custom_length is not None:
            assert custom_length > 0, custom_length
            self.custom_length = custom_length
        else:
            self.custom_length = None

        self._trainer = None
        self._batch_size = None

    def _wrap_sync_loaders_cycle(self):  # noqa: D400, D401
        """
        Wraps all loaders to make sure they are cycled with the same length

        Returns:
            the wrapped loaders

        """
        mode = self.mode
        all_lengths = apply_to_collection(
            self.loaders, Iterable, get_len, wrong_dtype=(Sequence, Mapping)
        )
        assert isinstance(all_lengths, Mapping)

        # gather all length from other processes
        # here all_lenghts is {'task': length}
        _, world_size = get_dist_info()
        global_data = [None for _ in range(world_size)]
        all_gather_object(global_data, all_lengths)

        # calc need length for all tasks
        all_data: Dict = {}
        for i, d in enumerate(global_data):
            assert len(d) == 1, f"rank {i} has more than 1 dataloader"
            for k in d:
                # k is task name
                if k not in all_data:
                    all_data[k] = d[k]
                else:
                    # dataloader within same task should has same length
                    assert (
                        all_data[k] == d[k]
                    ), f"rank {i} has diff dataset len {all_data[k]}:{d[k]}"
        lengths = []
        for k in all_data:
            lengths.append(all_data[k])

        if mode == "min_size":
            length = min(lengths)
        else:
            length = max(lengths)

        _old_info = {k: get_len(v) for k, v in self.loaders.items()}

        fill_none = self.mode == "validation"
        if isinstance(self.loaders, Mapping):
            self.loaders = type(self.loaders)(
                {
                    k: CycleIterator(v, length=length, fill_none=fill_none)
                    if has_len(v)
                    else v
                    for k, v in self.loaders.items()
                }
            )
        else:
            raise ValueError(
                f"Invalid Datatype for loaders: "
                f"{type(self.loaders).__name__}"
            )

        for k, v in self.loaders.items():
            logger.info(
                f"SyncLoaderLength {k}: {_old_info[k]} => {get_len(v)}"
            )

    @property
    def sampler(self) -> Union[Iterable, Sequence, Mapping]:
        """Return a collections of samplers extracting from loaders."""
        return apply_to_collection(
            self.loaders,
            Iterable,
            getattr,
            "sampler",
            None,
            wrong_dtype=(Sequence, Mapping),
        )

    @property
    def batch_size(self) -> int:  # noqa: D205,D400
        """
        Return batch size sum of all sub loaders, multipled by task sampling
        factor.
        """
        if not self._batch_size:
            if isinstance(self.loaders, dict):
                if self.task_sampler is not None:
                    sampling_factors = {
                        k: self.task_sampler._config[k]["sampling_factor"]
                        for k in self.loaders
                    }
                else:
                    # If no task_sampler found, use uniform sample.
                    sampling_factors = {k: 1 for k in self.loaders}
                self._batch_size = sum(
                    [
                        self.loaders[k].batch_size * sampling_factors[k]
                        for k in self.loaders
                    ]
                )
            else:
                raise NotImplementedError

        return self._batch_size

    def _wrap_loaders_max_size_cycle(self) -> Any:  # noqa: D205,D400
        """
        Wrap all loaders to make sure they are cycled until the longest
        loader is exhausted.

        Returns:
            the wrapped loaders
        """
        all_lengths = apply_to_collection(
            self.loaders, Iterable, get_len, wrong_dtype=(Sequence, Mapping)
        )

        if isinstance(all_lengths, (int, float)):
            length = all_lengths

        elif isinstance(all_lengths, Sequence):
            length = max(all_lengths)

        elif isinstance(all_lengths, Mapping):
            # apply sampling_factor of task on all_lengths
            if self.task_sampler is not None:
                valid_tasks = self.get_start_epoch_task()
                if len(valid_tasks) != len(self.task_sampler.tasks):
                    for unvalid_task in self.task_sampler.tasks:
                        if unvalid_task in valid_tasks:
                            continue
                        logger.warning(
                            f"Computing max size of MultitaskLoader"
                            f"ignores task: {unvalid_task}"
                        )
                all_lengths = {
                    k: all_lengths[k] * valid_tasks[k]["sampling_factor"]
                    for k in valid_tasks
                }
            length = max(all_lengths.values())

        _old_info = {k: get_len(v) for k, v in self.loaders.items()}

        fill_none = self.mode == "validation"
        if isinstance(self.loaders, Mapping):
            self.loaders = type(self.loaders)(
                {
                    k: CycleIterator(v, length=length, fill_none=fill_none)
                    if has_len(v)
                    else v
                    for k, v in self.loaders.items()
                }
            )

        elif isinstance(self.loaders, Sequence):
            self.loaders = type(self.loaders)(
                [
                    CycleIterator(v, length=length, fill_none=fill_none)
                    for v in self.loaders
                ]
            )

        # dataloaders are iterable but not sequence
        elif isinstance(self.loaders, Iterable):
            # only one dataloader, just keep it the same.
            pass
        else:
            raise ValueError(
                f"Invalid Datatype for loaders: "
                f"{type(self.loaders).__name__}"
            )

        for k, v in self.loaders.items():
            rank_zero_info(
                f"{k}: {_old_info[k]} => "
                f"{_old_info[k] if fill_none else get_len(v) }"
            )  # noqa

    def __iter__(self) -> Any:  # noqa: D205,D400
        """
        Create and return an iterator, `MultitaskLoaderIterator`,
        for the combined loader.
        """
        return MultitaskLoaderIterator(
            self.loaders,
            self.task_sampler,
            self._trainer,
            self.return_task,
            self.custom_length,
        )

    @staticmethod
    def _calc_num_batches(loaders: Any) -> Union[int, float]:
        """
        Compute the length (aka the number of batches) of `MultitaskLoader`.

        Args:
            loaders: a collections of loaders.

        Returns:
            length: the minimum length of loaders

        """
        all_lengths = apply_to_collection(
            loaders, Iterable, get_len, wrong_dtype=(Sequence, Mapping)
        )

        if isinstance(all_lengths, (int, float)):
            return all_lengths

        else:
            return _nested_calc_num_data(all_lengths, min)

    def __len__(self) -> int:
        return self._calc_num_batches(self.loaders)

    def set_trainer(self, trainer):
        # TODO(1.0): use StopIteration to get step/epoch information #
        self._trainer = trainer

    def get_start_epoch_task(self) -> Dict:
        """Get start_epoch=0 task from task_sampler."""
        if self.task_sampler is None:
            return {}
        start_epoch_task = {}
        for task in self.task_sampler.tasks:
            task_config = self.task_sampler.config[task]
            if "start_step" in task_config:
                raise NotImplementedError
            if task_config.get("start_epoch", 0) != 0:
                continue
            start_epoch_task[task] = task_config
        return start_epoch_task


@OBJECT_REGISTRY.register
class MultitaskInfLoader(object):
    def __init__(
        self,
        loaders: Dict,
        task_sampler: Optional[TaskSampler] = None,
        return_task: bool = False,
    ):
        """Initialize method.

        Args:
            loaders: the loaders to sample from. Should be a dict.
            task_sampler: task sampler. Used to sample task in each iter.
            mode: the mode. Supported are 'min_size' which stops if the\
                shortest loader is exhausted and 'max_size' which stops
                if the longest loader is exhausted and cycles through the
                smaller ones, and 'validation' which stops when all loaders
                finish without cycle iterator.
            return_task: return output as "batch, task" pairs, used by graph
                model input. Default return dict with task as key.
            custom_length : Custom length of `MultitaskLoaderIterator`,
                i.e. the batch num. If greater than real length, no-op.

        """

        if task_sampler is not None:
            for task in task_sampler.tasks:
                assert task in loaders, f"{task} not found in dataloader"

        self.loaders = {
            k: build_from_registry(v)
            for k, v in loaders.items()
            if k in task_sampler.tasks
        }
        self.task_sampler = task_sampler
        self._loader_iters = None

        self.return_task = return_task

        self._trainer = None
        self._batch_size = None

    @property
    def batch_size(self) -> int:
        if not self._batch_size:
            if isinstance(self.loaders, dict):
                if self.task_sampler is not None:
                    sampling_factors = {
                        k: self.task_sampler.config[k]["sampling_factor"]
                        for k in self.loaders
                    }
                else:
                    # If no task_sampler found, use uniform sample.
                    sampling_factors = {k: 1 for k in self.loaders}
                self._batch_size = sum(
                    [
                        self.loaders[k].batch_size * sampling_factors[k]
                        for k in self.loaders
                    ]
                )
            else:
                raise NotImplementedError

        return self._batch_size

    def __next__(self):

        if self.task_sampler is not None:
            task = self.task_sampler.sample_task()
            if isinstance(task, str):
                task = [task]
            next_batch = OrderedDict((t, self._next_task(t)) for t in task)

        batch_task_pairs = []
        assert isinstance(next_batch, dict)
        for name, batch in next_batch.items(): 
            _name = name if self.return_task else None
            task_batch = {}
            task_batch.update(batch)
            task_batch[name] = {name: batch}
            """Assign task name into data"""
            batch_task_pairs.append((task_batch, _name))
        next_batch = tuple(batch_task_pairs)
        return next_batch

    def _next_task(self, task):
        try:
            batch = next(self._loader_iters[task])
        except StopIteration:
            logger.warning(f"{task} loader reset")
            self._loader_iters[task] = iter(self.loaders[task])
            batch = next(self._loader_iters[task])
        return batch

    def __iter__(self) -> Any:  # noqa: D205,D400
        if self._loader_iters is None:
            self._loader_iters = apply_to_collection(
                self.loaders, Iterable, iter, wrong_dtype=(Sequence, Mapping)
            )
        return self


class MultitaskLoaderIterator(object):  # noqa: D205,D400
    """
    Custom Iterator returning data from multple loaders,
    and allows sampling in parallel.
    """

    def __init__(
        self,
        loaders: Any,
        task_sampler: Optional[TaskSampler] = None,
        trainer: Optional["Trainer"] = None,
        return_task: bool = False,
        custom_length: Optional[int] = None,
    ):  # noqa: D205,D400
        """

        Args:
            loaders: the loaders to sample from. Can be all kind of collection
            task_sampler: task sampler. Used to sample task in each iter.
            trainer: trainer instance.
            return_task: return output as "batch, task" pairs, used by graph
                model input.
            custom_length : Custom length of `MultitaskLoaderIterator`,
                i.e. the batch num. If greater than real length, no-op.

        """
        self.loaders = loaders
        self.task_sampler = task_sampler
        self.trainer = trainer
        self.return_task = return_task
        self.custom_length = custom_length
        self._loader_iters = None
        self._batch_cnt = None

    @property
    def loader_iters(self) -> Any:
        """Get the `_loader_iters` and create one if it is None."""
        if self._loader_iters is None:
            self._batch_cnt = 0
            self._loader_iters = self.create_loader_iters(self.loaders)

        return self._loader_iters

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        """
        Fetch the next batch from multiple data loaders with task scheduler.

        Returns:
            a collections of batch data

        """
        loader_iters = self.loader_iters

        if (
            self.custom_length is not None
            and self._batch_cnt >= self.custom_length
        ):
            logger.info(
                format_msg(
                    "manually decrease length of `MultitaskLoaderIterator` to"
                    " %d and trigger StopIteration" % self.custom_length,
                    MSGColor.GREEN,
                )
            )
            raise StopIteration
        if self.task_sampler is not None:
            # TODO(0.2): trainer has no `iter` attr anymore, get
            #  epoch_id by callback or do loader.set_epoch in Trainer.
            task = self.task_sampler.sample_task()
            if isinstance(task, str):
                task = [task]
            loader_iters = OrderedDict((t, loader_iters[t]) for t in task)
        next_batch = self.request_next_batch(loader_iters)
        # filter None batch
        if isinstance(next_batch, dict):
            next_batch = OrderedDict(
                (k, v) for k, v in next_batch.items() if v is not None
            )
        if self.return_task:
            batch_task_pairs = []
            assert isinstance(next_batch, dict)
            for _name, batch in next_batch.items():
                batch_task_pairs.append((batch, _name))
            next_batch = tuple(batch_task_pairs)
        self._batch_cnt += 1
        return next_batch

    @staticmethod
    def request_next_batch(
        loader_iters: Union[Iterator, Sequence, Mapping]
    ) -> Any:
        """
        Return the batch of data from multiple iterators.

        Args:
            loader_iters: a collections of iterators

        Returns
            Any: a collections of batch data

        """
        return apply_to_collection(loader_iters, Iterator, next)

    @staticmethod
    def create_loader_iters(
        loaders: Union[Any, Iterator, Sequence, Mapping]
    ) -> Union[Any, Iterator, Sequence, Mapping]:
        """
        Create and return a collection of iterators from loaders.

        Args:
            loaders: a collections of loaders

        Returns
            a collections of iterators

        """
        # dataloaders are Iterable but not Sequences. Need this to specifically
        # exclude sequences.
        return apply_to_collection(
            loaders, Iterable, iter, wrong_dtype=(Sequence, Mapping)
        )


def _nested_calc_num_data(
    data: Union[Mapping, Sequence], compute_func: Callable
):

    if isinstance(data, int):
        return data

    if isinstance(data, Mapping):
        data = list(data.values())

    if not isinstance(data, Sequence):
        raise TypeError(
            f"Expected data to be int, Sequence or Mapping,"
            f"but got {type(data).__name__}"
        )

    new_data = []

    for x in data:
        if isinstance(x, (Mapping, Sequence)):
            new_data.append(_nested_calc_num_data(x, compute_func))
        else:
            new_data.append(x)

    return compute_func(new_data)
