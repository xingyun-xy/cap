# Copyright (c) Changan Auto. All rights reserved.
import copy
import itertools
import logging
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Union

from cap.utils.distributed import (
    create_process_group,
    dist_initialized,
    get_dist_info,
    set_local_process_group,
)
from cap.utils.logger import MSGColor, format_msg, rank_zero_info

__all__ = ["TaskSampler", "update_from_white_list"]

logger = logging.getLogger(__name__)


class Task:
    # task sampling factor
    sampling_factor: int = 1

    # start training step, default is 0, meaning training from beginning.
    start_step: int = 0

    # end training step, default is -1, meaning training to the end.
    end_step: int = -1

    # start training epoch, default is 0, meaning training from beginning.
    start_epoch: int = 0

    # end training epoch, default is -1, meaning training to the end.
    end_epoch: int = -1

    # train task on which gpu group, used in task parallel.
    # 1 gpu group contains 1 or more gpus. Task in same gpu group trains on
    # same gpus.
    gpu_group: str = "0"

    # task on which rank. Don't set rank directly, it's automated.
    rank: List[int] = None

    # task name, auto set by build_task.
    name: str = ""

    def __repr__(self):
        msg = f"name: \n{self.name}\n"
        msg += f"sampling_factor: \t{self.sampling_factor}\n"
        msg += f"start_step: \t{self.start_step}\n"
        msg += f"end_step: \t{self.end_step}\n"
        msg += f"start_epoch: \t{self.start_epoch}\n"
        msg += f"end_epoch: \t{self.end_epoch}\n"
        msg += f"gpu_group: \t{self.gpu_group}\n"
        msg += f"rank: \t{self.rank}\n"
        return msg


def build_task(name, config: Union[int, Dict]):
    """Build task instance from config."""
    if isinstance(config, int):
        config = {"sampling_factor": config}
    else:
        assert isinstance(
            config, dict
        ), "Task config only support int or dict."
    task = Task()
    task.name = name
    for k in config:
        setattr(task, k, config[k])
        if k == "end_epoch" or k == "end_step":
            logger.warning(
                f"{k} with max dataloader may cause bug, " f"take careful"
            )
    # check valid
    assert task.end_epoch < 0 or task.end_epoch > task.start_epoch
    assert task.end_step < 0 or task.end_step > task.start_step
    return task


def expand_task(
    method: str,
    task_factory: Dict[str, Task],
    unions: Optional[List] = None,
    config=None,
) -> List[List[str]]:
    r"""
    Expand task to list, with sampling_factor times.

    For different method, output unions behave differently.

    * sample_all:
        output unions: [['task1' * factor, 'task2' * factor, 'task3' * factor, ...]]

    * sample_one:
        output unions: [['task1'] * factor, ['task2'] * factor, ['task3'] * factor, ...]

    * sample_part:
        output unions: [['task1', 'task2'] * factor, ['task3'] * factor, ...]

    * sample_repeat:
        output unions: [['task1', 'task2'] * factor, ['task1', 'task2', 'task3'] * factor, ...]


    Args:
        method: Sample method

        config: Task sampler config used to extract chosen tasks for sample_repeat
            and sample_dynamic methods
    """  # noqa
    assert method in TaskSampler.SUPPORTED_SAMPLE_METHOD
    if method == "sample_all":
        unions = [list(task_factory)]
        for name, task in task_factory.items():
            if task.sampling_factor > 1:
                for _ in range(task.sampling_factor - 1):
                    idx = unions[0].index(name)
                    unions[0].insert(idx, name)
            elif task.sampling_factor == 1:
                continue
            elif task.sampling_factor == 0:
                unions[0].remove(name)
            else:
                raise ValueError("sampling_factor should be positive")
    elif method == "sample_repeat":
        unions = []
        for chosen_task in config["chosen_tasks"]:
            union = []
            chosen_task = set(chosen_task)
            for name, task in task_factory.items():
                if name in chosen_task:
                    for _ in range(task.sampling_factor):
                        union.append(name)
            unions.append(union)
    elif method == "sample_one":
        unions = [[name] for name in task_factory]
        for name, task in task_factory.items():
            if task.sampling_factor > 1:
                for _ in range(task.sampling_factor - 1):
                    idx = unions.index([name])
                    unions.insert(idx, [name])
            elif task.sampling_factor == 1:
                continue
            elif task.sampling_factor == 0:
                unions.remove([name])
            else:
                raise ValueError("sampling_factor should be positive")
    else:
        assert unions is not None, "unions is required when use sample_part"

        flatten_unions = list(itertools.chain.from_iterable(unions))
        if len(flatten_unions) != len(set(flatten_unions)):
            raise NotImplementedError("Duplicate task not implemented")

        extra_unions = []
        for union in unions:
            sampling_factors = [task_factory[t].sampling_factor for t in union]
            if len(set(sampling_factors)) == 1:
                continue
            logger.warning(
                f"Found inequal sampling_factor for one union "
                f"task: {union}, be sure what you are doing."
            )
            min_sampling_factor = min(sampling_factors)
            if min_sampling_factor > 1:
                for _ in range(min_sampling_factor - 1):
                    extra_unions.append(union)
            for task, factor in zip(union, sampling_factors):
                if factor == min_sampling_factor:
                    continue
                for _ in range(factor - min_sampling_factor):
                    # add extra task to unions
                    extra_unions.append([task])
        unions.extend(extra_unions)
    return unions


class TaskSampler:
    """
    Task sampler used in multitask training.

    Args:
        task_config: task sampler config dict. For each task, you can set
            `Task` attributes. See `Task` for more details.
        shuffle: shuffle task order in one cycle.
        method: sample method, support:
            * sample_all: sample all task at one time.
            * sample_one: sample one task at one time.
            * sample_part: sample part tasks at one time, based on `unions`.
            * sample_repeat:
                if end_steps in config is None:
                    repeatly sample tasks in chosen_tasks input.
                else:
                    sample tasks in chosen_tasks in order,
                    sample each group of tasks controlled by end_steps.
        unions: task unions list, only required in sample_part mode.
        gpu_weights: gpu group information used when set `gpu_group` in task.
            See example below.

    Example::

        cfg = OrderedDict(
            person=dict(sampling_factor=1, start_epoch=0, end_epoch=5),
            vehicle=dict(sampling_factor=2, start_epoch=0, end_epoch=5),
            lane=dict(sampling_factor=3, start_epoch=3, end_epoch=5),
            real3d=dict(sampling_factor=2, start_epoch=4, end_epoch=5),
        )
        scheduler = TaskSampler(cfg, shuffle=False, method='sample_one')

        # sample repeat
        cfg = OrderedDict(
            person=dict(sampling_factor=1, start_epoch=0, end_epoch=5),
            vehicle=dict(sampling_factor=2, start_epoch=0, end_epoch=5),
            lane=dict(sampling_factor=3, start_epoch=3, end_epoch=5),
            real3d=dict(sampling_factor=2, start_epoch=4, end_epoch=5),
            end_steps=[5,10,20],
            chosen_tasks=[['person'],['pserson','vehicle'],['vehicle']],
        )
        scheduler = TaskSampler(cfg, shuffle=False, method='sample_repeat')

        # task parallel
        cfg = OrderedDict(
            person=dict(sampling_factor=1, gpu_group="0"),
            vehicle=dict(sampling_factor=2, gpu_group="0"),
            lane=dict(sampling_factor=3, gpu_group="1"),
            real3d=dict(sampling_factor=2, gpu_group="1"),
        )
        gpu_weights = {"0": 2, "1": 2}
        scheduler = TaskSampler(cfg, gpu_weights=gpu_weights)

    """

    SUPPORTED_SAMPLE_METHOD = (
        "sample_one",
        "sample_all",
        "sample_part",
        "sample_repeat",
    )

    def __init__(
        self,
        task_config: Dict,
        shuffle: bool = False,
        method: str = "sample_one",
        unions: Optional[List] = None,
        gpu_weights: Optional[List] = None,
    ):

        self._config = self._format_config(task_config)
        self._shuffle = shuffle
        self._sample_method = method
        self._unions = unions
        self._gpu_weights = gpu_weights
        assert method in self.SUPPORTED_SAMPLE_METHOD
        self._init_helper()

        self._tasks = None

        # counter inside a cycle
        self._inner_cycle_idx = -1
        # step counter
        self._inner_step_cnt = 0
        self._curr_step = 0
        self._curr_stage = 0
        self._pointer = 0

    def _init_helper(self):
        """Help init members from self._config."""
        self._is_parallel = False
        self._process_group = None

        self._update_config_if_parallel()

        self._task_factory = OrderedDict()
        for name in self._config:
            if name not in ["end_steps", "chosen_tasks"]:
                self._task_factory[name] = build_task(name, self._config[name])
        self._expand_tasks = expand_task(
            self._sample_method, self._task_factory, self._unions, self._config
        )
        self._cycle_length = len(self._expand_tasks)

    @property
    def tasks(self) -> list:
        """Property tasks come from task_factory."""
        return list(self._task_factory.keys())

    @property
    def config(self) -> Dict:
        return self._config

    @config.setter
    def config(self, config):
        """Update config and related vars."""
        self._config = self._format_config(config)
        self._init_helper()

    def __len__(self):
        return self._cycle_length

    def __iter__(self):
        return self

    def __next__(self):
        task = self.sample_task()
        if self._inner_step_cnt == self._cycle_length:
            self._inner_step_cnt = 0
            raise StopIteration
        return task

    def _format_config(self, config):
        """Format config."""
        cfg = copy.deepcopy(config)
        for name, cfg_i in cfg.items():
            if isinstance(cfg_i, int):
                cfg[name] = {"sampling_factor": cfg_i}
        return cfg

    def is_task_valid(self, task_name, epoch=None, step=None):
        """Check if task in valid in current epoch."""
        assert epoch is not None or step is not None
        if isinstance(task_name, list):
            if len(task_name) > 1:
                raise NotImplementedError(
                    "step/epoch mode is not supported in sample_part mode"
                )
            task_name = task_name[0]
        task = self._task_factory[task_name]
        if epoch is not None:
            return epoch >= task.start_epoch and (
                task.end_epoch < 0 or epoch < task.end_epoch
            )
        if step is not None:
            return step >= task.start_step and (
                task.end_step < 0 or step < task.end_step
            )

    def get_valid_tasks(self, epoch=None, step=None):
        """Get valid task factory at current epoch or step."""
        if epoch is None and step is None:
            return self._task_factory
        if epoch is not None and step is not None:
            raise ValueError("Please don't use epoch ans step together.")
        valid_tasks = OrderedDict()
        for name, task in self._task_factory.items():
            if self.is_task_valid(name, epoch, step):
                valid_tasks[name] = task
        return valid_tasks

    def has_step_or_epoch(self) -> bool:
        """If task_config contains step or epoch config."""
        for _name, task in self._config.items():
            if (
                "start_step" in task
                or "end_step" in task
                or "start_epoch" in task
                or "end_epoch" in task
            ):
                return True
        return False

    def sample_task(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> List[str]:
        if self._sample_method == "sample_repeat":

            if self._config.get("end_steps", None) is None:
                if self._pointer >= len(self._expand_tasks):
                    self._pointer = 0
                task = self._expand_tasks[self._pointer]
                self._pointer += 1
            else:
                if self._curr_step >= self._config["end_steps"][-1]:
                    logger.warning(
                        format_msg(
                            "Try to sample more than step size", MSGColor.RED
                        )
                    )
                if self._curr_step >= self._config["end_steps"][
                    self._curr_stage
                ] and self._curr_stage < (len(self._config["end_steps"]) - 1):
                    self._curr_stage += 1
                task = self._expand_tasks[self._curr_stage]
                self._curr_step += 1
        else:
            """Sample a task based on inner counter."""
            if self.has_step_or_epoch() and step is not None:
                assert epoch is None, "step and epoch can't be used together."
                raise NotImplementedError

            if self.has_step_or_epoch() and epoch is not None:
                assert step is None, "step and epoch can't be used together."
                # when start as a new epoch, reset inner counter to -1.
                if not hasattr(self, "_last_epoch"):
                    self._last_epoch = -1
                if epoch != self._last_epoch:
                    self._inner_cycle_idx = -1

            self._inner_cycle_idx += 1
            self._inner_cycle_idx %= self._cycle_length

            # shuffle task_order if need
            if self._inner_cycle_idx == 0 and self._shuffle:
                random.shuffle(self._expand_tasks)

            # sample a task
            task = self._expand_tasks[self._inner_cycle_idx]

            # if epoch or step given, update task to valid one.
            if self.has_step_or_epoch() and (
                epoch is not None or step is not None
            ):
                # TODO(0.5): infinite loop #
                while not self.is_task_valid(task, epoch=epoch, step=step):
                    self._inner_cycle_idx += 1
                    self._inner_cycle_idx %= self._cycle_length
                    task = self._expand_tasks[self._inner_cycle_idx]
                self._last_epoch = epoch

            self._inner_step_cnt += 1
        return task

    def _update_config_if_parallel(self) -> None:
        """Update task in config if parallel, and assign rank in config."""  # noqa
        if self._gpu_weights:
            assert isinstance(
                self._gpu_weights, dict
            ), "gpu_weights should be dict"
            if not dist_initialized():
                self._is_parallel = False
                logger.warning(
                    "TaskSampler is set gpu_group, but dist is not "
                    "initialized, skip"
                )
                return
        else:
            self._is_parallel = False
            return

        # Convert gpu_group and gpu_weights to group_weight_dict
        gpu_groups = [cfg_i["gpu_group"] for _, cfg_i in self._config.items()]
        group_weight_dict = {
            group: self._gpu_weights[group] for group in gpu_groups
        }

        # Get actual gpu nums for each gpu group
        rank, world_size = get_dist_info()
        weight_sum = sum(group_weight_dict.values())
        gpu_nums = [
            int(world_size * w / weight_sum)
            for w in group_weight_dict.values()
        ]
        assert sum(gpu_nums) == world_size

        # Convert gpu_groups and gpu_nums to task_ranks
        group_rank_dict = {}
        base = 0
        current_group = None
        for group, gpu_num in zip(group_weight_dict.keys(), gpu_nums):
            group_rank_dict[group] = [base + i for i in range(gpu_num)]
            if base <= rank < (base + gpu_num):
                current_group = group
            base += gpu_num

        # Display task parallel info.
        rank_zero_info("TaskSample in parallel mode.")
        for task_name in self._config:
            gpu_group = self._config[task_name]["gpu_group"]
            rank = group_rank_dict[gpu_group]
            rank_zero_info(f"task_name: {','.join([str(i) for i in rank])}")

        # create sub process group for all groups
        group2pg = {}
        for group in group_rank_dict:
            ranks = group_rank_dict[group]
            pg = create_process_group(ranks)
            group2pg[group] = pg

        # Update self._config for current selected group
        config_in_rank = OrderedDict()
        for task_name in self._config:
            gpu_group = self._config[task_name]["gpu_group"]
            if gpu_group == current_group:
                config_in_rank[task_name] = self._config[task_name]
                config_in_rank[task_name]["rank"] = group_rank_dict[
                    gpu_group
                ]  # noqa
        self._config = config_in_rank
        self._is_parallel = True
        self._process_group = group2pg[current_group]
        set_local_process_group(self._process_group)

    def is_parallel(self):
        return self._is_parallel


# TODO(HDLT-299): move to TaskSampler #
def update_from_white_list(
    sampler: TaskSampler, white_list: List[str]
) -> None:
    """Ignore the useless task and update the sampler.

    Args:
        white_list(list): the useful task list

    Returns:
        None
    """
    logger.warning(
        format_msg("You are setting the task_sampler", MSGColor.RED)
    )
    sampler_config = copy.deepcopy(sampler.config)
    sampler_tasks = copy.deepcopy(sampler.tasks)
    for task in list(sampler.config.keys()):
        if task not in white_list:
            sampler_config.pop(task)
            sampler_tasks.remove(task)
        else:
            pass
    sampler.config = sampler_config
