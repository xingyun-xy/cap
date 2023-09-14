# Copyright (c) Changan Auto. All rights reserved.
from itertools import accumulate
from math import gcd
from typing import Dict, List

import numpy as np
import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY

__all__ = [
    "RepeatDataset",
    "ComposeDataset",
    "ResampleDataset",
    "ConcatDataset",
]


@OBJECT_REGISTRY.register
class RepeatDataset(data.Dataset):
    """
    A wrapper of repeated dataset.

    Using RepeatDataset can reduce the data loading time between epochs.

    Args:
        dataset (torch.utils.data.Dataset): The datasets for repeating.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        if hasattr(self.dataset, "flag"):
            self.flag = np.tile(self.dataset.flag, times)
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


@OBJECT_REGISTRY.register
class ComposeDataset(data.Dataset):
    """Dataset wrapper for multiple datasets with precise batch size.

    Args:
        datasets: config for each dataset.
        batchsize_list: batchsize for each task dataset.
    """

    def __init__(self, datasets: List[Dict], batchsize_list: List[int]):
        self.datasets = datasets
        self.batchsize_list = batchsize_list
        self.total_batchsize = sum(batchsize_list)

        self.len_list = [len(dataset) for dataset in self.datasets]
        self.max_len = max(self.len_list)
        self.total_len = sum(self.len_list)
        self.dataset_bounds = []
        flag = 0
        for bachsize in self.batchsize_list:
            self.dataset_bounds.append(flag + bachsize)
            flag += bachsize
        self.iter_time = 0

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        if self.iter_time >= self.total_batchsize:
            self.iter_time = 0
        for i, bound in enumerate(self.dataset_bounds):
            if self.iter_time < bound:
                self.iter_time += 1
                idx = idx % len(self.datasets[i])
                assert idx < len(self.datasets[i]), (
                    f"{idx} exceeds " f"{len(self.datasets[i])}"
                )
                return self.datasets[i][idx]

    def __repr__(self):
        return "ComposeDataset"

    def __str__(self):
        return str(self.datasets)


def list_gcd(list_of_int):
    assert len(list_of_int) > 0
    g = list_of_int[0]
    for i in range(1, len(list_of_int)):
        g = gcd(g, list_of_int[i])
    return g


@OBJECT_REGISTRY.register
class ComposeProbDataset(data.Dataset):
    """Dataset wrapper for multiple datasets with precise batch size.

    Args:
        datasets: config for each dataset.
        batchsize_list: batchsize for each task dataset.
    """

    def __init__(self, datasets: List[Dict], sample_weights: List[int]):
        self.datasets = datasets
        max_gcd = list_gcd(sample_weights)
        sample_weights = [x // max_gcd for x in sample_weights]
        self.sample_weights = sample_weights
        self.len_list = [len(dataset) for dataset in self.datasets]
        self.max_len = max(self.len_list)
        self.total_len = sum(self.len_list)

        self.dataset_bounds = list(accumulate(self.sample_weights))
        self.iter_time = 0
        self.totol_inters = self.dataset_bounds[-1]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        if self.iter_time >= self.totol_inters:
            self.iter_time = 0
        for i, bound in enumerate(self.dataset_bounds):
            if self.iter_time < bound:
                self.iter_time += 1
                idx = idx % len(self.datasets[i])
                assert idx < len(self.datasets[i]), (
                    f"{idx} exceeds " f"{len(self.datasets[i])}"
                )
                return self.datasets[i][idx]

    def __repr__(self):
        return "ComposeDataset"

    def __str__(self):
        return str(self.datasets)


@OBJECT_REGISTRY.register
class ComposeRandomDataset(data.Dataset):
    """Dataset wrapper for multiple datasets with precise batch size.

    Args:
        datasets: config for each dataset.
        batchsize_list: batchsize for each task dataset.
    """

    def __init__(self, datasets: List[Dict], sample_weights: List[int]):
        self.datasets = datasets
        self.total_weights = sum(sample_weights)
        self.sample_weights = [p / self.total_weights for p in sample_weights]

    def __len__(self):
        len_list = [len(dataset) for dataset in self.datasets]
        total_len = sum(len_list)
        return total_len

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        dataset_idx = np.random.choice(
            len(self.datasets), 1, p=self.sample_weights
        )[0]

        idx = idx % len(self.datasets[dataset_idx])
        assert idx < len(self.datasets[dataset_idx]), (
            f"{idx} exceeds " f"{len(self.datasets[dataset_idx])}"
        )
        return self.datasets[dataset_idx][idx]

    def __repr__(self):
        return "ComposeDataset"

    def __str__(self):
        return str(self.datasets)


@OBJECT_REGISTRY.register
class ResampleDataset(data.Dataset):
    """
    A wrapper of resample dataset.

    Using ResampleDataset can resample on original dataset
        with specific interval.

    Args:
        dataset (dict): The datasets for resampling.
        resample_interval (int): resample interval.
    """

    def __init__(self, dataset, resample_interval: int = 1):
        assert resample_interval >= 1 and isinstance(
            resample_interval, int
        ), "resample interval not valid!"
        self.dataset = dataset
        self.resample_interval = resample_interval
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx * self.resample_interval]

    def __len__(self):
        return self._ori_len // self.resample_interval


@OBJECT_REGISTRY.register
class ConcatDataset(data.ConcatDataset):
    """A wrapper of concatenated dataset with group flag.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`,
    addititionally concatenat the group flag of all dataset.

    Args:
        datasets: A list of datasets.
        with_flag: If concatenate datasets flags. If True,
            concatenate all datasets flag (all datasets must
            has flag attribute in this case).
            Default to False.
    """

    def __init__(
        self,
        datasets,
        with_flag: bool = False,
    ):
        super(ConcatDataset, self).__init__(datasets)

        if with_flag:
            flags = []
            for dataset in datasets:
                assert hasattr(dataset, "flag"), "dataset must has group flag"
                assert isinstance(
                    dataset.flag, np.ndarray
                ), "dataset flag must is numpy array instance"
                assert (
                    len(dataset) == dataset.flag.shape[0]
                ), "dataset flag length at axis 0 must equal to the dataset length"  # noqa: E501
                flags.append(dataset.flag)
            self.flag = np.concatenate(flags)
