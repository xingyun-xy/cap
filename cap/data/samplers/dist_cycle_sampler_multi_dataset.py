# Copyright (c) Changan Auto. All rights reserved.
import math
from typing import List, Optional

import torch
from torch.utils.data.distributed import DistributedSampler

from cap.data.datasets.dataset_wrappers import ComposeDataset
from cap.registry import OBJECT_REGISTRY

__all__ = ["DistributedCycleMultiDatasetSampler"]


@OBJECT_REGISTRY.register
class DistributedCycleMultiDatasetSampler(
    DistributedSampler
):  # noqa: D205,D400
    """
    In one epoch period, do cyclic sampling on the dataset according to
    iter_time.

    Args:
        dataset: compose dataset
        num_replicas (int): same as DistributedSampler
        rank (int): Same as DistributedSampler
        shuffle: if shuffle data
        seed: random seed
    """

    def __init__(
        self,
        dataset: ComposeDataset,
        batchsize_list: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super(DistributedCycleMultiDatasetSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, seed
        )  # noqa
        assert len(batchsize_list) > 0
        self.dataset_len_list = self.dataset.len_list
        self.batchsize_list = batchsize_list
        self.num_datasets = len(batchsize_list)

        # getting datasets max iter_time
        self.iter_time = max(
            [
                math.ceil(length / batchsize)
                for length, batchsize in zip(
                    self.dataset_len_list, self.batchsize_list
                )
            ]
        )
        # padding the data's length to make sure the data will be trained,
        # on the one epoch
        self.train_need_num_imgs = [
            bs * self.iter_time for bs in batchsize_list
        ]
        # getting the datasets repeat time
        self.repeated_num_list = [
            int(num_imgs // self.dataset_len_list[i])
            for i, num_imgs in enumerate(self.train_need_num_imgs)
        ]  # noqa
        self.random_select_num_list = [
            num_imgs
            - self.repeated_num_list[i] * self.dataset_len_list[i]  # noqa
            for i, num_imgs in enumerate(self.train_need_num_imgs)
        ]  # noqa

        # the num_dateset must be divided by bachsize*num_replicas
        self.num_samples_list = [
            int(
                math.ceil(
                    num_imgs * 1.0 / (self.num_replicas * batchsize_list[i])
                )
            )  # noqa
            for i, num_imgs in enumerate(self.train_need_num_imgs)
        ]  # noqa
        # num_samples is the compose dataset's length
        self.num_samples = sum(
            [
                num_samples * bs
                for num_samples, bs in zip(
                    self.num_samples_list, batchsize_list
                )
            ]
        )

        self.total_size_list = [
            num_samples * self.num_replicas * batchsize_list[i]  # noqa
            for i, num_samples in enumerate(self.num_samples_list)
        ]  # noqa

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices_list = []
            for i, repeated_num in enumerate(self.repeated_num_list):
                repeated_indices = []
                for _ in range(repeated_num):
                    repeated_indices.extend(
                        torch.randperm(
                            self.dataset_len_list[i], generator=g
                        ).tolist()  # noqa
                    )
                extra_random_indices = torch.randperm(
                    self.dataset_len_list[i], generator=g
                ).tolist()[
                    : self.random_select_num_list[i]
                ]  # noqa
                indices_list.append(repeated_indices + extra_random_indices)
        else:
            for i in range(len(self.total_size_list)):
                assert self.total_size_list[i] == self.dataset_len_list[i]
            indices_list = [
                list(range(num_imgs)) for num_imgs in self.dataset_len_list
            ]  # noqa

        indices_list = [
            indices
            + indices[: (self.total_size_list[i] - len(indices))]  # noqa
            for i, indices in enumerate(indices_list)
        ]
        for i in range(len(indices_list)):
            assert len(indices_list[i]) == self.total_size_list[i]

        # subsample
        indices_list = [
            indices[self.rank : self.total_size_list[i] : self.num_replicas]
            for i, indices in enumerate(indices_list)
        ]  # noqa

        # generate final indices according to batchsize_list
        indices = []
        iter_time = len(indices_list[0]) // self.batchsize_list[0]
        for i in range(len(self.batchsize_list)):
            assert len(indices_list[i]) % self.batchsize_list[i] == 0
            assert iter_time == len(indices_list[i]) / self.batchsize_list[i]
        for i in range(iter_time):
            for j in range(self.num_datasets):
                batchsize = self.batchsize_list[j]
                indices.extend(
                    indices_list[j][i * batchsize : (i + 1) * batchsize]
                )
        return iter(indices)
