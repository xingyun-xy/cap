import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info, get_local_process_group

__all__ = [
    "DistributedGroupSampler",
]


@OBJECT_REGISTRY.register
class DistributedGroupSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Each batch data indices are sampled from one group in all of
    the groups. Groups are organized according to the dataset flags.

    .. note::
        Dataset is assumed to be constant size and must has
        flag attribute. Different number in flag array represent
        different groups. for example, in aspect ratio group flag,
        there are two groups, in which 0 represent h/w >= 1 and 1
        represent h/w < 1 group. Dataset flag must is numpy array
        instance, the dtype must is np.uint8 and length at axis 0
        must equal to the dataset length.

    Args:
        dataset: Dataset used for sampling.
        samples_per_gpu: Number samplers for each gpu.
            Default is 1.
        num_replicas: Number of processes participating in
            distributed training.
        rank: Rank of the current process within num_replicas.
        seed: random seed used in torch.Generator().
            This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(
        self,
        dataset,
        samples_per_gpu: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        if num_replicas is None or rank is None:
            rank, num_replicas = get_dist_info(get_local_process_group())

        super(DistributedGroupSampler, self).__init__(
            dataset,
            num_replicas,
            rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

        self.samples_per_gpu = samples_per_gpu

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag
        assert isinstance(
            self.flag, np.ndarray
        ), "dataset flag must is numpy array instance"
        assert (
            len(dataset) == self.flag.shape[0]
        ), "dataset flag length at axis 0 must equal to the dataset length"
        assert (
            self.flag.dtype == np.uint8
        ), "dataset flag dtype must is np.uint8"
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for size in self.group_sizes:
            self.num_samples += (
                int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas
                    )
                )
                * self.samples_per_gpu
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size <= 0:
                continue

            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            # add .numpy() to avoid bug when selecting indice in parrots.
            indice = indice[
                list(torch.randperm(int(size), generator=g).numpy())
            ].tolist()
            extra = int(
                math.ceil(
                    size * 1.0 / self.samples_per_gpu / self.num_replicas
                )
            ) * self.samples_per_gpu * self.num_replicas - len(indice)
            # pad indice
            tmp = indice.copy()
            for _ in range(extra // size):
                indice.extend(tmp)
            indice.extend(tmp[: extra % size])
            indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j]
            for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g
                )
            )
            for j in range(
                i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu
            )
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
