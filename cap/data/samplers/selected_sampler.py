import math
from typing import Callable, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from cap.registry import OBJECT_REGISTRY

T_co = TypeVar("T_co", covariant=True)

__all__ = ["SelectedSampler"]


@OBJECT_REGISTRY.register
class SelectedSampler(DistributedSampler[T_co]):
    """
    Distributed sampler that supports user-defined indices.

    Args:
        indices_function (Callable): Callback function given by user. Input are
            dataset and return a indices list.
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, `world_size` is retrieved
            from the current distributed group.
        rank (int, optional): Rank of the current process in `num_replicas`.
            By default, `rank` is retrieved from the current distributed group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle
            the indices.
        seed (int, optional): random seed used to shuffle the sampler if
            `shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader`
        iterator is necessary to make shuffling work properly across multiple
        epochs. Otherwise, the same ordering will be always used.

    """

    def __init__(
        self,
        indices_function: Callable,
        dataset: Dataset,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.indices_function = indices_function
        self.set_total_indices()
        self.set_num_samples()
        self.total_size = self.num_samples * self.num_replicas

    def set_total_indices(self) -> None:
        self.total_indices = torch.tensor(
            self.indices_function(self.dataset), dtype=torch.int64
        )

    def set_num_samples(self) -> None:
        if self.drop_last and len(self.total_indices) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.total_indices) - self.num_replicas)
                / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(
                len(self.total_indices) / self.num_replicas
            )

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            randperm = torch.randperm(len(self.total_indices), generator=g)
            indices = self.total_indices[randperm].tolist()
        else:
            indices = self.total_indices.tolist()

        if self.drop_last:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        else:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]

        assert len(indices) == self.total_size

        # get subsample for ench process
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
