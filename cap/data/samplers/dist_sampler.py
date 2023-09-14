# Copyright (c) Changan Auto. All rights reserved.
from typing import Optional

import torch
from torch.utils.data.distributed import DistributedSampler

from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info, get_local_process_group

__all__ = ["DistSamplerHook"]


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias(torch.utils.data.DistributedSampler)
class DistSamplerHook(DistributedSampler):  # noqa: D205,D400
    """
    The hook api for torch.utils.data.DistributedDampler.
    Used to get local rank and num_replicas before create DistributedSampler.

    Args:
        dataset: compose dataset
        num_replicas: same as DistributedSampler
        rank: Same as DistributedSampler
        shuffle: if shuffle data
        seed: random seed
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None or rank is None:
            rank, num_replicas = get_dist_info(get_local_process_group())
        super(DistSamplerHook, self).__init__(
            dataset, num_replicas, rank, shuffle, seed, drop_last
        )  # noqa
