# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Union

from torch.utils.data import DataLoader, IterableDataset

__all__ = ["has_len", "get_len"]

logger = logging.getLogger(__name__)


def has_iterable_dataset(dataloader: DataLoader):
    return hasattr(dataloader, "dataset") and isinstance(
        dataloader.dataset, IterableDataset
    )


def has_len(dataloader: DataLoader) -> bool:
    """Check if a given Dataloader has __len__ method implemented.

    i.e. if it is a finite dataloader or infinite dataloader.
    """

    try:
        # try getting the length
        if len(dataloader) == 0:
            raise ValueError(
                "`Dataloader` returned 0 length. Please make sure"
                "that it returns at least 1 batch"
            )
        has_len = True
    except TypeError:
        has_len = False
    except NotImplementedError:
        has_len = False

    if has_len and has_iterable_dataset(dataloader):
        logger.warning(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-processing data loading "
            "(e.g. batch size > 1), this can lead to unintended side effects "
            "since the samples will be duplicated."
        )
    return has_len


def get_len(dataloader: DataLoader) -> Union[int, float]:
    try:
        length = len(dataloader)
    except Exception:
        length = float("inf")
    return length
