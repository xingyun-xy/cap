# Copyright (c) Changan Auto. All rights reserved.

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["one_hot"]


def one_hot(data: Tensor, num_classes: int) -> Tensor:
    r"""
    Convert data to one hot format.

    Negative values will be converted to all zero vectors.
    Args:
        data (Tensor): Input data of shape (N, 1, ...).
        num_classes (int): Class number.

    Returns:
        Tensor: One hot output of shape (N, num_classes, ...).
    """
    data = (data + 1).clamp(0, num_classes)
    data = F.one_hot(data.to(dtype=torch.long), num_classes + 1)
    data = data.transpose(1, -1).squeeze(-1)

    return data[:, 1:, ...]
