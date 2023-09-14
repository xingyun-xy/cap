# Copyright (c) Changan Auto. All rights reserved.

from typing import Any, List, Union

import numpy as np
import torch
from changan_plugin_pytorch.qtensor import QTensor

__all__ = [
    "take_row",
    "insert_row",
    "select_sample",
    "mean_with_mask",
]


def take_row(in_tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """loop-free batched row-wise indexing.

    The behavior is equivalent to::

        res = torch.stack([t[i] for t, i in zip(in_tensor, index)], dim=0)

    and the result is::

        res[i, j] = in_tensor[i, index[i, j]]

    Args:
        in_tensor (torch.Tensor): Input tensor with shape (B, N, ...).
        index (torch.Tensor): Index tensor with shape (B, M), where
            each entry should be less than N.
    """
    arr = torch.arange(in_tensor.shape[0], device=index.device)[:, None]
    flatten_index = (index + arr * in_tensor.shape[1]).flatten()

    last_dims = in_tensor.shape[2:]
    flatten_target = in_tensor.view(-1, *last_dims)
    if flatten_target.shape[0] == 0:
        flatten_target = (
            torch.unsqueeze(torch.ones(last_dims, device=index.device), dim=0)
            * -1
        )
    indexed = flatten_target[flatten_index.type(torch.long)].view(
        in_tensor.shape[0], -1, *last_dims
    )
    return indexed


def insert_row(
    in_tensor: torch.Tensor,
    index: torch.Tensor,
    target: Union[int, float, torch.Tensor],
) -> None:
    """Insert target to in_tensor by index provide by index param.

    The behavior is equivalent to::

      torch.stack([t[i] for t, i in zip(in_tensor, index)], dim=0) = target[i]

    and the result is::

      in_tensor[i, index[i, j]] = target[i, j]

    while the in_tensor will be modified by target.

    Args:
        in_tensor (torch.Tensor): Input tensor with shape (B, N, ...).
        index (torch.Tensor): Index tensor with shape (B, M), where
            each entry should be less than N.
        target (int, float, torch.Tensor): Target tensor provided.
            If target is torch.Tensor, it must be with shape (B, M).
    """

    last_dims = in_tensor.shape[2:]

    arr = torch.arange(in_tensor.shape[0], device=index.device)[:, None]
    flatten_index = (index + arr * in_tensor.shape[1]).flatten().long()

    flatten_target = in_tensor.view(-1, *last_dims)
    if isinstance(target, torch.Tensor):
        target = target.view(flatten_target[flatten_index].shape)
    flatten_target[flatten_index] = target


def select_sample(data: Any, bool_index: Union[List, torch.Tensor]):
    r"""Select sample according to bool index, return a tensor after selecting.

    Args:
        data : torch.tensor/QTensor/a list of tensor/ a dict of tensor,
            each tensor`s shape is (b,...)
        bool_index : torch.tensor, shape is (b,)

    """

    if type(data) == torch.Tensor:
        return data[bool_index]

    elif type(data) == QTensor:
        return QTensor(data.data[bool_index], data.scale.clone(), data.dtype)

    elif isinstance(data, (list, tuple)):
        return [select_sample(x, bool_index) for x in data]

    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = select_sample(v, bool_index)
        return res
    else:
        raise TypeError("donot support the type to select")


def mean_with_mask(
    x: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Mean of elements along the last dimension in x with True flags.

    Args:
        x: Input tensor or ndarray with any shapes.
        mask: Mask, which can be broadcast to x.
    """
    if isinstance(x, torch.Tensor):
        agent_num = torch.sum(mask, dim=-1)
        x = torch.sum(x * mask, dim=-1) / torch.clamp(agent_num, min=1.0)
    elif isinstance(x, np.ndarray):
        agent_num = np.sum(mask, axis=-1)
        x = np.sum(x * mask, axis=-1) / np.clip(
            agent_num, a_max=None, a_min=1.0
        )
    else:
        raise NotImplementedError(f"unspport input type {type(x)}")
    return x
