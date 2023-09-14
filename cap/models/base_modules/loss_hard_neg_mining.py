# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional

import torch
from torch import nn


class LossHardNegativeMining(nn.Module):
    """hard negative mining based on input loss.

    Args:
        keep_pos: whether to keep all positive losses. If keep pos,
            losses w.r.t positive labels are all kept.
        neg_ratio: If all positive losses are kept, this value controls
            the number of entries of kept negative losses.
        hard_ratio: The ratio of hard negatives samples among
            1. all negative samples when keep_pos is True
            2. all kept samples when keep_pos is False
        ignore_largest_n: Number of entries to be ignored in each group
            with largest loss values.
        min_keep_num: minimum entries to be kept in each group.
        max_keep_num: max entries to be kept in each group.
        loss_thresh: Entries with loss value below this threshold will
            be ignored. If set to None, no one will be ignored.
        per_channel: Whether to do hard negative mining channel-wise.
    """

    _POSITIVE = 1
    _NEGATIVE = 0
    _IGNORE = -1

    def __init__(
        self,
        keep_pos: bool = True,
        neg_ratio: float = 0.5,
        hard_ratio: float = 0.5,
        ignore_largest_n: int = 0,
        min_keep_num: int = -1,
        max_keep_num: int = -1,
        loss_thresh: Optional[float] = None,
        per_channel: bool = True,
    ):
        """Do hard negative mining on loss value according to type mask."""
        super().__init__()
        self._keep_pos = keep_pos
        self._neg_ratio = neg_ratio
        self._hard_ratio = hard_ratio
        self._ignore_largest_n = ignore_largest_n
        self._min_keep_num = min_keep_num
        self._max_keep_num = max_keep_num
        self._loss_thresh = loss_thresh
        self._per_channel = per_channel

    @property
    def POSITIVE(self):
        return self._POSITIVE

    @property
    def NEGATIVE(self):
        return self._NEGATIVE

    @property
    def IGNORE(self):
        return self._IGNORE

    def forward(
        self,
        loss: torch.Tensor,
        type_mask: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D205,D400
        """
        Args:
            loss: loss tensor
            type_mask: type_mask tensor with the same shape as loss.
                The value of each entry is LossHardNegMining.POSITIVE
                (1) if it corresponds to positve sample, .NEGATIVE (0)
                for negative sample, .IGNORE (-1) if it should be ignored.
        Returns:
            A binary tensor with the same shape as loss. Entries with
                value 1 represents kept losses, and 0 the ignored.
                Suggeted to be applied on loss tensor on forward run.
        """

        assert loss.shape == type_mask.shape

        n, c = loss.shape[:2]
        group_num = c if self._per_channel else 1

        out_mask = (
            (type_mask == self._POSITIVE)
            if self._keep_pos
            else torch.zeros_like(type_mask)
        )

        # clone to avoid in-place modification of loss value
        loss = loss.view(n, c, -1).detach().clone()
        type_mask = type_mask.view(n, c, -1).detach()

        # filter out losses below threshold value, if applicable
        if self._loss_thresh is not None:
            loss = loss * (loss >= self._loss_thresh)

        # if keep foreground:
        #   loss of neg samples: keep value
        #   loss of pos samples: 0
        #   loss of ig samples: -1
        # else:
        #   loss of neg, pos samples: keep value
        #   loss of ig samples: -1
        loss *= (
            (type_mask == self._NEGATIVE)
            if self._keep_pos
            else (type_mask != self._IGNORE)
        )

        loss -= (type_mask == self._IGNORE) * 1

        loss_grouped = loss.transpose(0, 1).reshape(group_num, -1)

        type_mask_grouped = type_mask.transpose(0, 1).reshape(group_num, -1)

        idx = torch.arange(loss.numel(), device=loss.device).view(n, c, -1)
        idx_grouped = idx.transpose(0, 1).reshape(group_num, -1)

        count = (
            (type_mask_grouped == self._POSITIVE)
            if self._keep_pos
            else loss_grouped > 0
        ).sum(dim=1)

        ignore_count = (loss_grouped <= 0).sum(dim=1)

        loss_indices = loss_grouped.argsort(dim=1, descending=True)

        total_count = torch.clamp_min_(
            loss_grouped.shape[1] - self._ignore_largest_n - ignore_count, 0
        )

        if self._keep_pos:
            total_remain = (count / (1 - self._neg_ratio)).to(
                count.dtype
            ) - count
            selected = count
        else:
            total_remain = torch.clamp_min_(count, 0)
            selected = torch.zeros_like(count)

        if self._min_keep_num > 0:
            total_remain = torch.maximum(
                self._min_keep_num - selected, total_remain
            )
        if self._max_keep_num > 0:
            total_remain = torch.minimum(
                self._max_keep_num - selected, total_remain
            )

        total_remain = total_remain.clamp_min_(1)
        total_remain = torch.minimum(total_remain, total_count).clamp_min_(0)
        hard_remain = (
            (self._hard_ratio * total_remain)
            .to(total_remain.dtype)
            .clamp_min_(0)
        )
        normal_remain = total_remain - hard_remain

        chosen_idx = torch.zeros_like(idx_grouped).view(-1)
        chosen_count = 0

        for i in range(group_num):
            if total_count[i] < 1 or total_remain[i] < 1:
                continue

            # apply argsort result of loss values to indices,
            # to sort indices by loss.
            idx_grouped[i] = idx_grouped[i][loss_indices[i]]

            chosen_idx[chosen_count : chosen_count + hard_remain[i]] = (
                torch.arange(hard_remain[i], device=hard_remain.device)
                + self._ignore_largest_n
                + i * idx_grouped.shape[1]
            )
            chosen_count += hard_remain[i]

            # conditionally append randomly sampled normal negatives
            if normal_remain[i] > 0:
                rand_neg_idx = torch.randperm(
                    total_count[i] - self._ignore_largest_n - hard_remain[i]
                ).to(device=hard_remain.device)
                rand_neg_idx += self._ignore_largest_n + hard_remain[i]
                chosen_idx[chosen_count : chosen_count + normal_remain[i]] = (
                    rand_neg_idx[: normal_remain[i]] + i * idx_grouped.shape[1]
                )

                chosen_count += normal_remain[i]

        if chosen_count > 0:
            chosen_idx = chosen_idx[:chosen_count]
            out_mask_flatten = out_mask.view(-1)
            out_mask_flatten[idx_grouped.flatten()[chosen_idx]] = 1

        return out_mask
