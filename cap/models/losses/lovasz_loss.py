# -*- coding: utf-8 -*-
# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to https://github.com/bermanmaxim/LovaszSoftmax

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

__all__ = ["LovaszSoftmaxLoss"]


def mean(iters, ignore_nan=False, empty=0):
    """Calculate mean value of iterable object, compatible with nan case."""
    iters = iter(iters)
    if ignore_nan:
        iters = ifilterfalse(torch.isnan, iters)
    try:
        n = 1
        accum = next(iters)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for _, v in enumerate(iters, 2):
        n += 1
        accum += v
    if n == 1:
        return accum
    return accum / n


@OBJECT_REGISTRY.register
class LovaszSoftmaxLoss(nn.Module):
    """Calculate lovasz softmax loss.

    Args:
        classes (str or list[int]): 'all' for all, 'present' for classes
          present in labels, or a list of classes to average.
        per_image (bool): whether to calculate loss per image or not.
        loss_weight (float): Global weight of loss. Defaults is 1.
        ignore_index (int): ignored value in gt_seg.
        loss_name (str): The key of loss in return dict. If None, return loss
            directly.

    """

    def __init__(
        self,
        classes="present",
        per_image=False,
        loss_weight: float = 1.0,
        ignore_index: int = -1,
        loss_name: Optional[str] = None,
    ):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_name = loss_name

    @autocast(enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):
        prob = F.softmax(pred, dim=1)
        if self.per_image:
            loss_lovasz = mean(
                lovasz_softmax_flat(
                    *flatten_probs(
                        prob_per_img.unsqueeze(0),
                        target_per_img.unsqueeze(0),
                        self.ignore_index,
                    ),
                    classes=self.classes
                )
                for prob_per_img, target_per_img in zip(prob, target)
            )
        else:
            loss_lovasz = lovasz_softmax_flat(
                *flatten_probs(prob, target, self.ignore_index),
                classes=self.classes
            )

        if self.loss_name is None:
            return loss_lovasz * self.loss_weight
        else:
            return {self.loss_name: loss_lovasz * self.loss_weight}


def lovasz_softmax_flat(probs, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss.

    Args:
      probs (tensor): class probabilities at each prediction (between 0 and 1),
        The shape is (P, C).
      labels (tensor): ground truth labels (between 0 and C - 1), shape (P).
      classes (str or list[int]): 'all' for all, 'present' for classes present
      in labels, or a list of classes to average.
    """
    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs.sum() * 0.0
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probs(probs, labels, ignore_index=None):
    """Flattens predictions in the batch."""
    if probs.dim() == 3:
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    valid_probs = probs[valid.nonzero().squeeze(-1)]
    valid_labels = labels[valid]
    return valid_probs, valid_labels


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
