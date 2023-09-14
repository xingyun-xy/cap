# Copyright (c) Changan Auto. All rights reserved.
from collections.abc import Sequence
from typing import Any, List, Union

import torch
from torchvision.ops.boxes import nms

from cap.utils.apply_func import _as_list

__all__ = [
    "_check_strides",
    "_take_features",
    "_get_paddings_indicator",
    "multi_class_nms",
]


def _check_strides(strides: Union[int, Sequence], valid_strides: Sequence):
    # TODO(min.du, 0.1): why return strides? #
    strides = _as_list(strides)
    for stride_i in strides:
        assert stride_i in valid_strides
    return strides


def _take_features(
    features: List[Any],
    feature_scales: List[int],
    take_scales: List[int],
) -> List[Any]:
    r"""Select features from all features.

    Args:
        features (list): a list contains all features.
        feature_scales (list): a list contains the scales
            corresponding to features.
        take_scales (list): a list contains scales to selectd.

    """
    features = _as_list(features)
    feature_scales = _as_list(feature_scales)
    take_scales = _as_list(take_scales)
    assert len(features) == len(feature_scales), "%d vs. %d" % (
        len(features),
        len(feature_scales),
    )

    take_features = []
    for scale in take_scales:
        assert scale in feature_scales
        take_features.append(features[feature_scales.index(scale)])
    return take_features


def _get_paddings_indicator(
    actual_num: torch.Tensor, max_num: int, axis=0
) -> torch.Tensor:
    """Create boolean mask by actual number of a padded tensor.

    This function helps to identify pillars where there's too little data.

    Example:

    actual_num = [[3,3,3,3,3]] (5 pillars, each contains 3 lidar points)
    max_num: 4 (turns to [[0, 1, 2, 3, 4]])
    will return: [[T, T, T, F, F]]

    Args:
        actual_num (torch.Tensor): NxM tensor, where N is batch size and M is
            total number of pillars. In certain cases N can be omitted.
        max_num (int): max number of points allowed in a pillar.
        axis (int, optional): axis position. Defaults to 0.

    Returns:
        [torch.Tensor]: indicates where the tensor should be padded.
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device
    ).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


def multi_class_nms(bbox, score, category, iou_threshold):
    """Multiple classes NMS.

    Args:
        bbox (n, 4): boxes to perform NMS on. They are expected to be in
        `(x1, y1, x2, y2)` format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        score (n): scores for each one of the boxes
        category (n): category id for each one of the boxes
        iou_threshold : discards all overlapping boxes with IoU > iou_threshold
    Return:
        keep : int64 tensor with the indices of the elements that have
            been kept by NMS
    """
    index = torch.arange(bbox.shape[0], dtype=torch.int64)
    unique_category = torch.unique(category)
    res_idx = []
    for cate_id in unique_category:
        mask = category == cate_id
        if mask.sum() == 0:
            continue
        c_bbox = bbox[mask]  # (?,4)
        c_score = score[mask]  # (?,)
        keep = nms(c_bbox, c_score, iou_threshold)
        c_id = index[mask][keep].tolist()
        res_idx.extend(c_id)
    return torch.tensor(res_idx, dtype=torch.int64).to(bbox.device)
