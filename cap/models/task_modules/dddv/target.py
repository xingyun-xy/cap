# Copyright (c) Changan Auto. All rights reserved.

from typing import Mapping

import torch
from torch import nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["DepthTarget"]


@OBJECT_REGISTRY.register
class DepthTarget(nn.Module):  # noqa: D205,D400
    """Generate a downsampled depth gt with non-zero-min-pooling method
    for 2.5d task.

    Args:
        gt_depth_name (str): name of depth gt.
        downsample_scale (int): the ratio of input shape to output shape.
            NOTE: make sure downsample_scale is powers of 2.

    """

    def __init__(self, gt_depth_name: str, downsample_scale: int):
        super(DepthTarget, self).__init__()
        self.gt_depth_name = gt_depth_name
        assert (
            downsample_scale > 1
            and isinstance(downsample_scale, int)
            and (downsample_scale & (downsample_scale - 1)) == 0
        ), "downsample_scale is not valid!"
        self.max_pooling = nn.MaxPool2d(
            kernel_size=downsample_scale, stride=downsample_scale
        )
        pad_value = int(0.5 * (downsample_scale - 1))
        self.zero_pad = nn.ZeroPad2d((pad_value, 0, pad_value, 0))

    def forward(self, label_dict: Mapping, pred_dict: Mapping) -> Mapping:

        depth = label_dict[self.gt_depth_name]

        _, _, h, w = depth.shape
        pad_depth = self.zero_pad(depth)
        depth = pad_depth[:, :, 0:h, 0:w]

        max_pool_depth = self.max_pooling(depth)
        depth *= -1
        depth[depth == 0] = float("-inf")
        min_pool_depth = self.max_pooling(depth) * -1
        depth = torch.min(min_pool_depth, max_pool_depth)

        label_dict[self.gt_depth_name] = depth

        return label_dict
