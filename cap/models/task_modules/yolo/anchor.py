# Copyright (c) Changan Auto. All rights reserved.

from typing import List

import numpy as np
import torch
from torch import nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["YOLOV3AnchorGenerator"]


@OBJECT_REGISTRY.register
class YOLOV3AnchorGenerator(nn.Module):
    """Anchors generator for yolov3.

    Args:
        anchors (List): list if anchor size.
        strides (List): strides of feature map for anchors.
        image_size (List): Input size of image.
    """

    def __init__(
        self,
        anchors: List,
        strides: List,
        image_size: List,
    ):
        super(YOLOV3AnchorGenerator, self).__init__()
        self.anchors = anchors
        self.strides = strides
        self.image_size = image_size

    def forward(self, x):
        assert len(self.anchors) == len(self.strides)
        results = []
        bs = x[0].size(0) if isinstance(x, list) else x.size(0)
        device = x[0].device if isinstance(x, list) else x.device

        for anchors, strides in zip(self.anchors, self.strides):
            num_anchors = len(anchors)
            scaled_anchors = [
                (a_w / strides, a_h / strides) for a_w, a_h in anchors
            ]
            anchor_xyxy = torch.tensor(
                np.concatenate(
                    (np.zeros((num_anchors, 2)), np.array(anchors)), 1
                ),
                device=device,
            )
            anchor_xyxy = anchor_xyxy.unsqueeze(0).unsqueeze(0)

            # Calculate anchor w, h
            in_h = int(self.image_size[0] / strides)
            in_w = int(self.image_size[1] / strides)
            anchor_w = (
                torch.tensor(scaled_anchors, device=device)[:, 0]
                .float()
                .unsqueeze(-1)
            )
            anchor_h = (
                torch.tensor(scaled_anchors, device=device)[:, 1]
                .float()
                .unsqueeze(-1)
            )
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w)

            anchor_xyxy = anchor_xyxy.repeat(bs, in_h * in_w, 1, 1).view(
                bs, -1, 4
            )
            # Calculate offsets for each grid
            grid_x = (  # yapf:disable
                torch.linspace(0, in_w - 1, in_w, device=device)
                .repeat(in_w, 1)
                .repeat(bs * num_anchors, 1, 1)
                .float()
            )
            grid_y = (  # yapf:disable
                torch.linspace(0, in_h - 1, in_h, device=device)
                .repeat(in_h, 1)
                .t()
                .repeat(bs * num_anchors, 1, 1)
                .float()
            )

            if self.training:
                results.append(anchor_xyxy)
            else:
                results.append([anchor_w, anchor_h, grid_x, grid_y])
        results = torch.cat(results, 1) if self.training else results
        return results
