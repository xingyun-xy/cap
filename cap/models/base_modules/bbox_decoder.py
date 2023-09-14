# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Tuple

import torch
import torch.nn as nn

from cap.core.box_utils import box_center_to_corner, box_corner_to_center
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class XYWHBBoxDecoder(nn.Module):
    """Encode bounding box in XYWH ways (proposed in RCNN).

    Args:
        legacy_bbox (:obj:'bool', optional): Whether to represent bbox
            in legacy way. Default is False.
        reg_mean (:obj:'bool', tuple): Mean value to be subtracted from
            bbox regression task in each coordinate.
        reg_std (:obj:'bool', tuple): Standard deviation value to be
            divided from bbox regression task in each coordinate.
    """

    def __init__(
        self,
        legacy_bbox: Optional[bool] = False,
        reg_mean: Optional[Tuple] = (0.0, 0.0, 0.0, 0.0),
        reg_std: Optional[Tuple] = (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()

        assert len(reg_mean) == 4 and len(reg_std) == 4

        self.register_buffer(
            "reg_mean", torch.tensor(reg_mean), persistent=False
        )
        self.register_buffer(
            "reg_std", torch.tensor(reg_std), persistent=False
        )

        self._legacy_bbox = legacy_bbox

    def forward(
        self, boxes: torch.Tensor, boxes_delta: torch.Tensor
    ) -> torch.Tensor:

        box_cx, box_cy, box_w, box_h = box_corner_to_center(
            boxes, split=True, legacy_bbox=self._legacy_bbox
        )

        boxes_delta = (
            boxes_delta.detach().clone() * self.reg_std + self.reg_mean
        )

        dx, dy, dw, dh = torch.split(boxes_delta, 1, dim=-1)

        pred_cx = dx * box_w + box_cx
        pred_cy = dy * box_h + box_cy
        pred_w = torch.exp(dw) * box_w
        pred_h = torch.exp(dh) * box_h

        pred_boxes = box_center_to_corner(
            torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=-1),
            legacy_bbox=self._legacy_bbox,
        )

        return pred_boxes
