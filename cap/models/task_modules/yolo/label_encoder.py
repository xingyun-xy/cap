# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, Optional

import torch
import torch.nn as nn

from cap.core.box_utils import box_corner_to_center
from cap.registry import OBJECT_REGISTRY
from cap.utils.tensor_func import insert_row, take_row

__all__ = ["YOLOV3LabelEncoder"]


@OBJECT_REGISTRY.register
class YOLOV3LabelEncoder(nn.Module):
    """Encode gt and matching results for yolov3.

    Args:
       class_encoder (torch.nn.Module): config of class label encoder
    """

    def __init__(
        self,
        class_encoder: torch.nn.Module,
    ):
        super().__init__()
        self.class_encoder = class_encoder

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        ig_flag: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward method.

        Args:
            boxes (torch.Tensor): (B, N, 4), batched predicted boxes
            gt_boxes (torch.Tensor): (B, M, 5+), batched ground
                truth boxes, might be padded.
            match_pos_flag (torch.Tensor): (B, N) matched result
                of each predicted box
            match_gt_id (torch.Tensor): (B, M) matched gt box index
                of each predicted box
            ig_flag (torch.Tensor): (B, N) ignore matched result
                of each predicted box
        """
        mask = torch.zeros_like(match_pos_flag).float()
        tconf = torch.zeros_like(match_pos_flag).float()
        tcls = torch.zeros_like(match_pos_flag).float()
        tboxes = torch.zeros_like(boxes).float()

        gt_bboxes = box_corner_to_center(gt_boxes[..., :4])
        # ignore thresh
        mask[match_pos_flag > 0] = -1.0

        # best match
        insert_row(mask, match_gt_id, 1.0)
        insert_row(tconf, match_gt_id, 1.0)
        insert_row(tcls, match_gt_id, gt_boxes[..., 4].float())
        target_xy = gt_bboxes[..., 0:2] - gt_bboxes[..., 0:2]
        target_wh = torch.log(
            gt_bboxes[..., 2:4] / take_row(boxes, match_gt_id)[..., 0:2]
            + 1e-16
        )
        insert_row(
            tboxes,
            match_gt_id,
            torch.cat([target_xy, target_wh], dim=-1).float(),
        )

        tcls = self.class_encoder(tcls).float()

        return {"mask": mask, "tconf": tconf, "tcls": tcls, "tboxes": tboxes}
