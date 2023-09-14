# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Tuple

import torch
import torch.nn as nn

from cap.core.box_utils import bbox_overlaps
from cap.registry import OBJECT_REGISTRY

__all__ = ["YOLOV3Matcher"]


# TODO(zhigang.yang, 0.5): merge with MaxIoUMatcher
@OBJECT_REGISTRY.register
class YOLOV3Matcher(nn.Module):
    """Bounding box classification label matcher by max iou.

       Different rule and return condition with MaxIoUMatcher.
       YOLOv3MaxIoUMatcher will be merged with MaxIoUMatcher in future.

    Args:
        ignore_thresh (float): Boxes whose IOU larger than ``ignore_thresh``
            is regarded as ignore samples for losses.
    """

    def __init__(self, ignore_thresh: float):
        super().__init__()
        self.ignore_thresh = ignore_thresh

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_boxes_num: torch.Tensor,
        im_hw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Method.

        Args:
            boxes (torch.Tensor): Box tensor with shape (B, N, 4) or (N, 4)
                when boxes are identical in the whole batch.
            gt_boxes (torch.Tensor): GT box tensor with shape
                (B, M, 5+). In one sample, if the number of gt boxes is
                less than M, the first M entries should be filled with
                real data, and others padded with arbitrary values.
            gt_boxes_num (torch.Tensor): GT box num tensor with shape (B).
                The actual number of gt boxes for each sample. Cannot be
                greater than M.

        Returns:
            (tuple): tuple contains:

                flag (torch.Tensor): flag tensor with shape (B, N). Entries
                    with value 1 represents ignore, 0 for neg.
                matched_pred_id (torch.Tensor): matched_pred_id tensor in
                    (B, M). The best matched of gt_boxes.
        """

        bs = boxes.shape[0]
        match_gts = []
        flags = []
        for b in range(bs):
            ious = bbox_overlaps(gt_boxes[b, :, :4], boxes[b])
            best_ious = torch.argmax(ious, -1)
            match_gts.append(best_ious.unsqueeze(0))

            ious = ious.t()
            flag_b = (torch.max(ious, -1)[0] > self.ignore_thresh).int()
            flags.append(flag_b.unsqueeze(0))
        flag = torch.cat(flags, 0)
        match_pred_id = torch.cat(match_gts, 0)
        return flag, match_pred_id
