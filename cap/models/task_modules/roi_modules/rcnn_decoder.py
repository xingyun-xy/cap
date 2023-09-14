# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn.functional import batched_nms

from cap.core.data_struct.base_struct import DetBoxes2D
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class RCNNDecoder(nn.Module):
    def __init__(
        self,
        bbox_decoder: nn.Module,
        cls_act_type: str,
        legacy_bbox: bool = False,
        im_hw: Optional[Tuple[int, int]] = None,
        nms_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        cls_name_mapping: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.bbox_decoder = bbox_decoder

        if cls_act_type in ("sigmoid", "identity"):
            self.index_start = 0
        elif cls_act_type == "softmax":
            self.index_start = 1
        else:
            raise NotImplementedError

        self.legacy_bbox = legacy_bbox
        self.im_hw = im_hw
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.cls_name_mapping = cls_name_mapping

    @torch.no_grad()
    def forward(
        self,
        batch_rois: List[torch.Tensor],
        head_out: Dict[str, List[torch.Tensor]],
    ):
        try:
            batch_rois = torch.stack(
                [rois.dequantize() for rois in batch_rois]
            )
        except NotImplementedError:
            batch_rois = torch.stack(
                [rois.as_subclass(torch.Tensor) for rois in batch_rois]
            )

        split_size = batch_rois.shape[1]

        batch_cls_pred = torch.stack(
            head_out["rcnn_cls_pred"].split(split_size), dim=0
        )[:, :, :, 0, 0]
        batch_reg_pred = torch.stack(
            head_out["rcnn_reg_pred"].split(split_size), dim=0
        )[:, :, :, 0, 0]

        # get cls score for foreground classes
        batch_cls_pred = batch_cls_pred[..., self.index_start :]

        # restore boxes from anchors and rpn prediction
        batch_boxes = self.bbox_decoder(batch_rois, batch_reg_pred)

        batch_size, num_anchors, num_classes = batch_cls_pred.shape
        cls_idx = torch.arange(num_classes, device=batch_boxes.device) + 1
        cls_idx = cls_idx.view(1, 1, -1).repeat(batch_size, num_anchors, 1)

        # combine box coordinates, class label, and scores (class-agnostic box)
        # to form batch_boxes of (B, N * num_classes, 6)
        batch_bboxes = torch.cat(
            [
                batch_boxes[:, :, None].repeat(1, 1, num_classes, 1),
                batch_cls_pred[..., None],
                cls_idx[..., None],
            ],
            dim=-1,
        ).flatten(start_dim=1, end_dim=-2)

        # NMS (temporarily in loop)
        res_bboxes = []
        for i in range(batch_size):
            valid_bboxes = batch_bboxes[i]

            boxes = DetBoxes2D(
                boxes=valid_bboxes[:, :4],
                scores=valid_bboxes[:, -2],
                cls_idxs=valid_bboxes[:, -1],
                cls_name_mapping=self.cls_name_mapping,
            )

            if self.nms_threshold is not None:
                if self.score_threshold is not None:
                    boxes = boxes.with_scores_gt(self.score_threshold)
                keep_idx = batched_nms(
                    boxes.boxes,
                    boxes.scores,
                    boxes.cls_idxs,
                    self.nms_threshold,
                )

                boxes = boxes[keep_idx]

            res_bboxes.append(boxes)

        return {"pred_boxes": res_bboxes}
