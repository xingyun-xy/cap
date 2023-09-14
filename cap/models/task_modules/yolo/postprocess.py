# Copyright (c) Changan Auto. All rights reserved.

from typing import List, Sequence

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn.functional import nms

from cap.core.box_utils import box_center_to_corner
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list

__all__ = ["YOLOV3PostProcess"]


@OBJECT_REGISTRY.register
class YOLOV3PostProcess(nn.Module):
    """
    The postprocess of YOLOv3.

    Args:
        num_classes (int): The num classes of class branch.
        strides (List): strides of feature map.
        score_thresh (float): Score thresh of postprocess before nms.
        nms_thresh (float): Nms thresh.
        top_k (int): The output num of bboxes after postprocess.

    """

    def __init__(
        self,
        num_classes: int,
        strides: List,
        score_thresh: float = 0.01,
        nms_thresh: float = 0.45,
        top_k: int = 200,
    ):
        super(YOLOV3PostProcess, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    @torch.no_grad()
    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        anchors: List,
    ):
        inputs = _as_list(inputs)
        prediction = []
        for input, anchor, stride in zip(inputs, anchors, self.strides):
            prediction.append(self.get_preds_each_level(input, anchor, stride))
        prediction = torch.cat(prediction, 1)

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        box_corner = box_center_to_corner(prediction[:, :, :4])
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            score = (
                image_pred[:, 4]
                .unsqueeze(-1)
                .repeat(1, self.num_classes)
                .reshape(-1)
            )
            bbox = (
                image_pred[:, :4]
                .unsqueeze(-1)
                .repeat(1, 1, self.num_classes)
                .permute(0, 2, 1)
                .reshape(-1, 4)
            )
            class_conf = image_pred[:, 5 : 5 + self.num_classes].reshape(-1)
            class_pred = (
                torch.arange(self.num_classes, device=class_conf.device)
                .unsqueeze(0)
                .repeat(image_pred.shape[0], 1)
                .reshape(-1)
            )
            score = score * class_conf
            bboxes, scores, labels = nms(
                bbox,
                score,
                class_pred,
                self.nms_thresh,
                score_threshold=self.score_thresh,
                pre_nms_top_n=None,
                output_num=self.top_k,
            )
            output[image_i] = torch.cat(
                (bboxes, labels.unsqueeze(-1), scores.unsqueeze(-1)), -1
            )

        return output

    def get_preds_each_level(self, input, anchor, stride):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        input = (
            input.view(bs, -1, self.num_classes + 5, in_h, in_w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(input[..., 0])
        y = torch.sigmoid(input[..., 1])
        w = input[..., 2]
        h = input[..., 3]
        conf = torch.sigmoid(input[..., 4])
        pred_cls = torch.sigmoid(input[..., 5:])

        device = input.device

        # Add offset and scale with anchors
        anchor_w, anchor_h, grid_x, grid_y = anchor
        pred_boxes = torch.zeros_like(input[..., :4], device=device).float()
        pred_boxes[..., 0] = x + grid_x.view(x.shape)
        pred_boxes[..., 1] = y + grid_y.view(y.shape)
        pred_boxes[..., 2] = torch.exp(w) * anchor_w.view(w.shape)
        pred_boxes[..., 3] = torch.exp(h) * anchor_h.view(h.shape)

        _scale = torch.tensor([stride, stride] * 2, device=device).float()

        out = torch.cat(
            (
                pred_boxes.view(bs, -1, 4) * _scale,
                conf.view(bs, -1, 1),
                pred_cls.view(bs, -1, self.num_classes),
            ),
            -1,
        )
        return out
