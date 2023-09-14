# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection, gluonnas

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY
from .utils import weight_reduce_loss

__all__ = ["GIoULoss"]


@OBJECT_REGISTRY.register
class GIoULoss(nn.Module):
    """Generalized Intersection over Union Loss.

    Args:
        loss_name (str): The key of loss in return dict.
        loss_weight (float): Global weight of loss. Defaults is 1.0.
        eps (float): A small value to avoid zero denominator.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        dict: A dict containing the calculated loss, the key of loss is
        loss_name.
    """

    def __init__(self, loss_name, loss_weight=1.0, eps=1e-6, reduction="mean"):
        super(GIoULoss, self).__init__()
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _cal_giou_loss(pred, target, eps=1e-6):
        # overlap
        lt = torch.max(pred[..., :2], target[..., :2])
        rb = torch.min(pred[..., 2:], target[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        # union
        ap = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        ag = (target[..., 2] - target[..., 0]) * (
            target[..., 3] - target[..., 1]
        )
        union = ap + ag - overlap + eps
        # IoU
        ious = overlap / union
        # enclose area
        enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
        enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] + eps
        # GIoU
        gious = ious - (enclose_area - union) / enclose_area
        loss = 1 - gious
        return loss

    @autocast(enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):
        """
        Forward method.

        Args:
            pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                represent upper-left and lower-right point, with shape(N, 4).
            target (torch.Tensor): Corresponding gt_boxes, the same shape as
                pred.
            weight (torch.Tensor): Element-wise weight loss weight, with
                shape(N,).
            avg_factor (float): Average factor that is used to average the
                loss.
        """
        # cast to fp32
        result_dict = {}
        result_dict[self.loss_name] = None
        if pred is not None:
            pred = pred.float()
            if weight is not None and not torch.any(weight > 0):
                return {self.loss_name: (pred * weight).sum()}

            if weight is not None and weight.dim() > 1:
                # reduce the weight of shape (n, 4) to (n,) to match the
                # giou_loss of shape (n,)
                assert weight.shape == pred.shape
                weight = weight.mean(-1)

            loss = self._cal_giou_loss(pred, target, self.eps)
            loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)

            result_dict[self.loss_name] = self.loss_weight * loss

        return result_dict
