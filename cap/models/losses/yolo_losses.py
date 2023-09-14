# Copyright (c) Changan Auto. All rights reserved.

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

__all__ = ["YOLOV3Loss"]


@OBJECT_REGISTRY.register
class YOLOV3Loss(nn.Module):
    """
    The loss module of YOLOv3.

    Args:
        loss_xy (torch.nn.Module): Losses of xy.
        loss_wh (torch.nn.Module): Losses of wh.
        loss_conf (torch.nn.Module): Losses of conf.
        loss_cls (torch.nn.Module): Losses of cls.
    """

    def __init__(
        self,
        loss_xy: torch.nn.Module,
        loss_wh: torch.nn.Module,
        loss_conf: torch.nn.Module,
        loss_cls: torch.nn.Module,
        lambda_loss: list,
    ):
        super(YOLOV3Loss, self).__init__()
        self.loss_xy = loss_xy
        self.loss_wh = loss_wh
        self.loss_conf = loss_conf
        self.loss_cls = loss_cls
        self.lambda_loss = lambda_loss

    @autocast(enabled=False)
    def forward(self, input, target=None):
        bs = input.size(0)
        prediction = input.float()
        x = prediction[..., 0]
        y = prediction[..., 1]
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = prediction[..., 4]
        pred_cls = prediction[..., 5:]

        mask = target["mask"]
        tconf, tcls = target["tconf"], target["tcls"]
        tx = target["tboxes"][..., 0]
        ty = target["tboxes"][..., 1]
        tw = target["tboxes"][..., 2]
        th = target["tboxes"][..., 3]

        pos_mask = (mask == 1).int()
        neg_mask = (mask == 0).int()

        loss_x = self.loss_xy(x, tx) * pos_mask
        loss_y = self.loss_xy(y, ty) * pos_mask
        loss_w = self.loss_wh(w, tw) * pos_mask
        loss_h = self.loss_wh(h, th) * pos_mask
        loss_conf = (
            self.loss_conf(conf, tconf) * pos_mask
            + self.loss_conf(conf, tconf) * neg_mask
        )
        loss_cls = self.loss_cls(pred_cls, tcls) * pos_mask

        loss_xy_res = (loss_x + loss_y) / bs
        loss_wh_res = (loss_w + loss_h) / bs
        loss_conf_res = loss_conf / bs
        loss_cls_res = loss_cls / bs
        return loss_xy_res, loss_wh_res, loss_conf_res, loss_cls_res
