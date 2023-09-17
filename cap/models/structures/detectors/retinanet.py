# Copyright (c) Changan Auto. All rights reserved.
import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from cap.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["RetinaNet"]


@OBJECT_REGISTRY.register
class RetinaNet(nn.Module):
    """The basic structure of retinanet.

    Args:
        backbone: backbone module or dict for building backbone module.
        neck: neck module or dict for building neck module.
        head: head module or dict for building head module.
        anchors: anchors module or dict for building anchors module.
        targets: targets module or dict for building target module.
        post_process: post_process module or dict for building
            post_process module.
        loss_cls: loss_cls module or dict for building loss_cls module.
        loss_reg: loss_reg module or dict for building loss_reg module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        anchors: Optional[nn.Module] = None,
        targets: Optional[nn.Module] = None,
        post_process: Optional[nn.Module] = None,
        loss_cls: Optional[nn.Module] = None,
        loss_reg: Optional[nn.Module] = None,
    ):
        super(RetinaNet, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.anchors = anchors
        self.targets = targets
        self.post_process = post_process
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg

    def rearrange_head_out(self, inputs: List[torch.Tensor], num: int):
        outputs = []
        for t in inputs:
            outputs.append(t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, num))
        return torch.cat(outputs, dim=1)

    def forward(self, data: Dict):
        _, _, height, width = data["img"].size()
        feat = self.backbone(data["img"])
        feat = self.neck(feat) if self.neck else feat
        cls_scores, bbox_preds = self.head(feat)

        if self.post_process is None:
            return cls_scores, bbox_preds

        anchors = self.anchors(feat)
        if self.training:
            cls_scores = self.rearrange_head_out(
                cls_scores, self.head.num_classes
            )
            bbox_preds = self.rearrange_head_out(bbox_preds, 4)
            gt_labels = [
                torch.cat(
                    [data["gt_bboxes"][i], data["gt_classes"][i][:, None] + 1],
                    dim=-1,
                )
                for i in range(len(data["gt_classes"]))
            ]
            gt_labels = [gt_label.float() for gt_label in gt_labels]
            _, labels = self.targets(anchors, gt_labels)
            avg_factor = labels["reg_label_mask"].sum()
            if avg_factor == 0:
                avg_factor += 1
            cls_loss = self.loss_cls(
                pred=cls_scores.sigmoid(),
                target=labels["cls_label"],
                weight=labels["cls_label_mask"],
                avg_factor=avg_factor,
            )
            reg_loss = self.loss_reg(
                pred=bbox_preds,
                target=labels["reg_label"],
                weight=labels["reg_label_mask"],
                avg_factor=avg_factor,
            )
            return {
                "cls_loss": cls_loss,
                "reg_loss": reg_loss,
            }
        else:
            preds = self.post_process(
                anchors,
                cls_scores,
                bbox_preds,
                [torch.tensor(shape) for shape in data["resized_shape"]],
            )
            assert (
                "pred_bboxes" not in data.keys()
            ), "pred_bboxes has been in data.keys()"
            data["pred_bboxes"] = preds
            return data

    def fuse_model(self):
        if self.backbone:
            self.backbone.fuse_model()
        if self.neck:
            self.neck.fuse_model()
        if self.head:
            self.head.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [
            self.backbone,
            self.neck,
            self.head,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        for module in [
            self.anchors,
            self.loss_cls,
            self.loss_reg,
            self.post_process,
            self.targets,
        ]:
            if module is not None:
                module.qconfig = None
