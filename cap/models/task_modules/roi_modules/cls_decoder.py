# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from cap.core.data_struct.base_struct import ClsLabels
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class SoftmaxRoIClsDecoder(nn.Module):
    def __init__(self, cls_name_mapping: Optional[Dict[int, str]] = None):
        super().__init__()
        self.cls_name_mapping = cls_name_mapping

    @torch.no_grad()
    def forward(
        self,
        batch_rois: torch.Tensor,
        head_out: Dict[str, List[torch.Tensor]],
    ):
        split_size = batch_rois[0].shape[0]

        batch_cls_pred = torch.stack(
            head_out["rcnn_cls_pred"].split(split_size), dim=0
        )[:, :, :, 0, 0]

        batch_scores, batch_cls_idxs = batch_cls_pred.max(dim=-1)

        results = []
        for scores, cls_idxs in zip(batch_scores, batch_cls_idxs):
            results.append(
                ClsLabels(
                    scores=scores,
                    cls_idxs=cls_idxs,
                    cls_name_mapping=self.cls_name_mapping,
                )
            )

        return {"pred_cls": results}
