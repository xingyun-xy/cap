# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, List

import torch
import torch.nn as nn

from cap.core.box_utils import zoom_boxes
from cap.core.data_struct.base_struct import Lines2D, Points2D
from cap.registry import OBJECT_REGISTRY

__all__ = ["GroundLinePointDecoder"]


@OBJECT_REGISTRY.register
class GroundLinePointDecoder(nn.Module):
    """
    Decoder for groud line point.

    Args:
        roi_expand_param: Param to expand roi.
        input_padding: Input padding.

    """

    def __init__(
        self,
        roi_expand_param: float = 1.0,
    ):
        super().__init__()
        self.roi_expand_param = roi_expand_param

    @torch.no_grad()
    def forward(
        self,
        batch_rois: List[torch.Tensor],
        head_out: Dict[str, List[torch.Tensor]],
    ):
        gdl_label_features = head_out["rcnn_cls_pred"]
        gdl_reg_features = head_out["rcnn_reg_pred"]

        try:
            batch_rois = torch.stack(
                [rois.dequantize() for rois in batch_rois]
            )
        except NotImplementedError:
            batch_rois = torch.stack(
                [rois.as_subclass(torch.Tensor) for rois in batch_rois]
            )

        bs = batch_rois.shape[0]
        gdl_label_features = gdl_label_features.reshape(
            bs,
            -1,
            gdl_label_features.shape[1],
            gdl_label_features.shape[2],
            gdl_label_features.shape[3],
        )
        gdl_reg_features = gdl_reg_features.reshape(
            bs,
            -1,
            gdl_reg_features.shape[1],
            gdl_reg_features.shape[2],
            gdl_reg_features.shape[3],
        )

        result = []
        for rois, gdl_label_feature, gdl_reg_feature in zip(
            batch_rois, gdl_label_features, gdl_reg_features
        ):
            gdl_pts = self.single_forward(
                rois, gdl_label_feature, gdl_reg_feature
            )
            pts = {
                f"points{i}": Points2D(
                    points=gdl_pts[:, i * 2 : i * 2 + 2],
                    cls_idxs=torch.zeros_like(gdl_pts[:, -1]),
                    scores=gdl_pts[:, -1],
                )
                for i in range(2)
            }
            result.append(Lines2D(**pts))
        return {"pred_gdl": result}

    def single_forward(self, rois, gdl_label_feature, gdl_reg_feature):
        rois = rois[..., :4].reshape(-1, 4)
        rois = zoom_boxes(rois, (self.roi_expand_param, self.roi_expand_param))

        rois_x1, rois_y1, rois_x2, rois_y2 = torch.split(
            rois, 1, dim=-1
        )  # [N, 1]
        rois_height = rois_y2 - rois_y1

        gdl_label_pred = gdl_label_feature.reshape(-1, 1)
        delta_left, delta_right = torch.split(
            gdl_reg_feature.reshape(-1, 2), 1, dim=-1
        )
        left = rois_y2 + delta_left * rois_height
        right = rois_y2 + delta_right * rois_height

        gdl_pts = torch.cat(
            [rois_x1, left, rois_x2, right, gdl_label_pred], dim=-1
        )
        return gdl_pts
