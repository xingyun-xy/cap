# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, List

import torch
import torch.nn as nn

from cap.core.box_utils import zoom_boxes
from cap.core.data_struct.base_struct import Lines2D, Points2D
from cap.registry import OBJECT_REGISTRY

__all__ = ["FlankPointDecoder"]


@OBJECT_REGISTRY.register
class FlankPointDecoder(nn.Module):
    """
    Decoder for flank point.

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
        flank_label_features = head_out["rcnn_cls_pred"]
        flank_reg_features = head_out["rcnn_reg_pred"]

        try:
            batch_rois = torch.stack(
                [rois.dequantize() for rois in batch_rois]
            )
        except NotImplementedError:
            batch_rois = torch.stack(
                [rois.as_subclass(torch.Tensor) for rois in batch_rois]
            )

        bs = batch_rois.shape[0]
        flank_label_features = flank_label_features.reshape(
            bs,
            -1,
            flank_label_features.shape[1],
            flank_label_features.shape[2],
            flank_label_features.shape[3],
        )
        flank_reg_features = flank_reg_features.reshape(
            bs,
            -1,
            flank_reg_features.shape[1],
            flank_reg_features.shape[2],
            flank_reg_features.shape[3],
        )

        result = []
        for rois, flank_label_feature, flank_reg_feature in zip(
            batch_rois, flank_label_features, flank_reg_features
        ):
            # with shape (N,num_points,3)
            flank_points = self.single_forward(
                rois, flank_label_feature, flank_reg_feature
            )
            num_points = flank_points.shape[-2]
            points = {
                f"points{i}": Points2D(
                    points=flank_points[:, i, :2],
                    cls_idxs=torch.zeros_like(flank_points[:, i, -1]),
                    scores=flank_points[:, i, -1],
                )
                for i in range(num_points)
            }
            result.append(Lines2D(**points))

        return {"pred_flank": result}

    def single_forward(self, rois, flank_label_feature, flank_reg_feature):
        rois = rois[..., :4].reshape(-1, 4)
        rois = zoom_boxes(rois, (self.roi_expand_param, self.roi_expand_param))

        # with shape (N, 1)
        rois_x1, rois_y1, rois_x2, rois_y2 = torch.split(rois, 1, dim=-1)
        rois_height = rois_y2 - rois_y1
        rois_width = rois_x2 - rois_x1
        rois_cx = (rois_x1 + rois_x2) * 0.5

        # with shape (N,1,1)
        flank_label_pred = flank_label_feature.reshape(-1, 1, 1)
        # with shape (N,num_points,2)
        flank_reg_pred = flank_reg_feature.reshape(
            flank_label_pred.shape[0], -1, 2
        )

        # with shape (N,num_points,1)
        point_x = rois_cx + rois_width * flank_reg_pred[:, :, 0]
        point_x = point_x[..., None]
        point_y = rois_y1 + rois_height * torch.exp(flank_reg_pred[:, :, 1])
        point_y = point_y[..., None]

        # with shape (N,num_points,1)
        flank_label_pred = flank_label_pred.broadcast_to(point_x.size())
        # with shape (N,num_points,3)
        flank_points = torch.cat([point_x, point_y, flank_label_pred], dim=-1)

        return flank_points
