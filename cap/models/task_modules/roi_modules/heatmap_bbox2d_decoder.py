# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Sequence

import torch
import torch.nn as nn
from changan_plugin_pytorch.functional import batched_nms_with_padding

from cap.core.box_utils import zoom_boxes
from cap.core.data_struct.base_struct import MultipleBoxes2D
from cap.registry import OBJECT_REGISTRY

__all__ = ["HeatmapBox2dDecoder"]


@OBJECT_REGISTRY.register
class HeatmapBox2dDecoder(nn.Module):
    """
    Decoder for heatmap box2d.

    Args:
        input_padding: Input padding.
        roi_wh_zoom_scale: Zoom scale for roi wh.

    """

    def __init__(
        self,
        score_threshold=0.0,
        roi_wh_zoom_scale: Sequence = (1.0, 1.0),
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.roi_wh_zoom_scale = roi_wh_zoom_scale

    @torch.no_grad()
    def forward(self, batch_rois, head_out):

        batch_size = len(batch_rois)

        try:
            rois = torch.cat([rois.dequantize() for rois in batch_rois], dim=0)
        except NotImplementedError:
            rois = torch.cat(
                [rois.as_subclass(torch.Tensor) for rois in batch_rois], dim=0
            )

        rois = zoom_boxes(rois, self.roi_wh_zoom_scale)
        rois_x1, rois_y1 = rois[..., 0], rois[..., 1]
        rois_w = rois[..., 2] - rois[..., 0]
        rois_h = rois[..., 3] - rois[..., 1]

        label_map, offset = (
            head_out["rcnn_cls_pred"],
            head_out["rcnn_reg_pred"],
        )
        _, num_classes, h, w = label_map.shape

        label_map = label_map.reshape(-1, num_classes, w * h)
        batch_scores, max_label_map_idx = torch.max(label_map, dim=-1)
        max_label_map_idx_h = torch.floor(max_label_map_idx / w)
        max_label_map_idx_w = max_label_map_idx - max_label_map_idx_h * w

        offset = offset.reshape(-1, 1, 4, w * h).repeat(1, num_classes, 1, 1)
        max_offset_idx = max_label_map_idx[..., None].repeat(1, 1, 4)
        max_offset = torch.gather(
            offset, index=max_offset_idx[..., None].expand_as(offset), dim=-1
        )[..., 0]

        back_scale_w = (rois_w / w)[:, None]
        back_scale_h = (rois_h / h)[:, None]
        relative_box = torch.cat(
            [
                back_scale_w
                * (max_offset[..., 0] + max_label_map_idx_w + 0.5),
                back_scale_h
                * (max_offset[..., 1] + max_label_map_idx_h + 0.5),
                back_scale_w
                * (max_offset[..., 2] + max_label_map_idx_w + 0.5),
                back_scale_h
                * (max_offset[..., 3] + max_label_map_idx_h + 0.5),
            ],
            dim=1,
        )
        rois = torch.stack([rois_x1, rois_y1, rois_x1, rois_y1], dim=-1)
        batch_boxes = relative_box + rois

        batch_labels = torch.stack(
            [
                torch.ones_like(batch_boxes[..., 0]) * i
                for i in range(num_classes)
            ],
            dim=-1,
        )

        res = []
        for boxes, scores, labels in zip(
            batch_boxes.view(batch_size, -1, *batch_boxes.shape[1:]),
            batch_scores.view(batch_size, -1, *batch_scores.shape[1:]),
            batch_labels.view(batch_size, -1, *batch_labels.shape[1:]),
        ):
            boxes_list = [box[None] for box in boxes.unbind()]
            scores_list = list(scores.unbind())
            cls_idxs_list = list(labels.unbind())

            for i, scores in enumerate(scores_list):
                idx = scores >= self.score_threshold
                boxes_list[i] = boxes_list[i][idx]
                scores_list[i] = scores_list[i][idx]
                cls_idxs_list[i] = cls_idxs_list[i][idx]

            res.append(
                MultipleBoxes2D(
                    boxes_list=boxes_list,
                    scores_list=scores_list,
                    cls_idxs_list=scores_list,
                )
            )

        return {"pred_boxes": res}


@OBJECT_REGISTRY.register
class HeatmapMultiBox2dDecoder(nn.Module):
    """
    Decoder for heatmap box2d.

    Args:
        input_padding: Input padding.
        roi_wh_zoom_scale: Zoom scale for roi wh.

    """

    def __init__(
        self,
        subbox_score_thresh: float,
        nms_iou_thresh: float,
        legacy_bbox: bool,
        max_box_num: Optional[int] = None,
        roi_wh_zoom_scale: Sequence = (1.0, 1.0),
    ):
        super().__init__()
        if max_box_num is not None:
            assert max_box_num > 0
        self.legacy_bbox = legacy_bbox
        self.max_box_num = max_box_num
        self.subbox_score_thresh = subbox_score_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.legacy_bbox = legacy_bbox
        self.roi_wh_zoom_scale = roi_wh_zoom_scale

    @torch.no_grad()
    def forward(self, batch_rois, head_out):

        batch_size = len(batch_rois)

        # (num_rois, 4)

        try:
            rois = torch.cat([rois.dequantize() for rois in batch_rois], dim=0)
        except NotImplementedError:
            rois = torch.cat(
                [rois.as_subclass(torch.Tensor) for rois in batch_rois], dim=0
            )

        rois = zoom_boxes(rois, self.roi_wh_zoom_scale)

        label_map, offset = (
            head_out["rcnn_cls_pred"],
            head_out["rcnn_reg_pred"],
        )
        _, num_classes, h, w = label_map.shape
        mask = label_map > self.subbox_score_thresh

        # num_rois, num_cls, 4, h, w
        offset = offset[:, None].repeat(1, num_classes, 1, 1, 1)

        # num_rois, num_cls, h, w
        offset_x1, offset_y1, offset_x2, offset_y2 = offset.unbind(dim=2)

        # (num_rois, 1, 1, 1)
        rois_x1, rois_y1 = (
            rois[..., 0][:, None, None, None],
            rois[..., 1][:, None, None, None],
        )
        rois_w = (rois[..., 2] - rois[..., 0])[:, None, None, None]
        rois_h = (rois[..., 3] - rois[..., 1])[:, None, None, None]

        back_scale_w = rois_w / w
        back_scale_h = rois_h / h

        # num_rois, num_cls, h, w
        ind_w_feature = torch.stack(
            [offset.new_ones(offset_x1.shape[:3]) * i for i in range(w)],
            dim=-1,
        )
        ind_h_feature = torch.cat(
            [
                offset.new_ones((*offset_y1.shape[:2], 1, w)) * i
                for i in range(h)
            ],
            dim=-2,
        )

        relative_box_x1 = back_scale_w * (offset_x1 + ind_w_feature + 0.5)
        relative_box_y1 = back_scale_h * (offset_y1 + ind_h_feature + 0.5)
        relative_box_x2 = back_scale_w * (offset_x2 + ind_w_feature + 0.5)
        relative_box_y2 = back_scale_h * (offset_y2 + ind_h_feature + 0.5)

        boxes_x1 = relative_box_x1 + rois_x1
        boxes_y1 = relative_box_y1 + rois_y1
        boxes_x2 = relative_box_x2 + rois_x1
        boxes_y2 = relative_box_y2 + rois_y1

        label_map *= mask
        boxes_x1 *= mask
        boxes_y1 *= mask
        boxes_x2 *= mask
        boxes_y2 *= mask

        label_map = label_map.view(-1, num_classes, w * h)
        boxes_x1 = boxes_x1.view(-1, num_classes, w * h)
        boxes_y1 = boxes_y1.view(-1, num_classes, w * h)
        boxes_x2 = boxes_x2.view(-1, num_classes, w * h)
        boxes_y2 = boxes_y2.view(-1, num_classes, w * h)

        max_box_num = h * w if self.max_box_num is None else self.max_box_num
        res_scores, max_label_map_idx = torch.topk(
            label_map, k=max_box_num, dim=-1
        )

        res_box_x1_list = []
        res_box_y1_list = []
        res_box_x2_list = []
        res_box_y2_list = []

        for idx_i in max_label_map_idx.unbind(-1):
            res_box_x1_list.append(
                torch.gather(
                    boxes_x1,
                    index=idx_i[..., None].expand_as(boxes_x1),
                    dim=-1,
                )[..., 0]
            )
            res_box_y1_list.append(
                torch.gather(
                    boxes_y1,
                    index=idx_i[..., None].expand_as(boxes_y1),
                    dim=-1,
                )[..., 0]
            )
            res_box_x2_list.append(
                torch.gather(
                    boxes_x2,
                    index=idx_i[..., None].expand_as(boxes_x2),
                    dim=-1,
                )[..., 0]
            )
            res_box_y2_list.append(
                torch.gather(
                    boxes_y2,
                    index=idx_i[..., None].expand_as(boxes_y2),
                    dim=-1,
                )[..., 0]
            )

        res_box_x1 = torch.stack(res_box_x1_list, dim=-1)
        res_box_y1 = torch.stack(res_box_y1_list, dim=-1)
        res_box_x2 = torch.stack(res_box_x2_list, dim=-1)
        res_box_y2 = torch.stack(res_box_y2_list, dim=-1)

        res_boxes = torch.stack(
            [
                res_box_x1,
                res_box_y1,
                res_box_x2,
                res_box_y2,
            ],
            dim=-1,
        ).flatten(start_dim=1, end_dim=2)

        res_labels = (
            torch.cat(
                [
                    res_scores.new_ones(
                        (res_scores.shape[0], 1, *res_scores.shape[2:])
                    )
                    * i
                    for i in range(num_classes)
                ],
                dim=1,
            )
            .flatten(start_dim=1, end_dim=2)
            .long()
        )

        res_scores = res_scores.flatten(start_dim=1, end_dim=2)

        res = []
        for roi_boxes, roi_scores, roi_labels in zip(
            res_boxes.view(batch_size, -1, *res_boxes.shape[1:]),
            res_scores.view(batch_size, -1, *res_scores.shape[1:]),
            res_labels.view(batch_size, -1, *res_labels.shape[1:]),
        ):
            boxes_list, scores_list, cls_idxs_list = [], [], []
            for boxes, scores, labels in zip(
                roi_boxes, roi_scores, roi_labels
            ):
                mask = scores >= self.subbox_score_thresh
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                if boxes.numel() > 0:
                    keep_idx = batched_nms_with_padding(
                        boxes[None],
                        scores[None],
                        labels[None],
                        self.nms_iou_thresh,
                        len(boxes),
                        max_box_num,
                        self.legacy_bbox,
                    )[0]
                    keep_idx = keep_idx[keep_idx >= 0]

                    boxes_list.append(boxes[keep_idx])
                    scores_list.append(scores[keep_idx])
                    cls_idxs_list.append(labels[keep_idx])
                else:
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                    cls_idxs_list.append(labels)

            res.append(
                MultipleBoxes2D(
                    boxes_list=boxes_list,
                    scores_list=scores_list,
                    cls_idxs_list=cls_idxs_list,
                )
            )

        return {"pred_boxes": res}
