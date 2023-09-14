# Copyright (c) Changan Auto. All rights reserved.
import torch
from changan_plugin_pytorch.nn.detection_post_process import (
    DetectionPostProcess,
)

from cap.registry import OBJECT_REGISTRY

__all__ = ["RetinaNetPostProcess"]


# TODO(kongtao.hu, 0.1): Modify the class name, it should be universal
@OBJECT_REGISTRY.register
class RetinaNetPostProcess(DetectionPostProcess):
    """The postprocess of RetinaNet.

    Args:
        score_thresh (float): Filter boxes whose score is lower than this.
        nms_thresh (float): thresh for nms.
        detections_per_img (int): Get top n boxes by score after nms.
        topk_candidates (int): Get top n boxes by score after decode.
    """

    def __init__(
        self,
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        topk_candidates: int = 1000,
    ):
        super(RetinaNetPostProcess, self).__init__(
            score_threshold=score_thresh,
            background_class_idx=None,
            size_threshold=None,
            image_size=None,
            pre_decode_top_n=None,
            post_decode_top_n=topk_candidates,
            iou_threshold=nms_thresh,
            pre_nms_top_n=None,
            post_nms_top_n=detections_per_img,
            nms_on_each_level=False,
            mode="normal",
        )

    @torch.no_grad()
    def forward(self, boxes, scores, regressions, image_shapes=None):
        scores = [score.sigmoid() for score in scores]
        ret_boxes, ret_scores, ret_labels = super(
            RetinaNetPostProcess, self
        ).forward(
            boxes=boxes,
            scores=scores,
            regressions=regressions,
            image_shapes=image_shapes,
        )
        predictions = []
        for (ret_box, ret_score, ret_label) in zip(
            ret_boxes, ret_scores, ret_labels
        ):
            ret_score = ret_score.unsqueeze(-1)
            ret_label = ret_label.unsqueeze(-1)
            pred = torch.cat([ret_box, ret_score, ret_label], dim=-1)
            predictions.append(pred)
        return predictions
