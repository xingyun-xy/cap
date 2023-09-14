from numbers import Integral, Real

import torch
from torch.jit.annotations import List, Tuple, Optional
from changan_plugin_pytorch.utils import fx_helper
from changan_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn,
)

from .functional import decode, filter_by_max_score, get_top_n, nms

__all__ = ["DetectionPostProcess"]


@fx_helper.wrap
class DetectionPostProcess(torch.nn.Module):
    """
    General post process for object detection models.
    Compatible with YOLO, SSD, RetinaNet, Faster-RCNN (RPN & RCNN), etc.
    Note that this is a float OP, please use after DequantStubs.

    Args:
        score_threshold (int, optional):
            Filter boxes whose score is lower than this.
            Defaults to 0.
        regression_scale (Tuple[float, float, float, float], optional):
            Scale to be multiplyed to box regressions.
            Defaults to None.
        background_class_idx (int, optional):
            Specify the class index to be ignored.
            Defaults to None.
        size_threshold (float, optional):
            Filter bixes whose height or width smaller than this.
            Defaults to None.
        image_size (Tuple[int, int], optional):
            Clip boxes to image sizes. Defaults to None.
        pre_decode_top_n (int, optional):
            Get top n boxes by objectness (first element in the score vector)
                before decode.
            Defaults to None.
        post_decode_top_n (int, optional):
            Get top n boxes by score after decode.
            Defaults to None.
        iou_threshold (float, optional):
            IoU threshold for nms.
            Defaults to None.
        pre_nms_top_n (int, optional):
            Get top n boxes by score before nms.
            Defaults to None.
        post_nms_top_n (int, optional):
            Get top n boxes by score after nms.
            Defaults to None.
        nms_on_each_level (bool, optional):
            Whether do nms on each level seperately.
            Defaults to False.
        mode (str, optional):
            Only support 'normal' and 'yolo'. If set to 'yolo':
                1. Box will be filtered by objectness rathen than
                    classification scores.
                2. dx, dy in regressions will be treated as absolute offset.
                3. Objectness will be multiplyed to classification scores.
            Defaults to 'normal'.
    """

    def __init__(
        self,
        # Params for filter_by_max_score.
        score_threshold=0,
        # Params for decode.
        regression_scale=None,
        background_class_idx=None,
        size_threshold=None,
        image_size=None,
        pre_decode_top_n=None,
        post_decode_top_n=None,
        # Params for nms.
        iou_threshold=None,
        pre_nms_top_n=None,
        post_nms_top_n=None,
        nms_on_each_level=False,
        mode="normal",
    ):
        super(DetectionPostProcess, self).__init__()
        assert isinstance(
            score_threshold, (Real, type(None))
        ), "param 'score_threshold' must be real type"
        assert isinstance(
            regression_scale, (list, tuple, type(None))
        ), "param 'regression_scale' must be list or tuple"
        if regression_scale:
            assert (
                len(regression_scale) == 4
            ), "param 'regression_scale' must be 4 in length"
            for v in regression_scale:
                assert isinstance(
                    v, Real
                ), "members of 'regression_scale' must be real type"
        assert isinstance(
            background_class_idx, (Integral, type(None))
        ), "param 'background_class_idx' must be int type"
        assert isinstance(
            size_threshold, (Real, type(None))
        ), "param 'size_threshold' must be real type"
        assert isinstance(
            image_size, (list, tuple, type(None))
        ), "param 'image_size' must be list or tuple"
        if image_size:
            assert (
                len(image_size) == 2
            ), "param 'image_size' must be 2 in length"
            for v in image_size:
                assert isinstance(
                    v, Integral
                ), "members of 'image_size' must be int type"
        assert isinstance(
            pre_decode_top_n, (Integral, type(None))
        ), "param 'pre_decode_top_n' must be int type"
        assert isinstance(
            post_decode_top_n, (Integral, type(None))
        ), "param 'post_decode_top_n' must be int type"
        assert isinstance(
            iou_threshold, (Real, type(None))
        ), "param 'iou_threshold' must be real type"
        assert isinstance(
            pre_nms_top_n, (Integral, type(None))
        ), "param 'pre_nms_top_n' must be int type"
        assert isinstance(
            post_nms_top_n, (Integral, type(None))
        ), "param 'post_nms_top_n' must be int type"
        assert isinstance(
            nms_on_each_level, (bool, type(None))
        ), "param 'nms_on_each_level' must be bool type"
        assert mode in (
            "normal",
            "yolo",
        ), "Only support mode in ('normal', 'yolo')"

        self.regression_scale = regression_scale
        self.background_class_idx = background_class_idx
        self.score_threshold = score_threshold
        self.size_threshold = size_threshold
        self.iou_threshold = iou_threshold
        self.pre_decode_top_n = pre_decode_top_n
        self.post_decode_top_n = post_decode_top_n
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.image_size = image_size
        self.nms_on_each_level = nms_on_each_level
        self.mode = mode

    @script_quantized_fn
    def forward(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        regressions: List[torch.Tensor],
        image_shapes: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not type(boxes) in (tuple, list):
            boxes = [boxes]
            scores = [scores]
            regressions = [regressions]

        # Manually unbind tensor to avoid trace warnings.
        boxes = [box.unbind() for box in boxes]
        scores = [score.unbind() for score in scores]
        regressions = [regression.unbind() for regression in regressions]

        # Transpose data from list along levels to list along batch.
        boxes = map(list, zip(*boxes))
        scores = map(list, zip(*scores))
        regressions = map(list, zip(*regressions))

        ret_boxes, ret_scores, ret_labels = [], [], []

        for (
            batch_idx,
            (per_image_boxes, per_image_scores, per_image_regressions),
        ) in enumerate(zip(boxes, scores, regressions)):
            all_level_boxes, all_level_scores, all_level_labels = [], [], []

            for (
                per_level_boxes,
                per_level_scores,
                per_level_regressions,
            ) in zip(per_image_boxes, per_image_scores, per_image_regressions):
                (
                    _,
                    per_level_scores,
                    per_level_boxes,
                    per_level_regressions,
                ) = filter_by_max_score(
                    per_level_scores,
                    per_level_boxes,
                    per_level_regressions,
                    self.score_threshold,
                    (0, 1) if self.mode == "yolo" else None,
                )

                if self.pre_decode_top_n:
                    (
                        per_level_scores,
                        per_level_boxes,
                        per_level_regressions,
                    ) = get_top_n(
                        per_level_scores,
                        [per_level_boxes, per_level_regressions],
                        self.pre_decode_top_n,
                        0,
                    )

                per_level_boxes, per_level_scores, per_level_labels = decode(
                    per_level_boxes,
                    per_level_regressions,
                    per_level_scores,
                    self.regression_scale,
                    self.background_class_idx,
                    (
                        None
                        if self.image_size is None
                        else torch.tensor(self.image_size, dtype=torch.int)
                    )
                    if image_shapes is None
                    else image_shapes[batch_idx],
                    self.size_threshold,
                    self.mode == "yolo",
                )

                if self.post_decode_top_n:
                    (
                        per_level_scores,
                        per_level_boxes,
                        per_level_labels,
                    ) = get_top_n(
                        per_level_scores,
                        [per_level_boxes, per_level_labels],
                        self.post_decode_top_n,
                        None,
                    )

                if self.nms_on_each_level and (self.iou_threshold is not None):
                    per_level_boxes, per_level_scores, per_level_labels = nms(
                        per_level_boxes,
                        per_level_scores,
                        per_level_labels,
                        self.iou_threshold,
                        None if self.mode == "yolo" else self.score_threshold,
                        self.pre_nms_top_n,
                        self.post_nms_top_n,
                    )

                all_level_boxes.append(per_level_boxes)
                all_level_scores.append(per_level_scores)
                all_level_labels.append(per_level_labels)

            pred_per_image_boxes = torch.cat(all_level_boxes, dim=0)
            pred_per_image_scores = torch.cat(all_level_scores, dim=0)
            pred_per_image_labels = torch.cat(all_level_labels, dim=0)

            if (not self.nms_on_each_level) and (
                self.iou_threshold is not None
            ):
                (
                    pred_per_image_boxes,
                    pred_per_image_scores,
                    pred_per_image_labels,
                ) = nms(
                    pred_per_image_boxes,
                    pred_per_image_scores,
                    pred_per_image_labels,
                    self.iou_threshold,
                    None if self.mode == "yolo" else self.score_threshold,
                    self.pre_nms_top_n,
                    self.post_nms_top_n,
                )

            ret_boxes.append(pred_per_image_boxes)
            ret_scores.append(pred_per_image_scores)
            ret_labels.append(pred_per_image_labels)

        return tuple(ret_boxes), tuple(ret_scores), tuple(ret_labels)
