# Copyright (c) Changan Auto. All rights reserved.
# Source code reference to mmdetection

from typing import Any, Dict, Sequence

import torch
from torchvision.ops.boxes import batched_nms

from cap.models.base_modules.postprocess import PostProcessorBase
from cap.registry import OBJECT_REGISTRY
from .target import distance2bbox, get_points

__all__ = ["FCOSDecoder", "multiclass_nms"]


@OBJECT_REGISTRY.register
class FCOSDecoder(PostProcessorBase):  # noqa: D205,D400
    """

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        strides (Sequence[int]): A list contains the strides of fcos_head
            output.
        transforms (Sequence[dict]): A list contains the transform config.
        inverse_transform_key (Sequence[str]): A list contains the inverse
            transform info key.
        nms_use_centerness (bool, optional): If True, use centerness as a
            factor in nms post-processing.
        nms_sqrt (bool, optional): If True, sqrt(score_thr * score_factors).
        test_cfg (dict, optional): Cfg dict, including some configurations of
            nms.
        truncate_bbox (bool, optional): If True, truncate the predictive bbox
            out of image boundary. Default True.
        filter_score_mul_centerness (bool, optional): If True, filter out bbox
            by score multiply centerness, else filter out bbox by score.
            Default False.
    """

    def __init__(
        self,
        num_classes,
        strides,
        transforms=None,
        inverse_transform_key=None,
        nms_use_centerness=True,
        nms_sqrt=True,
        test_cfg=None,
        input_resize_scale=None,
        truncate_bbox=True,
        filter_score_mul_centerness=False,
    ):
        super(FCOSDecoder, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.transforms = transforms
        self.inverse_transform_key = inverse_transform_key
        self.nms_use_centerness = nms_use_centerness
        self.nms_sqrt = nms_sqrt
        self.test_cfg = test_cfg
        self.input_resize_scale = input_resize_scale
        if self.input_resize_scale is not None:
            assert self.input_resize_scale > 0
        self.truncate_bbox = truncate_bbox
        self.filter_score_mul_centerness = filter_score_mul_centerness

    def forward(self, pred: Sequence[torch.Tensor], meta_data: Dict[str, Any]):
        assert len(pred) == 3, (
            "pred must be a tuple containing cls_scores,"
            "bbox_preds, centernesses"
        )
        cls_scores, bbox_preds, centernesses = pred
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        results = {}
        det_results = []
        for img_id in range(bbox_preds[0].shape[0]):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]  # [cxh1xw1, cxh2xw2, cxh3xw3, ...]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]  # [4xh1xw1, 4xh2xw2, 4xh3xw3, ...]
            if centernesses is not None:
                centerness_pred_list = [
                    centernesses[i][img_id].detach() for i in range(num_levels)
                ]  # [1xh1xw1, 1xh2xw2, 1xh3xw3, ...]
            else:
                centerness_pred_list = [None] * len(cls_score_list)

            h_index = meta_data["layout"][img_id].index("h")
            w_index = meta_data["layout"][img_id].index("w")
            if "pad_shape" in meta_data:
                max_shape = (
                    meta_data["pad_shape"][img_id][h_index],
                    meta_data["pad_shape"][img_id][w_index],
                )
            elif "img_shape" in meta_data:
                max_shape = (
                    meta_data["img_shape"][img_id][h_index],
                    meta_data["img_shape"][img_id][w_index],
                )
            else:
                max_shape = (
                    meta_data["img_height"][img_id],
                    meta_data["img_width"][img_id],
                )
            max_shape = max_shape if self.truncate_bbox else None
            inverse_info = {}
            for key, value in meta_data.items():
                if (
                    self.inverse_transform_key
                    and key in self.inverse_transform_key
                ):
                    inverse_info[key] = value[img_id]
            det_bboxes = self._decode_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                mlvl_points,
                max_shape,
                inverse_info,
            )
            det_results.append(det_bboxes)

        results["pred_bboxes"] = det_results
        results["img_name"] = meta_data["img_name"]
        results["img_id"] = meta_data["img_id"]
        return results

    def _decode_single(
        self,
        cls_score_list,
        bbox_pred_list,
        centerness_pred_list,
        mlvl_points,
        max_shape,
        inverse_info,
    ):
        """Decode the output of a single picture into a prediction result.

        Args:
            cls_score_list (list[torch.Tensor]): List of all levels' cls_score,
                each has shape (N, num_points * num_classes, H, W).
            bbox_pred_list (list[torch.Tensor]): List of all levels' bbox_pred,
                each has shape (N, num_points * 4, H, W).
            centerness_pred_list (list[torch.Tensor]): List of all levels'
                centerness_pred, each has shape (N, num_points * 1, H, W).
            mlvl_points (list[torch.Tensor]): List of all levels' points.
            max_shape (Sequence): Maximum allowable shape of the decoded bbox.

        Returns:
            det_bboxes (torch.Tensor): Decoded bbox, with shape (N, 6),
                represents x1, y1, x2, y2, cls_score, cls_id (0-based).
        """
        cfg = self.test_cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        has_centerness = centerness_pred_list[0] is not None
        if has_centerness:
            mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
            cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_points
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = (
                cls_score.permute(1, 2, 0)
                .reshape(-1, self.num_classes)
                .sigmoid()
            )
            if has_centerness:
                centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if has_centerness:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                else:
                    max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                if centerness is not None:
                    centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=max_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if has_centerness:
                mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if self.input_resize_scale is not None:
            mlvl_bboxes[:, :5] /= self.input_resize_scale
        # inverse transform for mapping to the original image
        if self.transforms:
            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    mlvl_bboxes = transform.inverse_transform(
                        inputs=mlvl_bboxes,
                        task_type="detection",
                        inverse_info=inverse_info,
                    )
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # FG labels to [0, num_class-1], BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if has_centerness:
            mlvl_centerness = torch.cat(mlvl_centerness)
        else:
            mlvl_centerness = None  # score_factors wile be set as None
        score_thr = cfg.get("score_thr", "0.05")
        nms = cfg.get("nms").get("name", "nms")
        iou_threshold = cfg.get("nms").get("iou_threshold", 0.6)
        max_per_img = cfg.get("max_per_img", 100)
        det_bboxes = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr,
            nms,
            iou_threshold,
            max_per_img,
            score_factors=mlvl_centerness if self.nms_use_centerness else None,
            nms_sqrt=self.nms_sqrt,
            filter_score_mul_centerness=self.filter_score_mul_centerness,
        )
        return det_bboxes


def multiclass_nms(
    multi_bboxes,
    multi_scores,
    score_thr,
    nms,
    iou_threshold,
    max_per_img=-1,
    score_factors=None,
    nms_sqrt=False,
    filter_score_mul_centerness=False,
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (str): nms type, candidate values are ['nms', 'soft_nms'].
        iou_threshold (float): NMS IoU threshold
        max_per_img (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        nms_sqrt (bool): If True, sqrt(score_thr * score_factors)
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4
        )  # NC4
    scores = multi_scores[:, :-1]  # NC
    if score_factors is not None:
        if not nms_sqrt:
            scores_new = scores * score_factors[:, None]
        else:
            scores_new = torch.sqrt(scores * score_factors[:, None])
    # filter out boxes with low scores
    if filter_score_mul_centerness and score_factors is not None:
        valid_mask = scores_new > score_thr
        scores = scores_new
    elif score_factors is not None:
        valid_mask = scores > score_thr
        scores = scores_new
    else:
        valid_mask = scores > score_thr

    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1),
    ).view(-1, 4)
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, 1), dtype=torch.float32)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )

        return torch.cat((bboxes, labels), dim=-1)

    if nms == "nms":
        keep = batched_nms(bboxes, scores, labels, iou_threshold)
    else:
        raise NotImplementedError

    dets = torch.cat([bboxes[keep], scores[keep][:, None]], -1)

    if max_per_img > 0:
        dets = dets[:max_per_img]
        keep = keep[:max_per_img]

    return torch.cat((dets, labels[keep].view(-1, 1).float()), dim=-1)
