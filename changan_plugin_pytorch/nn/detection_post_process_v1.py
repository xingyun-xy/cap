from numbers import Integral, Real
from typing import Optional

import torch
from changan_plugin_pytorch.utils import fx_helper
from changan_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn, )
from torch import Tensor, nn
from torch.jit.annotations import List, Tuple
from torchvision.models.detection._utils import BoxCoder
import numpy as np


@fx_helper.wrap
class DetectionPostProcessV1(nn.Module):
    """
    Post process for object detection models. Only supported on bernoulli2.
    This operation is implemented on BPU, thus is expected to be faster
    than cpu implementation.
    This operation requires input_scale = 1 / 2 ** 4, or a rescale will be
    applied to the input data. So you can manually set the output scale
    of previous op (Conv2d for example) to 1 / 2 ** 4 to avoid the rescale
    and get best performance and accuracy.

    Major differences with DetectionPostProcess:
        1. Each anchor will generate only one pred bbox totally, but in
        DetectionPostProcess each anchor will generate one bbox for each
        class (num_classes bboxes totally).
        2. NMS has a margin param, box2 will only be supressed by box1 when
        box1.score - box2.score > margin (box1.score > box2.score in
        DetectionPostProcess).
        3. A offset can be added to the output class indices (
        using class_offsets).

    Args:
        num_classes (int): Class number.
        box_filter_threshold (float):
            Default threshold to filter box by max score.
        class_offsets (List[int]): Offset to be added to output class index
            for each branch.
        use_clippings (List[bool]): Whether clip box to image size.
            If input is padded, you can clip box to real content by providing
            image size.
        image_size (Tuple[int, int]): Fixed image size in (h, w), set to None
            if input have different sizes.
        nms_threshold (float): IoU threshold for nms.
        nms_margin (float): Only supress box2 when
            box1.score - box2.score > nms_margin
        pre_nms_top_k: Maximum number of bounding boxes in each image before
            nms.
        post_nms_top_k: Maximum number of output bounding boxes in each
            image.
        nms_padding_mode: The way to pad bbox to match the number of output
            bounding bouxes to post_nms_top_k, can be None, "pad_zero" or
            "rollover".
        bbox_min_hw: Minimum height and width of selected bounding boxes.
    """

    def __init__(
            self,
            num_classes: int,
            box_filter_threshold: float,
            class_offsets: List[int],
            use_clippings: bool,
            image_size: Tuple[int, int],
            nms_threshold: float,
            pre_nms_top_k: int,
            post_nms_top_k: int,
            nms_padding_mode: Optional[str] = None,
            nms_margin: float = 0.0,
            bbox_min_hw: Tuple[float, float] = (0, 0),
    ):
        super(DetectionPostProcessV1, self).__init__()
        assert isinstance(num_classes, Integral), "num_classes must be int"
        assert isinstance(box_filter_threshold,
                          Real), "box_filter_threshold must be real number"
        assert isinstance(
            class_offsets,
            (list, tuple)), "class_offsets must be list or tuple of int"
        for class_offset in class_offsets:
            assert isinstance(
                class_offset,
                Integral), "class_offsets must be list or tuple of int"
        assert isinstance(use_clippings, bool), "use_clippings must be bool"
        assert (isinstance(image_size, (list, tuple)) and len(image_size) == 2
                and isinstance(image_size[0], Integral) and isinstance(
                    image_size[1],
                    Integral)), "image_size must be list or tuple of two int"
        assert isinstance(nms_threshold,
                          Real), "nms_threshold must be real number"
        assert isinstance(pre_nms_top_k, Integral), "pre_nms_top_k must be int"
        assert isinstance(post_nms_top_k,
                          Integral), "post_nms_top_k must be int"
        assert nms_padding_mode in (
            None,
            "pad_zero",
            "rollover",
        ), "only support nms_padding_mode in 'pad_zero' and 'rollover'"
        assert isinstance(nms_margin, Real), "nms_margin must be real number"
        assert (nms_margin == 0.0
                ), "We do not support non zero value for nms_margin"
        assert (
            isinstance(bbox_min_hw, (list, tuple)) and len(bbox_min_hw) == 2
            and isinstance(bbox_min_hw[0], Real)
            and isinstance(bbox_min_hw[1], Real) and bbox_min_hw[0] >= 0
            and bbox_min_hw[1] >= 0
        ), "bbox_min_hw must be list or tuple of two nonnegative number"

        self.num_classes = num_classes
        self.box_filter_threshold = box_filter_threshold
        self.class_offsets = class_offsets
        self.use_clippings = use_clippings
        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.pre_nms_top_k = pre_nms_top_k
        self.post_nms_top_k = post_nms_top_k
        self.nms_padding_mode = nms_padding_mode
        self.nms_margin = nms_margin
        self.bbox_min_hw = bbox_min_hw

        self.coder = BoxCoder((1, 1, 1, 1))
        self.register_buffer("_image_size",
                             Tensor(image_size).view(1, 2),
                             persistent=False)

    @script_quantized_fn
    def forward(self,
                data: List[Tensor],
                anchors: List[Tensor],
                image_sizes=None) -> Tensor:
        """
        Forward pass of ~DetectionPostProcessV1

        Args:
            data (List[Tensor]): (N, (4 + num_classes) * anchor_num, H, W)
            anchors (List[Tensor]): (N, anchor_num * 4, H, W)
            image_sizes (Tensor[batch_size, (h, w)], optional):
                Defaults to None.

        Returns:
            List[
                Tuple[
                    Tensor[output_bbox_num, 4],
                    Tensor[output_bbox_num],
                    Tensor[output_bbox_num],
                ]
            ]:
                bbox (x1, y1, x2, y2), score, class_idx
        """
        if self.use_clippings:
            assert (self.image_size is not None or image_sizes is not None
                    ), "image size must be provided if use_clippings == True"
        branch_num = len(data)
        batch_size = data[0].size(0)

        per_image_rets_list = []

        # rearrange anchor and data
        flatten_anchors, deltas, scores = [], [], []

        # TODO: check if class offsets should work in batch-wise instead
        # of branch-wise
        cls_offsets = []

        for i in range(branch_num):
            flatten_anchors.append(anchors[i].detach().permute(
                0, 2, 3, 1).reshape(batch_size, -1, 4))
            unfolded_data = (data[i].detach().permute(0, 2, 3, 1).reshape(
                batch_size, -1, 4 + self.num_classes))

            deltas.append(unfolded_data[..., :4])
            scores.append(unfolded_data[..., 4:])

            cls_offsets.append(
                unfolded_data.new_ones(
                    (1, unfolded_data.shape[1])) * self.class_offsets[i])

        # (B, N, C)
        flatten_anchors = torch.cat(flatten_anchors, dim=1)
        deltas = torch.cat(deltas, dim=1)
        scores = torch.cat(scores, dim=1)

        # (1, N)
        cls_offsets = torch.cat(cls_offsets, dim=1)

        # (B, N)
        max_scores, max_score_idxs = torch.max(scores, dim=-1)

        # (B, N)
        cls_idxs = max_score_idxs + cls_offsets

        # decode, (B, N, 4)
        pred_boxes = self.coder.decode_single(
            deltas.view(-1, 4),
            flatten_anchors.view(-1, 4),
        ).view(batch_size, -1, 4)

        im_hw = image_sizes if image_sizes is not None else self._image_size
        pred_boxes = self.clip_boxes_to_image(pred_boxes, im_hw)

        combine_boxes = torch.cat(
            [pred_boxes, cls_idxs[..., None], max_scores[..., None]], dim=2)

        sorted_boxes = self._sort(combine_boxes)

        for single_boxes in sorted_boxes:

            # filter by score threshold
            score_mask = single_boxes[:, -1] >= self.box_filter_threshold

            # filter invalid
            valid_mask = torch.logical_and(
                single_boxes[:, 0] < single_boxes[:, 2] - self.bbox_min_hw[1],
                single_boxes[:, 1] < single_boxes[:, 3] - self.bbox_min_hw[0],
            )
            if not torch.onnx.is_in_onnx_export():
                single_boxes = single_boxes[torch.logical_and(
                    score_mask, valid_mask)]
            else:
                # ==================substitute2 start=========================
                device = valid_mask.device
                logical_and_mask = torch.logical_and(score_mask, valid_mask)
                all_zeros = torch.zeros_like(logical_and_mask,
                                             dtype=torch.long,
                                             device=device)
                index_t = torch.tensor(range(0, len(logical_and_mask)),
                                       dtype=torch.long,
                                       device=device)
                index_with_zero = torch.where(logical_and_mask, index_t,
                                              all_zeros)
                single_boxes = single_boxes[index_with_zero]
                # ==================substitute2 end=========================

            # nms
            single_boxes = self._nms(single_boxes)

            if not torch.onnx.is_in_onnx_export():
                # filter
                if single_boxes.shape[0] > self.post_nms_top_k:
                    single_boxes = single_boxes[:self.post_nms_top_k]

                # pad
                else:
                    if self.nms_padding_mode is not None:
                        single_boxes = self.pad_data(single_boxes)

            else:
                zero_boxes = single_boxes.new_zeros(
                    (self.post_nms_top_k, *single_boxes.shape[1:]))
                single_boxes = torch.cat([single_boxes, zero_boxes])
                single_boxes = single_boxes[:self.post_nms_top_k]

            pred_box = single_boxes[:, :4]
            cls_idxs = single_boxes[:, -2]
            max_scores = single_boxes[:, -1]

            per_image_rets_list.append((pred_box, max_scores, cls_idxs))

        return per_image_rets_list

    def pad_data(self, data: Tensor):
        if data.numel() == 0:
            return data.new_zeros((self.post_nms_top_k, *data.shape[1:]))
        else:
            if self.nms_padding_mode == "rollover":
                data = torch.cat([data] * (self.post_nms_top_k // len(data)))
                num_padded = self.post_nms_top_k - data.shape[0]
                return torch.cat([data, data[:num_padded]])
            else:
                num_padded = self.post_nms_top_k - data.shape[0]
                return torch.cat(
                    [data, data.new_zeros((num_padded, *data.shape[1:]))])
                # return data.new_zeros((self.post_nms_top_k, *data.shape[1:]))

    def clip_boxes_to_image(self, boxes, im_hw):
        if self.use_clippings:
            dim = boxes.dim()
            boxes_x = boxes[..., 0::2]
            boxes_y = boxes[..., 1::2]

            x_max = im_hw[:, 1][:, None, None]
            y_max = im_hw[:, 0][:, None, None]
            min_val = im_hw[:, 0][:, None, None] * 0
            boxes_x = boxes_x.clamp(min=min_val, max=x_max)
            boxes_y = boxes_y.clamp(min=min_val, max=y_max)

            clip_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
            boxes = clip_boxes.reshape_as(boxes)

        return boxes

    def _take_row(self, data, index):
        arr = torch.arange(data.shape[0], device=data.device)[:, None]
        flatten_index = (index + arr * data.shape[1]).flatten()

        last_dims = data.shape[2:]
        flatten_target = data.view(-1, *last_dims)

        indexed = flatten_target[flatten_index.long()].view(
            data.shape[0], -1, *last_dims)
        return indexed

    def _sort(self, combine_boxes: Tensor) -> Tensor:
        # sorted_idxs = combine_boxes[..., -1].argsort(descending=True, dim=-1)[
        #     ..., : self.pre_nms_top_k
        # ]
        sorted_idxs = combine_boxes[..., -1].sort(
            descending=True, dim=-1)[1][..., :self.pre_nms_top_k]

        return self._take_row(combine_boxes, sorted_idxs)

    def _nms(self, pred_boxes: Tensor) -> Tensor:
        from .functional import batched_nms

        boxes = pred_boxes[:, :4]
        idxs = pred_boxes[:, -2]
        scores = pred_boxes[:, -1]

        keep = batched_nms(boxes, scores, idxs, self.nms_threshold)
        return pred_boxes[keep]
