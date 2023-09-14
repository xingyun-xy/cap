from typing import Union

import torch
from changan_plugin_pytorch.functional import batched_nms
from changan_plugin_pytorch.functional import sort as stable_sort
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.script_helper import arange
from torch import Tensor
from torch.jit.annotations import List, Optional, Tuple

__all__ = [
    "filter",
    "channel_shuffle",
    "point_pillars_scatter",
    "window_partition",
    "window_reverse",
    "rle",
    "point_pillars_preprocess",
]


def filter_by_max_score(
    scores: Tensor,
    boxes: Tensor,
    regressions: Tensor,
    score_threshold: float,
    score_index_range: Optional[Tuple[int, int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Filter anchors by max score.
    The input will be transformed to (H * W * anchor_num, ...) and filtered.

    Args:
        boxes (Tensor(anchor_num * 4, H, W)): Boxes.
        scores (Tensor(anchor_num * num_classes, H, W)): Scores.
        regressions (Tensor(anchor_num * num_classes * 4 or anchor_num * 4,
            H, W)): Box regressions.
        score_threshold (float): Score threshold.
        score_index_range (Optional[Tuple[int, int]]):
            Specify the score index range counted for compare.

    Returns:
        Tensor(anchor_num, H, W): Mask.
        Tensor(M, 4): Boxes.
        Tensor(M, num_classes): Scores.
        Tensor(M, num_classes * 4 or 4): Regressions.
    """
    anchor_num = int(boxes.size(0) / 4)
    height = scores.size(1)
    width = scores.size(2)
    total_anchor_num = anchor_num * height * width

    scores_per_anchor = scores.reshape(anchor_num, -1, height, width)
    if score_index_range is None:
        max_scores, _ = scores_per_anchor.max(dim=1, keepdim=False)
    else:
        max_scores, _ = scores_per_anchor[
            :, score_index_range[0] : score_index_range[1], :, :
        ].max(dim=1, keepdim=False)
    mask = max_scores > score_threshold

    scores = scores.permute(1, 2, 0).reshape(total_anchor_num, -1)
    boxes = boxes.permute(1, 2, 0).reshape(total_anchor_num, 4)
    regressions = regressions.permute(1, 2, 0).reshape(total_anchor_num, -1)
    keep = mask.permute(1, 2, 0).reshape(total_anchor_num)

    return (mask, scores[keep], boxes[keep], regressions[keep])


def get_top_n(
    scores: Tensor, others: List[Tensor], n: int, score_index: Optional[int]
) -> List[Tensor]:
    """
    Get top n by score.

    Args:
        scores (Tensor(N, num_classes or None)): Scores.
        others (List[Tensor(N, ...)]): Other data.
        n (int): Output example number.
        score_index_range (Optional[Tuple[int, int]]):
            Specify class index to get top n if multi score is provided.

    Returns:
        List[Tensor]: Filtered scores and others.
    """
    if scores.ndim == 1 or scores.size(1) == 1:
        scores_for_top_n = scores.squeeze()
    else:
        assert (
            score_index is not None
        ), "Please spacify a index when multi scores are provided"
        scores_for_top_n = scores[:, score_index].squeeze()

    per_image_keep = stable_sort(
        scores_for_top_n, descending=True, stable=True
    )[1][: min(n, scores_for_top_n.size(0))]

    return [scores[per_image_keep]] + [data[per_image_keep] for data in others]


def decode(
    boxes: Tensor,
    regressions: Tensor,
    scores: Tensor,
    regression_scale: Optional[Tuple[float, float, float, float]],
    background_class_idx: Optional[int],
    clip_size: Optional[Tensor],
    size_threshold: Optional[float],
    abs_offset: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Generate labels.
    Remove background.
    Decode box.
    Clip box to image size.
    Remove too small boxes.

    Args:
        boxes (Tensor(N, 4)):
            Boxes in (x1, y1, x2, y2) format.
        regressions (Tensor(N, num_classes * 4 or 4)):
            Regressions in (dx, dy, dw, dh) format.
        scores (Tensor(N, num_classes)): Scores.
        labels (Tensor(N, num_classes)): Labels.
        regression_scale (Optional[Tuple[float, float, float, float]]):
            Scales to be multiplyed to regressions.
        background_class_idx (Optional[int]):
            Specify one class to be ignored.
        size_threshold (Optional[float]):
            Remove boxes whose height or width smaller than this.
        clip_size (Optional[Tuple[int, int]]):
            Clip boxes according to this image size.
        abs_offset (bool, optional):
            Whether treat dx and dy in regressions as absolute offset.
            Defaults to False.

    Returns:
        Tensor(N * num_classes, 4): Decoded boxes.
        Tensor(N * num_classes): Scores.
        Tensor(N * num_classes): Labels.
    """
    regression_num_per_box = regressions.size(1) // boxes.size(1)
    num_classes = scores.size(1)
    # Multiply class scores with objectness.
    if abs_offset:
        scores = scores[..., :1] * scores[..., 1:]

    # Generate labels.
    labels = (
        torch.arange(scores.size(1), device=scores.device)
        .unsqueeze(0)
        .expand_as(scores)
    )

    # Remove background.
    if background_class_idx is not None:
        if regressions.size(-1) > 4:
            regressions = torch.cat(
                [
                    regressions[..., : background_class_idx * 4],
                    regressions[..., background_class_idx * 4 + 4 :],
                ],
                dim=-1,
            )
        scores = torch.cat(
            [
                scores[..., :background_class_idx],
                scores[..., background_class_idx + 1 :],
            ],
            dim=-1,
        )
        labels = torch.cat(
            [
                labels[..., :background_class_idx],
                labels[..., background_class_idx + 1 :],
            ],
            dim=-1,
        )

    # Decode.
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = regressions[:, 0::4]
    dy = regressions[:, 1::4]
    dw = regressions[:, 2::4]
    dh = regressions[:, 3::4]

    if regression_scale is not None:
        dx = dx * regression_scale[0]
        dy = dy * regression_scale[1]
        dw = dw * regression_scale[2]
        dh = dh * regression_scale[3]

    if abs_offset:
        pred_ctr_x = dx + ctr_x[:, None]
        pred_ctr_y = dy + ctr_y[:, None]
    else:
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h
    boxes = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=-1)

    # Reshape.
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(
        boxes.size(0), num_classes if regression_num_per_box == 1 else 1
    )
    labels = labels.reshape(scores.shape)

    # Clip out of bound boxes.
    if clip_size is not None:
        boxes.select(-1, 0).clamp_(0, clip_size[1].item())
        boxes.select(-1, 1).clamp_(0, clip_size[0].item())
        boxes.select(-1, 2).clamp_(0, clip_size[1].item())
        boxes.select(-1, 3).clamp_(0, clip_size[0].item())

    # Filter too small boxes.
    if size_threshold is not None and size_threshold > 0:
        keep = (pred_w >= size_threshold) & (pred_h >= size_threshold)
        keep = keep.reshape(-1)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    # Expand predicted boxes if there is only
    # one set of regression for all classes.
    expand_times = scores.size(-1)
    boxes = (
        boxes.unsqueeze(1)
        .expand(boxes.size(0), expand_times, 4)
        .reshape(-1, 4)
    )
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    return (boxes, scores, labels)


def nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float,
    score_threshold: Optional[float],
    pre_nms_top_n: Optional[int],
    output_num: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Filter by score.
    Pre nms top k.
    Nms within boxes of the same label.
    Post nms top k.

    Args:
        boxes (Tensor(N, 4)): Boxes in (x1, y1, x2, y2) format.
        scores (Tensor(N)): Scores.
        labels (Tensor(N)): Labels.
        iou_threshold (float): IoU threshold.
        score_threshold (Optional[float]):
            Filter boxes whose score smaller than this before nms.
        pre_nms_top_n (Optional[int]): Get top n boxes by score before nms.
        output_num (Optional[int]): Get top n boxes by score after nms.

    Returns:
        Tensor(M, 4): Boxes.
        Tensor(M): Scores.
        Tensor(M): Labels.
    """
    if scores.numel() == 0:
        return (boxes, scores, labels)

    if score_threshold is not None:
        score_keep = scores > score_threshold
        boxes = boxes[score_keep]
        scores = scores[score_keep]
        labels = labels[score_keep]

    if pre_nms_top_n is not None:
        idx = stable_sort(scores, descending=True, stable=True)[1][
            : min(pre_nms_top_n, scores.size(0))
        ]
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

    nms_keep = batched_nms(boxes, scores, labels, iou_threshold)

    boxes = boxes[nms_keep]
    scores = scores[nms_keep]
    labels = labels[nms_keep]

    if output_num is not None:
        boxes = boxes[:output_num]
        scores = scores[:output_num]
        labels = labels[:output_num]

    return (boxes, scores, labels)


def anchor_generator(
    mlvl_base_anchors: List[torch.Tensor],
    strides: List[int],
    feature_maps: List[torch.Tensor],
) -> List[torch.Tensor]:
    mlvl_anchors = []

    for base_anchors, stride, feat in zip(
        mlvl_base_anchors, strides, feature_maps
    ):
        batch_size, _, height, width = feat.shape

        offset_x = arange(0, width * stride, stride, device_like=base_anchors)
        offset_y = arange(0, height * stride, stride, device_like=base_anchors)
        offset_y, offset_x = torch.meshgrid(offset_y, offset_x)
        offsets = torch.stack(
            [
                offset_x.ravel(),
                offset_y.ravel(),
                offset_x.ravel(),
                offset_y.ravel(),
            ],
            dim=0,
        )[None]

        # broadcast_add (n_anchor, 4, 1) + (1, 4, h*w) -> (n_anchor, 4, h*w)
        anchors = base_anchors[..., None] + offsets

        # anchors: (batch_size, n_anchor * 4, height, width)
        mlvl_anchors.append(
            anchors.view(1, -1, height, width).repeat(batch_size, 1, 1, 1)
        )

    return mlvl_anchors


def filter(
    *inputs: Union[Tuple[Tensor], Tuple[QTensor]],
    threshold: float,
    idx_range: Optional[Tuple[int, int]] = None,
) -> List[List[Tensor]]:
    """
    The output order is different with bpu, because that the compiler do
    some optimization and slice input following complex rules, which is
    hard to be done by plugin.

    All inputs are filtered along HW by the max value within a range in
    channel dim of the first input.
    Each NCHW input is splited, transposed and flattened to
    List[Tensor[H * W, C]] first.
    If input is QTensor, the output will be dequantized.

    Args:
        inputs (Union[Tuple[Tensor], Tuple[QTensor]]): Data in NCHW format.
            Each input shold have the same size in N, H, W.
            The output will be selected according to the first input.
        threshold (float): Threshold, the lower bound of output.
        idx_range (Optional[Tuple[int, int]], optional): The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.

    Returns:
        Union[List[List[Tensor]], List[List[QTensor]]]:
        A list with same length of batch size, and each element contains:
            max_value: Flattened max value within idx_range in channel dim.
            max_idx: Flattened max value index in channel dim.
            coord: The original coordinates of the output data in the
                input data in the shape of [M, (h, w)].
            (multi) data: Filtered data in the shape of [M, C].
    """
    from changan_plugin_pytorch.nn.quantized.functional import (
        filter as quantized_filter,
    )

    if isinstance(inputs[0], QTensor):
        return quantized_filter(
            [input.as_subclass(torch.Tensor) for input in inputs],
            [input.q_scale() for input in inputs],
            [input.q_zero_point() for input in inputs],
            [input.dtype for input in inputs],
            threshold,
            idx_range or (0, inputs[0].size(1)),
        )

    else:
        return quantized_filter(
            inputs,
            [torch.empty(0)] * len(inputs),
            [torch.empty(0)] * len(inputs),
            [str(input.dtype) for input in inputs],
            threshold,
            idx_range or (0, inputs[0].size(1)),
        )


def bgr_to_yuv444(input: Tensor, channel_reversal: bool) -> Tensor:
    """
    Convert image color format from bgr to yuv444.

    Args:
        input (Tensor): Input tensor with shape [N, C, H, W].
        channel_reversal (bool): Color channel order,
            set to True when used on RGB input.

    Returns:
        Tensor: Output tensor with shape [N, C, H, W].
    """
    return torch.ops.changan.bgr_to_yuv444(input, channel_reversal)


def max_iou_match(
    boxes: Tensor,
    gt_boxes: Tensor,
    gt_boxes_num: Tensor,
    pos_iou: float,
    neg_iou: float,
    allow_low_quality_match: bool,
    low_quality_match_iou: float,
    legacy_bbox: bool,
    overlap_type: str,
):
    """
    Match boxes to gt_boxes according to corresponding overlap type and
    threshold.

    Args:
        boxes (Tensor): Input boxes with shape [N, 4] or [B, N, 4] (don't
            need to repeat if boxes in each sample are identical)
        gt_boxes (Tensor): Input gt boxes with shape [B, M, 5+], for each
            gt box, first 4 values are the coordinates, and the fifth the
            class label.
        gt_boxes_num (Tensor): Input gt boxes num with shape [B], each
            value represents the number of gt boxes in each sample (to
            let padded boxes be ignored in computation)
        pos_iou (float): (lower) threshold for positive matching
        neg_iou (float): (upper) threshold for positive matching
        allow_low_quality_match (bool): whether low quality match is allowed
        low_quality_match_iou (float): iou threshold for low quality match
        legacy_bbox (bool): whether the gt boxes are encoded in legacy manner
        overlap_type (str): overlap type, iou or ioa

    Returns:
        flags: Output flag with shape [B, N].
        matched_gt_id: Output gt id with shape [B, N].
    """
    return torch.ops.changan.max_iou_match(
        boxes,
        gt_boxes,
        gt_boxes_num,
        pos_iou,
        neg_iou,
        allow_low_quality_match,
        low_quality_match_iou,
        legacy_bbox,
        overlap_type,
    )


def ig_region_match(
    boxes: Tensor,
    ig_boxes: Tensor,
    ig_boxes_num: Tensor,
    class_num: int,
    ig_region_overlap: float,
    legacy_bbox: bool,
    exclude_background: bool,
):
    """
    Match boxes to ig regions according to corresponding overlap threshold.

    Args:
        boxes (Tensor): Input boxes with shape [N, 4] or [B, N, 4] (don't
            need to repeat if boxes in each sample are identical)
        ig_regions (Tensor): Input ig regions with shape [B, M, 5+], for each
            ig region, first 4 values are the coordinates, and the fifth the
            class label.
        ig_regions_num (Tensor): Input ig regions num with shape [B], each
            value represents the number of ig regions in each sample (to
            let padded boxes be ignored in computation)
        class_num (int): number of classes
        ig_region_overlap (float): ioa (lower) threshold for matching
        legacy_bbox (bool): whether the gt boxes are encoded in legacy manner
        exclude_background (bool): whether to ignore background class in
            output flags

    Returns:
        Output flag with shape [B, N, C - 1] if exclude_background else
            [B, N, C]
    """
    return torch.ops.changan.ig_region_match(
        boxes,
        ig_boxes,
        ig_boxes_num,
        class_num,
        ig_region_overlap,
        legacy_bbox,
        exclude_background,
    )


def channel_shuffle(input: Tensor, groups: int):
    from changan_plugin_pytorch.nn.quantized.functional import (
        channel_shuffle as quantized_channel_shuffle,
    )

    ret = quantized_channel_shuffle(input.as_subclass(Tensor), groups)

    if isinstance(input, QTensor):
        return QTensor(ret, input.q_scale(), input.dtype)
    else:
        return ret


def point_pillars_scatter(
    voxel_features: Union[Tensor, QTensor],
    coords: Tensor,
    output_shape: Union[Tensor, List[int]],
) -> Union[Tensor, QTensor]:
    if isinstance(output_shape, Tensor):
        output_shape = output_shape.tolist()

    from changan_plugin_pytorch.nn.quantized.functional import (
        point_pillars_scatter as quantized_point_pillars_scatter,
    )

    ret = quantized_point_pillars_scatter(
        voxel_features.as_subclass(Tensor), coords, output_shape
    )

    if isinstance(voxel_features, QTensor):
        return QTensor(ret, voxel_features.q_scale(), voxel_features.dtype)
    else:
        return ret


def window_partition(x: Union[Tensor, QTensor], window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    from changan_plugin_pytorch.nn.quantized.functional import (
        window_partition as quantized_window_partition,
    )

    ret = quantized_window_partition(x.as_subclass(Tensor), window_size)
    if isinstance(x, QTensor):
        return QTensor(ret, x.q_scale(), x.dtype)
    else:
        return ret


def window_reverse(
    windows: Union[Tensor, QTensor], window_size: int, H: int, W: int
):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    from changan_plugin_pytorch.nn.quantized.functional import (
        window_reverse as quantized_window_reverse,
    )

    ret = quantized_window_reverse(
        windows.as_subclass(Tensor), window_size, H, W
    )
    if isinstance(windows, QTensor):
        return QTensor(ret, windows.q_scale(), windows.dtype)
    else:
        return ret


def rle(input: Union[Tensor, QTensor], dtype: torch.dtype) -> List[Tensor]:
    """
    RLE compression algorithm. Compress the tensor in the format of
    [pair_num, value, num, value, num, ...]

    pair_num: indicates (value, num) pair amount
    value: the compressed data, support int8/16 dtype
    num: amount of the contiguous value. It dtype corresponds to value dtype.
        (int8 value, uint8 num / int16 value, uint16 num)

    Args:
        input(Tensor/QTensor): The data to be compressed
        dtype(torch.dtype): The value field dtype in compressed result.
            !!! Note: Not compressed results dtype. Result dtype is int64 !!!
            Support torch.int8 or torch.int16. if input is torch.max
            indices out, dtype must be torch.int16
            if value dtype = torch.int8, num dtype is uint8, max num is 255
            if value dtype = torch.int16, num dtype is uint16, max num is 65535

    Returns:
        A list composed of N Tensor. Each Tensor represents the
        compressed result of each batch with format
        [pair_num, value, num, value, num, ...]


    Examples:
        input:
            [[[[0, 1],[1, 1]]],
             [[[0, 1], [0, 0]]]]
        output: [tensor[2, 0, 1, 1, 2], tensor[3, 0, 1, 1, 1, 0, 2]]
    """
    from changan_plugin_pytorch.nn.quantized.functional import (
        rle as quantized_rle,
    )

    return quantized_rle(input.as_subclass(Tensor), dtype)


def point_pillars_preprocess(
    points_list: List[Tensor],
    pc_range: Tensor,
    voxel_size: Tensor,
    max_voxels: int,
    max_points_per_voxel: int,
    use_max: bool,
) -> Tuple[Tensor, Tensor]:
    """PointPillars preprocess.

    Args:
        points_list: [(M1, ndim), (M2, ndim),...], List of PointCloud data.
        pc_range: (6,), indicate voxel range, format:
            [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: (3,), xyz, indicate voxel size.
        max_voxels: Indicate maximum voxels.
        max_points_per_voxel: Indicate maximum points contained in a voxel.
        use_max: Whether to use max_voxels, for deploy should be True.

    Returns:
        (features, coords): Encoded feature and coordinates in
            (idx, z, y, x) format.
    """

    from changan_plugin_pytorch.nn.quantized.functional import (
        point_pillars_preprocess as _point_pillars_process,
    )

    features, coords = _point_pillars_process(
        points_list,
        pc_range,
        voxel_size,
        max_voxels,
        max_points_per_voxel,
        use_max,
    )

    return features, coords
