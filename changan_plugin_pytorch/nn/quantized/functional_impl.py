import math
import warnings

import numpy as np
import torch
from changan_plugin_pytorch import functional as hF
import torch.nn.functional as F
from changan_plugin_pytorch.dtype import qinfo
from torch import Tensor
from torch.jit.annotations import (
    BroadcastingList2,
    BroadcastingList3,
    List,
    Optional,
    Tuple,
)
from torch.nn.modules.utils import _pair

from .activation_function_fit_utils import (
    _get_multi_table_params,
    _lut_int8_to_int8,
    _multi_table_fit,
)


def _quantize(
    input: Tensor, scale: Tensor, zero_point: Tensor, dtype: str, march: str
) -> Tensor:
    info = qinfo(dtype)
    return torch.ops.changan.gpu_scale_quantization(
        input,
        scale,
        zero_point,
        -1 if scale.numel() == 1 else 1,
        info.min,
        info.max,
        dtype,
        march,
    )


def _dequantize(
    input: Tensor, scale: Tensor, zero_point: Tensor, ch_axis: int, march: str
) -> Tensor:
    return torch.ops.changan.gpu_scale_dequantization(input, scale, ch_axis)


def _requantize(
    input: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if torch.all(torch.eq(scale, input_scale)):
        if dtype == input_dtype:
            return input
        else:
            in_info = qinfo(input_dtype)
            out_info = qinfo(dtype)
            # clip the boundary value
            # if int32 128 requantize to the same scale int8
            # directly change dtype will return -128 !!
            return input.clip(
                max(in_info.min, out_info.min), min(in_info.max, out_info.max)
            ).to(dtype=out_info._storage_type)

    return torch.ops.changan.gpu_scale_requantization(
        input,
        input_scale,
        scale,
        input_dtype,
        dtype,
        False,
        march,
    )


def _conv_convert_int_params(
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    weight_dtype: str,
    bias: Tensor,
    bias_scale: Tensor,
    bias_dtype: str,
    out_scale: Tensor,
    out_dtype: str,
    input2_scale: Optional[Tensor],
    is_conv_transpose2d: bool,
    groups: int,
    march: str,
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:

    return torch.ops.changan.convert_conv_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        out_scale,
        out_dtype,
        input2_scale
        if input2_scale is not None
        else torch.ones_like(out_scale),
        is_conv_transpose2d,
        groups,
        march,
    )


def _conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    groups: int,
    padding_mode: str,
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    convert_ret = _conv_convert_int_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        scale,
        dtype,
        sumin_scale,
        False,
        1,
        march,
    )
    filters = weight.size()[0]
    kernel_size = (weight.size()[2], weight.size(3))
    if march == "bernoulli":
        (
            bpu_weight,
            bpu_weight_shift,
            bpu_bias,
            bpu_bias_shift,
            bpu_input_shift,
            bpu_output_shift,
            bpu_sumin_shift,
            dequant_output_scale,
            _,
        ) = convert_ret
        if not torch.all(
            bpu_input_shift + bpu_weight_shift - bpu_output_shift >= 0
        ):
            warnings.warn(
                "Not support output left shift on Bernoulli hardware, "
                "which may cause unexpected result or accuracy mismatch here."
            )
        conv_ret = torch.ops.changan.gpu_scale_quanti_convolution_with_shift(
            input,
            bpu_weight,
            bpu_bias,
            sumin if sumin is not None else torch.zeros(1).to(input),
            bpu_input_shift.item(),
            bpu_output_shift.item(),
            bpu_bias_shift,
            bpu_weight_shift,
            bpu_sumin_shift.item(),
            True,  # use_bias
            filters,  # filters
            kernel_size,  # kernel_size
            stride,
            padding,
            dilation,
            activation,
            groups,
            True if sumin is not None else False,  # elementwise_input
            True
            if dtype == "qint32"
            else False,  # disable_output_quantization
            dtype,
            march,
        )
    else:
        # int-conv2d
        # Calculation formula：
        # * f <- convolution
        # * x <- feature,  w <- weight,  e <- sumin, b <- bias
        # * y = saturate8(saturate16(f(x, w) + (b << bias_left_shift) +
        #         truncate16(e << sumin_left_shift) * sumin_scale))
        #         >> accu_right_shift) * output_scale >> output_right_shift)
        (
            bpu_weight,
            bpu_bias,
            bpu_bias_lshift,
            bpu_escale,
            bpu_escale_lshift,
            bpu_oscale,
            bpu_accu_rshift,
            bpu_output_rshift,
            dequant_output_scale,
        ) = convert_ret
        conv_ret = torch.ops.changan.gpu_scale_quanti_convolution(
            input,
            bpu_weight,
            bpu_bias,
            sumin if sumin is not None else torch.zeros(1).to(input),
            bpu_oscale,
            bpu_accu_rshift,
            bpu_bias_lshift,
            bpu_output_rshift,
            bpu_escale,
            bpu_escale_lshift,
            True,  # use_bias,
            filters,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            groups,
            True if sumin is not None else False,  # elementwise_input,
            True
            if dtype == "qint32"
            else False,  # disable_output_quantization,
            dtype,  # out_quanti_type,
            march,
        )
    return conv_ret, dequant_output_scale


def _conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    stride: BroadcastingList3[int],
    padding: BroadcastingList3[int],
    dilation: BroadcastingList3[int],
    groups: int,
    padding_mode: str,
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    (
        bpu_weight,
        bpu_bias,
        bpu_bias_lshift,
        bpu_escale,
        bpu_escale_lshift,
        bpu_oscale,
        bpu_accu_rshift,
        bpu_output_rshift,
        dequant_output_scale,
    ) = _conv_convert_int_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        scale,
        dtype,
        sumin_scale,
        False,
        1,
        march,
    )
    filters = weight.size(0)
    kernel_size = (weight.size(2), weight.size(3), weight.size(4))
    conv_ret = torch.ops.changan.gpu_scale_quanti_convolution3d(
        input,
        bpu_weight,
        bpu_bias,
        sumin if sumin is not None else torch.zeros(1).to(input),
        bpu_oscale,
        bpu_accu_rshift,
        bpu_bias_lshift,
        bpu_output_rshift,
        bpu_escale,
        bpu_escale_lshift,
        True,  # use_bias,
        filters,
        kernel_size,
        stride,
        padding,
        dilation,
        activation,
        groups,
        True if sumin is not None else False,  # elementwise_input,
        True if dtype == "qint32" else False,  # disable_output_quantization,
        dtype,  # out_quanti_type,
        march,
    )
    return conv_ret, dequant_output_scale


def _max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    return_indices: bool,
    ceil_mode: bool,
    march: str,
) -> Tensor:
    return F.max_pool2d(
        input.to(dtype=torch.float32),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        # directly pass `return_indices` will cause
        # 'boolean dispatch not constant error' when trace
        False,
    ).to(input.dtype)


def _avg_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: None,
    input_scale: Tensor,
    input_zero_point: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    if march == "bernoulli":
        assert dtype == "qint8", "bernoulli only support int8 output"
        res = torch.ops.changan.gpu_scale_quanti_pooling(
            input,
            "avg",
            kernel_size,
            padding,
            stride,
            ceil_mode,
            "qint8",
            march,
        )
        return res, input_scale

    accu = torch.ops.changan.gpu_scale_quanti_pooling(
        input, "sum", kernel_size, padding, stride, ceil_mode, "qint32", march
    )

    intermediate_scale = input_scale * (1 / kernel_size[0] / kernel_size[1])

    if dtype == "qint32":
        return accu, intermediate_scale

    return (
        torch.ops.changan.gpu_scale_requantization(
            accu,
            intermediate_scale,
            scale,
            "qint32",
            dtype,
            True if march == "bayes" else False,
            march,
        ),
        scale,
    )


def _interpolate(
    input: Tensor,
    size: Optional[BroadcastingList2[int]],
    scale_factor: Optional[BroadcastingList2[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    march: str,
) -> Tensor:
    if size is not None:
        out_height, out_width = size
    else:
        out_height, out_width = -1, -1
    if scale_factor is not None:
        ratio_height, ratio_width = scale_factor
    else:
        ratio_height, ratio_width = -1.0, -1.0

    # Note!!!
    # We use center mode when implementing nearest interpolate
    # Result of torch 'nearest' interpolate shifts to the bottom right
    # https://github.com/pytorch/pytorch/issues/34808
    if align_corners is None:
        align_corners = False
    return torch.ops.changan.gpu_quanti_resize(
        input,
        mode,
        align_corners,
        out_height,
        out_width,
        ratio_height,
        ratio_width,
        march,
    )


def _pad(
    input: Tensor,
    pad: List[int],
    mode: str,
    value: float,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if mode == "constant":
        padding_value = float(
            _quantize(
                torch.tensor([float(value)], device=input.device),
                scale,
                zero_point,
                dtype,
                march,
            )[0],
        )
    else:
        padding_value = float(value)

    return torch.nn.functional.pad(
        input.to(dtype=torch.float32), pad, mode, value=padding_value
    ).to(dtype=input.dtype)


def _masked_fill(
    input: Tensor,
    mask: Tensor,
    value: float,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    filled_value = _quantize(
        torch.tensor([float(value)], device=input.device),
        scale,
        zero_point,
        dtype,
        march,
    )[0]

    return torch.masked_fill(
        input,
        mask,
        filled_value,
    )


def _roi_align_list(
    input: Tensor,
    boxes: List[Tensor],
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    march: str,
) -> Tensor:
    if isinstance(boxes, (list, tuple)):
        assert len(boxes) == input.size(
            0
        ), "The length of roi list should be equal to batch size"
        for _tensor in boxes:
            assert _tensor.size(1) == 4, (
                "The shape of the tensor in the boxes list is"
                + " not correct as List[Tensor[L, 4]]"
            )
    else:
        assert (
            False
        ), "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"

    if march == "bernoulli2" or march == "bernoulli":
        assert not boxes[
            0
        ].is_floating_point(), (
            "roi_align of bernoulli2 only support fixed point rois"
        )

    roi_quantized = not boxes[0].is_floating_point()

    output_size = _pair(output_size)

    rois = boxes

    device = input.device
    roi_dtype = rois[0].dtype

    max_roi_num = max([roi.size(0) for roi in rois])

    aligned_rois: List[Tensor] = []
    valid_mask = torch.empty(0, dtype=torch.bool, device=device)

    # check if illegal roi
    illegal_roi_mask = torch.empty(0, dtype=torch.bool, device=device)

    for roi_per_image in rois:
        append_length = max_roi_num - roi_per_image.size(0)
        aligned_rois.append(
            torch.cat(
                (
                    roi_per_image,
                    torch.zeros(
                        size=(append_length, 4), dtype=roi_dtype, device=device
                    ),
                ),
                dim=0,
            )
        )
        valid_mask = torch.cat(
            (
                valid_mask,
                torch.ones(
                    roi_per_image.size(0), dtype=torch.bool, device=device
                ),
                torch.zeros(append_length, dtype=torch.bool, device=device),
            ),
            dim=0,
        )
        # roi is on the left or on the top of the feature
        if_left_top = torch.logical_or(
            roi_per_image[:, 2] < 0, roi_per_image[:, 3] < 0
        )
        # roi is on the right or on the bottom of the feature
        if_right_bottom = torch.logical_or(
            roi_per_image[:, 0] * spatial_scale
            > (input.size(3) * 4 if roi_quantized else input.size(3)),
            roi_per_image[:, 1] * spatial_scale
            > (input.size(2) * 4 if roi_quantized else input.size(2)),
        )
        roi_out_feature = torch.logical_or(if_left_top, if_right_bottom)
        if_negative_roi = torch.logical_or(
            (roi_per_image[:, 2] - roi_per_image[:, 0] <= 0),
            (roi_per_image[:, 3] - roi_per_image[:, 1] <= 0),
        )

        illegal_roi_mask = torch.cat(
            (
                illegal_roi_mask,
                torch.logical_or(roi_out_feature, if_negative_roi),
                torch.zeros(append_length, dtype=torch.bool, device=device),
            ),
            dim=0,
        )

    batched_rois = torch.stack(aligned_rois, dim=0)

    ret = torch.ops.changan.gpu_quanti_roi_resize(
        input,
        batched_rois,
        spatial_scale,
        output_size[0] * sampling_ratio,
        output_size[1] * sampling_ratio,
        aligned,
        interpolate_mode,
        march,
    )

    ret[illegal_roi_mask] = 0
    return ret[valid_mask]


def _roi_align_tensor(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    march: str,
) -> Tensor:
    if isinstance(boxes, torch.Tensor):
        assert (
            boxes.size(1) == 5
        ), "The boxes tensor shape is not correct as Tensor[K, 5]"
    else:
        assert (
            False
        ), "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"

    rois = boxes

    rois_list: List[Tensor] = []
    forward_mapping = torch.empty(0, dtype=torch.int, device=input.device)

    for batch_idx in range(input.size(0)):
        if not boxes.is_floating_point():
            batch_idx *= 4
        rois_list.append(rois[rois[:, 0] == batch_idx, 1:])
        forward_mapping = torch.cat(
            (forward_mapping, (rois[:, 0] == batch_idx).nonzero()), dim=0
        )

    reverse_mapping = torch.argsort(
        forward_mapping.flatten(), descending=False
    )

    return _roi_align_list(
        input,
        rois_list,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
        interpolate_mode,
        march,
    )[reverse_mapping]


def _cat(
    input_list: List[Tensor],
    dim: int,
    input_scale_list: List[Tensor],
    input_zero_point_list: List[Tensor],
    input_dtype_list: List[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    rescaled_input_list: List[Tensor] = []
    for input, in_scale, in_zero_point, in_dtype in zip(
        input_list, input_scale_list, input_zero_point_list, input_dtype_list
    ):
        rescaled_input_list.append(
            _requantize(
                input,
                in_scale,
                in_zero_point,
                in_dtype,
                scale,
                zero_point,
                dtype,
                march,
            )
        )

    return torch.cat(rescaled_input_list, dim)


def _add(
    x: Tensor,
    y: Tensor,
    x_scale: Tensor,
    y_scale: Tensor,
    x_zero_point: Tensor,
    y_zero_point: Tensor,
    x_dtype: str,
    y_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if march == "bayes":
        shift = 14

        if x_scale.prod() > y_scale.prod():
            intermediate_scale = x_scale / (2 ** shift)
            x = x.to(dtype=torch.int32) << 14
            origin_shape = y.shape
            y = torch.ops.changan.gpu_scale_requantization(
                y.unsqueeze(1) if y.ndim < 2 else y,
                y_scale,
                intermediate_scale,
                y_dtype,
                "qint32",
                False,  # pre_rshift_with_round
                False,  # post_rshift_with_round
                march,
            ).reshape(origin_shape)
        else:
            intermediate_scale = y_scale / (2 ** shift)
            origin_shape = x.shape
            x = torch.ops.changan.gpu_scale_requantization(
                x.unsqueeze(1) if x.ndim < 2 else x,
                x_scale,
                intermediate_scale,
                x_dtype,
                "qint32",
                False,  # pre_rshift_with_round
                False,  # post_rshift_with_round
                march,
            ).reshape(origin_shape)
            y = y.to(dtype=torch.int32) << 14

    elif march == "bernoulli2":
        if x_scale.prod() > y_scale.prod():
            feature = y
            feature_scale = y_scale
            sumin = x
            sumin_scale = x_scale
        else:
            feature = x
            feature_scale = x_scale
            sumin = y
            sumin_scale = y_scale

        intermediate_scale = torch.max(
            feature_scale / 127, sumin_scale / (1 << 25)
        )

        weight = (
            hF.round(feature_scale / intermediate_scale)
            .clamp(-128, 127)
            .to(dtype=torch.int8)
        )
        feature = feature.to(dtype=torch.int32) * weight.reshape(1, -1, 1, 1)

        sumin_weight = hF.round(sumin_scale / intermediate_scale)
        m, e = torch.frexp(sumin_weight)
        qm = (
            (2 ** e.clamp_max(15) * m)
            .clamp_max((1 << 15) - 1)
            .to(dtype=torch.int)
        )
        left_shift = (e - 15).clamp_min(0)
        sumin = (
            sumin.to(dtype=torch.int32) << left_shift.reshape(1, -1, 1, 1)
        ) * qm.reshape(1, -1, 1, 1)

        x = feature
        y = sumin

    else:
        shift = 14
        intermediate_scale = torch.max(x_scale, y_scale) / (2 ** shift)
        if x_scale.numel() > 1:
            x = x.to(dtype=torch.int32) * hF.round(
                x_scale / intermediate_scale
            ).reshape(1, -1, 1, 1).to(dtype=torch.int32)
            y = y.to(dtype=torch.int32) * hF.round(
                y_scale / intermediate_scale
            ).reshape(1, -1, 1, 1).to(dtype=torch.int32)
        else:
            x = x.to(dtype=torch.int32) * hF.round(
                x_scale / intermediate_scale
            ).to(dtype=torch.int32)
            y = y.to(dtype=torch.int32) * hF.round(
                y_scale / intermediate_scale
            ).to(dtype=torch.int32)

    add_res = torch.add(x, y)
    add_res = _requantize(
        add_res,
        intermediate_scale,
        torch.zeros_like(intermediate_scale).to(dtype=torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )
    return add_res


def _grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    grid_scale: Tensor,
    grid_zero_point: Tensor,
    grid_dtype: str,
    march: str,
) -> Tensor:
    # Convert from xy to yx.
    grid_yx = torch.stack((grid[..., 1], grid[..., 0]), dim=-1)

    # Compute coord_shift.
    h, w = input.size(2), input.size(3)
    grid_h, grid_w = grid.size(2), grid.size(3)

    h = h if h > grid_h else grid_h
    w = w if w > grid_w else grid_w

    max_coord = h if h > w else w
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = min(coord_shift, 8)
    coord_shift = coord_shift if coord_shift > 0 else 0

    # Coord int16 quantization.
    grid_out_scale = torch.tensor(
        1.0 / (1 << coord_shift), dtype=torch.float, device=grid.device
    ).reshape(1)
    grid_yx = _requantize(
        grid_yx,
        grid_scale,
        grid_zero_point,
        grid_dtype,
        grid_out_scale,
        grid_zero_point,
        "qint16",
        march,
    )

    # Convert to absolute grid.
    N, H, W, _ = grid.shape
    base_coord = (
        torch.stack(
            [
                torch.arange(H, dtype=torch.int32, device=grid.device)
                .reshape(1, H, 1)
                .expand(N, H, W),
                torch.arange(W, dtype=torch.int32, device=grid.device)
                .reshape(1, 1, W)
                .expand(N, H, W),
            ],
            dim=-1,
        )
        * (1 << coord_shift)
    )
    absolute_grid = grid_yx + base_coord
    # Convert grid format from [n, h, w, (y, x)] to [n, 1, (y, x), h, w].
    absolute_grid = absolute_grid.permute(0, 3, 1, 2).unsqueeze(1)

    return torch.ops.changan.gpu_quanti_grid_sample(
        input,
        absolute_grid,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march,
    )


def _grid_sample_norm_grid(
    input: Tensor,
    grid: Tensor,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    grid_scale: Tensor,
    grid_zero_point: Tensor,
    grid_dtype: str,
    march: str,
) -> Tensor:
    # Compute coord_shift.
    h, w = input.size(2), input.size(3)
    grid_h, grid_w = grid.size(1), grid.size(2)

    max_coord = max_coord = max(max(h, w), max(grid_h, grid_w))
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)

    # convert grid from -1 ~ 1 to -(size - 1) / 2 ~ (size - 1) / 2
    # and same out scale
    grid_x = grid[..., :1]
    grid_y = grid[..., 1:]

    grid_out_scale = torch.tensor(
        1.0 / (1 << coord_shift), dtype=torch.float, device=grid.device
    ).reshape(1)

    rescaled_grid_x = _requantize(
        grid_x,
        grid_scale * (w - 1) / 2,
        grid_zero_point,
        grid_dtype,
        grid_out_scale,
        grid_zero_point,
        "qint16",
        march,
    )
    rescaled_grid_y = _requantize(
        grid_y,
        grid_scale * (h - 1) / 2,
        grid_zero_point,
        grid_dtype,
        grid_out_scale,
        grid_zero_point,
        "qint16",
        march,
    )

    # add (size - 1) / 2 to grid
    rescaled_grid_x += int((1 << coord_shift) * (w - 1) / 2)
    rescaled_grid_y += int((1 << coord_shift) * (h - 1) / 2)

    absolute_grid_yx = torch.cat((rescaled_grid_y, rescaled_grid_x), dim=3)
    # Convert grid format from [n, h, w, (y, x)] to [n, 1, (y, x), h, w].
    absolute_grid_yx = absolute_grid_yx.permute(0, 3, 1, 2).unsqueeze(1)

    return torch.ops.changan.gpu_quanti_grid_sample(
        input,
        absolute_grid_yx,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march,
    )


def _filter(
    inputs: List[Tensor],
    scales: List[Tensor],
    zero_points: List[Tensor],
    dtypes: List[str],
    threshold: float,
    idx_range: Tuple[int, int],
    march: str,
) -> List[List[Tensor]]:
    if inputs[0].dtype in (torch.int8, torch.int16, torch.int32):
        is_bpu_inference = True
        inputs = [
            _dequantize(data, scale, zero_point, -1, march)
            for data, scale, zero_point in zip(inputs, scales, zero_points)
        ]
    else:
        is_bpu_inference = False

    score = inputs[0]

    _qtype_limit = {
        "qint4": (-8, 7),
        "quint4": (0, 15),
        "qint8": (-128, 127),
        "qint16": (-32768, 32767),
    }

    if is_bpu_inference:
        if march == "bayes":
            working_threshold = (
                (
                    (
                        (threshold / scales[0] + zero_points[0])
                        .ceil()
                        .clamp(*_qtype_limit[dtypes[0]])
                        - zero_points[0]
                    )
                    * scales[0]
                )
                .to(dtype=score.dtype)
                .item()
            )
        else:
            working_threshold = (
                (
                    (
                        (threshold / scales[0] + zero_points[0])
                        .floor()
                        .clamp(*_qtype_limit[dtypes[0]])
                        - zero_points[0]
                    )
                    * scales[0]
                )
                .to(dtype=score.dtype)
                .item()
            )
    else:
        working_threshold = threshold

    max_value, max_idx = score[:, idx_range[0] : idx_range[1], :, :].max(
        dim=1, keepdim=False
    )
    max_idx = max_idx + idx_range[0]

    if march == "bayes":
        mask = max_value >= working_threshold
    else:
        mask = max_value > working_threshold

    batch_size, c, h, w = score.shape
    otype = torch.int16 if is_bpu_inference else score.dtype
    h_index = (
        torch.arange(h, device=score.device, dtype=otype)
        .reshape(1, 1, -1, 1)
        .expand(batch_size, 1, h, w)
    )
    w_index = (
        torch.arange(w, device=score.device, dtype=otype)
        .reshape(1, 1, 1, -1)
        .expand(batch_size, 1, h, w)
    )
    coord = torch.cat([h_index, w_index], dim=1)

    if is_bpu_inference:
        max_idx = max_idx.to(dtype=torch.int16)
        coord = coord.to(dtype=torch.int16)

    mask = mask.flatten(1, 2)
    max_value = max_value.flatten(1, 2)
    max_idx = max_idx.flatten(1, 2)
    coord = coord.permute(0, 2, 3, 1).flatten(1, 2)
    inputs = [input.permute(0, 2, 3, 1).flatten(1, 2) for input in inputs]

    batch_size, _, h, w = score.shape

    ret: List[List[Tensor]] = []
    for i in range(batch_size):
        m = mask[i]
        per_image_ret = [max_value[i][m], max_idx[i][m], coord[i][m]]
        per_image_ret += [data[i][m] for data in inputs]
        ret.append(per_image_ret)

    return ret


def _max(
    input: Tensor,
    dim: int,
    keepdim: bool,
    group: int,
    march: str,
) -> Tuple[Tensor, Tensor]:
    idx, value = torch.ops.changan.gpu_post_process_channel_argmax(
        input, group, march
    )
    return value, idx


def _sub(
    x: Tensor,
    y: Tensor,
    x_scale: Tensor,
    x_zero_point: Tensor,
    x_dtype: str,
    y_scale: Tensor,
    y_zero_point: Tensor,
    y_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if march == "bayes":
        max_shift = 14
        intermediate_scale = torch.max(x_scale, y_scale) / (2 ** max_shift)
        shift = (x_scale / intermediate_scale).log2().floor()
        intermediate_scale = x_scale / (2 ** shift)

        x = x.to(dtype=torch.int32) << shift.to(dtype=torch.int32)
        y = torch.ops.changan.gpu_scale_requantization(
            y,
            y_scale.negative(),
            intermediate_scale,
            y_dtype,
            "qint32",
            False,  # pre_rshift_with_round
            False,  # post_rshift_with_round
            march,
        )
        ret = torch.add(x, y)
        return _requantize(
            ret,
            intermediate_scale,
            torch.zeros_like(intermediate_scale).to(dtype=torch.long),
            "qint32",
            scale,
            zero_point,
            dtype,
            march,
        )
    else:
        info = qinfo(y_dtype)
        return _add(
            x,
            (y.to(torch.int32) * -1).clamp(info.min, info.max),
            x_scale,
            y_scale,
            x_zero_point,
            y_zero_point,
            x_dtype,
            y_dtype,
            scale,
            zero_point,
            dtype,
            march,
        )


def _lut(
    data: Tensor,
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    table: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    assert data_type == "qint8" and dtype == "qint8"
    return _lut_int8_to_int8(table, data, march)


def _get_multi_table_params_impl(
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    left_line_xmin: Tensor,
    left_line_ymin: Tensor,
    left_line_xmax: Tensor,
    left_line_ymax: Tensor,
    right_line_xmin: Tensor,
    right_line_ymin: Tensor,
    right_line_xmax: Tensor,
    right_line_ymax: Tensor,
    left_constant_fit_y: Tensor,
    right_constant_fit_y: Tensor,
    qint_dense_xmin: Tensor,
    qint_dense_xmax: Tensor,
    qint_sparse_xmin: Tensor,
    qint_sparse_xmax: Tensor,
    march: str,
):
    return _get_multi_table_params(
        data_scale,
        data_zero_point,
        data_type,
        scale,
        zero_point,
        dtype,
        left_line_xmin,
        left_line_ymin,
        left_line_xmax,
        left_line_ymax,
        right_line_xmin,
        right_line_ymin,
        right_line_xmax,
        right_line_ymax,
        left_constant_fit_y,
        right_constant_fit_y,
        qint_dense_xmin,
        qint_dense_xmax,
        qint_sparse_xmin,
        qint_sparse_xmax,
    )


def _multi_table_fit_impl(
    data: Tensor,
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    dense_table: Tensor,
    qint_dense_xmin: Tensor,
    qint_dense_xmax: Tensor,
    sparse_table: Tensor,
    qint_sparse_xmin: Tensor,
    qint_sparse_xmax: Tensor,
    left_line_xmin: Tensor,
    left_line_ymin: Tensor,
    left_line_xmax: Tensor,
    left_line_ymax: Tensor,
    right_line_xmin: Tensor,
    right_line_ymin: Tensor,
    right_line_xmax: Tensor,
    right_line_ymax: Tensor,
    qint_left_constant_xmin: Tensor,
    qint_left_constant_xmax: Tensor,
    qint_right_constant_xmin: Tensor,
    qint_right_constant_xmax: Tensor,
    left_constant_fit_y: Tensor,
    right_constant_fit_y: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    is_symmetric: bool,
    symmetric_k: int,
    symmetric_b: Tensor,
    march: str,
) -> Tensor:
    out = torch.zeros_like(data, dtype=torch.int32)
    if is_symmetric:
        assert data_type != "qint8", "input int8 cannot use symmetric mode"
        # if use symmetric mode, compiler calculation:
        # (right_out * k + b) >> 8 if out_type is int8
        # so if out_type is int8, transform out_scale and out_type to int16
        # and do right shfit use int16 result to be consistent with compiler
        mask = torch.logical_and(
            data.to(torch.int32) >= 0, data.to(torch.int32) <= 32767
        )
        right_data = torch.masked_select(data, mask)
        right_out = _multi_table_fit(
            right_data,
            data_scale,
            data_zero_point,
            data_type,
            dense_table,
            qint_dense_xmin,
            qint_dense_xmax,
            sparse_table,
            qint_sparse_xmin,
            qint_sparse_xmax,
            left_line_xmin,
            left_line_ymin,
            left_line_xmax,
            left_line_ymax,
            right_line_xmin,
            right_line_ymin,
            right_line_xmax,
            right_line_ymax,
            qint_left_constant_xmin,
            qint_left_constant_xmax,
            qint_right_constant_xmin,
            qint_right_constant_xmax,
            left_constant_fit_y,
            right_constant_fit_y,
            scale,
            zero_point,
            "qint16",
            march,
            is_symmetric,
        )
        out.masked_scatter_(mask, right_out)
        mask = torch.logical_and(
            data.to(torch.int32) < 0, data.to(torch.int32) >= -32768
        )
        left_data = (-1) * torch.masked_select(data.to(torch.int32), mask)
        left_out = (
            _multi_table_fit(
                left_data,
                data_scale,
                data_zero_point,
                data_type,
                dense_table,
                qint_dense_xmin,
                qint_dense_xmax,
                sparse_table,
                qint_sparse_xmin,
                qint_sparse_xmax,
                left_line_xmin,
                left_line_ymin,
                left_line_xmax,
                left_line_ymax,
                right_line_xmin,
                right_line_ymin,
                right_line_xmax,
                right_line_ymax,
                qint_left_constant_xmin,
                qint_left_constant_xmax,
                qint_right_constant_xmin,
                qint_right_constant_xmax + 1,
                left_constant_fit_y,
                right_constant_fit_y,
                scale,
                zero_point,
                "qint16",
                march,
                is_symmetric,
            )
            * symmetric_k
            + symmetric_b.to(torch.int32)
        )
        out.masked_scatter_(mask, left_out)
    else:
        out = _multi_table_fit(
            data,
            data_scale,
            data_zero_point,
            data_type,
            dense_table,
            qint_dense_xmin,
            qint_dense_xmax,
            sparse_table,
            qint_sparse_xmin,
            qint_sparse_xmax,
            left_line_xmin,
            left_line_ymin,
            left_line_xmax,
            left_line_ymax,
            right_line_xmin,
            right_line_ymin,
            right_line_xmax,
            right_line_ymax,
            qint_left_constant_xmin,
            qint_left_constant_xmax,
            qint_right_constant_xmin,
            qint_right_constant_xmax,
            left_constant_fit_y,
            right_constant_fit_y,
            scale,
            zero_point,
            "qint16",
            march,
        )
    if dtype == "qint8":
        out = ((out.to(torch.int32) + 128) >> 8).to(torch.int16)
        out = torch.clamp(out, -128, 127).to(torch.int8)
    else:
        out = torch.clamp(out, -32768, 32767).to(torch.int16)
    return out


def _matmul(
    input: Tensor,
    other: Tensor,
    input_trans: bool,
    other_trans: bool,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    other_scale: Tensor,
    other_zero_point: Tensor,
    other_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if input_trans:
        input = input.transpose(-1, -2)
    if other_trans:
        other = other.transpose(-1, -2)

    intermediate_scale = input_scale * other_scale

    if input_dtype == "qint16":
        # We need to consider two constraints:
        #  1. BPU input range is limited to [-32768, 32767 - 128]
        #  2. Avoid overflow of sum operation
        #     For [M, K] [K, N] matmul, there are K values to be sumed, and
        #     the result is of int32 type, so each value is limited to
        #     [INT32_MIN / K, INT32_MAX / K], input value is limited to
        #     [-sqrt(INT32_MAX / K), sqrt(INT32_MAX / K)] ~=
        #     [INT16_MIN * sqrt(2 / K), INT16_MAX * sqrt(2 / K)]
        #     Input type is int16, and the value range is multiplied with
        #     sqrt(2 / K), so the scale should multiply with sqrt(K / 2)
        scale_scale_1 = 32767 / (32767 - 128)
        scale_scale_2 = np.math.sqrt(input.size(-1) / 2)
        scale_scale = (
            scale_scale_1 if scale_scale_1 > scale_scale_2 else scale_scale_2
        )

        intermediate_scale *= scale_scale ** 2

        input = _requantize(
            input,
            input_scale,
            input_zero_point,
            input_dtype,
            input_scale * scale_scale,
            torch.zeros_like(input_scale).to(dtype=torch.long),
            "qint16",
            march,
        )
        other = _requantize(
            other,
            other_scale,
            other_zero_point,
            other_dtype,
            other_scale * scale_scale,
            torch.zeros_like(other_scale).to(dtype=torch.long),
            "qint16",
            march,
        )

    if input.is_cuda:
        res = torch.matmul(
            input.to(dtype=torch.float64), other.to(dtype=torch.float64)
        ).to(dtype=torch.int32)
    else:
        res = torch.matmul(
            input.to(dtype=torch.int32), other.to(dtype=torch.int32)
        )

    res = _requantize(
        res,
        intermediate_scale,
        torch.zeros_like(intermediate_scale).to(dtype=torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )

    return res


def _base_grid_generator(
    size: BroadcastingList2[int],
    with_third_channel: bool,
    device: torch.device,
    march: str,
) -> Tensor:
    size = _pair(size)

    x = (
        torch.arange(size[1], dtype=torch.int16, device=device)
        .unsqueeze(0)
        .expand(size)
    )
    y = (
        torch.arange(size[0], dtype=torch.int16, device=device)
        .unsqueeze(-1)
        .expand(size)
    )

    tensor_list = [x, y]

    if with_third_channel:
        ones = torch.ones(size, dtype=torch.int16, device=device)
        tensor_list.append(ones)

    return torch.stack(tensor_list, dim=-1)


def _mul(
    input: Tensor,
    other: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    other_scale: Tensor,
    other_zero_point: Tensor,
    other_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if "qbool" in [input_dtype, other_dtype]:
        res = torch.mul(input, other)
        return res
    else:
        assert input_dtype == other_dtype
        if march == "bernoulli":
            assert input_dtype == "qint8"
            assert input.dtype == torch.int8
            assert other.dtype == torch.int8
            input_shift = (-1) * torch.log2(input_scale).to(torch.int8)
            other_shift = (-1) * torch.log2(other_scale).to(torch.int8)
            out_shift = (-1) * torch.log2(scale).to(torch.int8)
            data_x, data_y = torch.broadcast_tensors(input, other)
            res = torch.ops.changan.gpu_quanti_mul(
                data_x, data_y, input_shift, other_shift, out_shift, march
            )
            return res
        else:
            oscale = input_scale * other_scale
            res = torch.mul(
                input.to(dtype=torch.int32), other.to(dtype=torch.int32)
            )
            res = _requantize(
                res,
                oscale,
                torch.zeros_like(oscale).to(dtype=torch.long),
                "qint32",
                scale,
                zero_point,
                dtype,
                march,
            )
            return res


def _sum(
    x: Tensor,
    dim: int,
    keepdim: bool,
    x_scale: Tensor,
    x_zero_point: Tensor,
    x_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    r = torch.sum(x, dim, keepdim, dtype=torch.int32)
    return _requantize(
        r, x_scale, x_zero_point, "qint32", scale, zero_point, dtype, march
    )


def _softmax(
    data: Tensor,
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    exp_out_scale: Tensor,
    exp_out_zero_point: Tensor,
    exp_out_type: str,
    reciprocal_out_scale: Tensor,
    reciprocal_out_zero_point: Tensor,
    reciprocal_out_type: str,
    exp_dense_table: Tensor,
    exp_qint_dense_xmin: Tensor,
    exp_qint_dense_xmax: Tensor,
    exp_sparse_table: Tensor,
    exp_qint_sparse_xmin: Tensor,
    exp_qint_sparse_xmax: Tensor,
    exp_left_line_xmin: Tensor,
    exp_left_line_ymin: Tensor,
    exp_left_line_xmax: Tensor,
    exp_left_line_ymax: Tensor,
    exp_right_line_xmin: Tensor,
    exp_right_line_ymin: Tensor,
    exp_right_line_xmax: Tensor,
    exp_right_line_ymax: Tensor,
    exp_qint_left_constant_xmin: Tensor,
    exp_qint_left_constant_xmax: Tensor,
    exp_qint_right_constant_xmin: Tensor,
    exp_qint_right_constant_xmax: Tensor,
    exp_left_constant_fit_y: Tensor,
    exp_right_constant_fit_y: Tensor,
    rescale_shift: Tensor,
    reciprocal_dense_table: Tensor,
    reciprocal_qint_dense_xmin: Tensor,
    reciprocal_qint_dense_xmax: Tensor,
    reciprocal_sparse_table: Tensor,
    reciprocal_qint_sparse_xmin: Tensor,
    reciprocal_qint_sparse_xmax: Tensor,
    reciprocal_left_line_xmin: Tensor,
    reciprocal_left_line_ymin: Tensor,
    reciprocal_left_line_xmax: Tensor,
    reciprocal_left_line_ymax: Tensor,
    reciprocal_right_line_xmin: Tensor,
    reciprocal_right_line_ymin: Tensor,
    reciprocal_right_line_xmax: Tensor,
    reciprocal_right_line_ymax: Tensor,
    reciprocal_qint_left_constant_xmin: Tensor,
    reciprocal_qint_left_constant_xmax: Tensor,
    reciprocal_qint_right_constant_xmin: Tensor,
    reciprocal_qint_right_constant_xmax: Tensor,
    reciprocal_left_constant_fit_y: Tensor,
    reciprocal_right_constant_fit_y: Tensor,
    march: str,
) -> Tensor:
    data = data.to(torch.int16) - torch.max(
        data, dim=1, keepdim=True
    ).values.to(torch.int16)
    exp_out = _multi_table_fit(
        data,
        data_scale,
        data_zero_point,
        "qint16",
        exp_dense_table,
        exp_qint_dense_xmin,
        exp_qint_dense_xmax,
        exp_sparse_table,
        exp_qint_sparse_xmin,
        exp_qint_sparse_xmax,
        exp_left_line_xmin,
        exp_left_line_ymin,
        exp_left_line_xmax,
        exp_left_line_ymax,
        exp_right_line_xmin,
        exp_right_line_ymin,
        exp_right_line_xmax,
        exp_right_line_ymax,
        exp_qint_left_constant_xmin,
        exp_qint_left_constant_xmax,
        exp_qint_right_constant_xmin,
        exp_qint_right_constant_xmax,
        exp_left_constant_fit_y,
        exp_right_constant_fit_y,
        exp_out_scale,
        exp_out_zero_point,
        exp_out_type,
        march,
        False,
    )
    exp_out = torch.clamp(exp_out, qinfo("qint16").min, qinfo("qint16").max)
    exp_sum = torch.sum(exp_out, 1, True)
    exp_sum = torch.clamp(
        (exp_sum / (2 ** rescale_shift)),
        qinfo("qint16").min,
        qinfo("qint16").max,
    ).to(torch.int16)
    exp_sum_scale = exp_out_scale * 2 ** rescale_shift
    reciprocal_out = _multi_table_fit(
        exp_sum,
        exp_sum_scale,
        torch.zeros_like(exp_sum_scale).to(dtype=torch.long),
        "qint16",
        reciprocal_dense_table,
        reciprocal_qint_dense_xmin,
        reciprocal_qint_dense_xmax,
        reciprocal_sparse_table,
        reciprocal_qint_sparse_xmin,
        reciprocal_qint_sparse_xmax,
        reciprocal_left_line_xmin,
        reciprocal_left_line_ymin,
        reciprocal_left_line_xmax,
        reciprocal_left_line_ymax,
        reciprocal_right_line_xmin,
        reciprocal_right_line_ymin,
        reciprocal_right_line_xmax,
        reciprocal_right_line_ymax,
        reciprocal_qint_left_constant_xmin,
        reciprocal_qint_left_constant_xmax,
        reciprocal_qint_right_constant_xmin,
        reciprocal_qint_right_constant_xmax,
        reciprocal_left_constant_fit_y,
        reciprocal_right_constant_fit_y,
        reciprocal_out_scale,
        reciprocal_out_zero_point,
        reciprocal_out_type,
        march,
        False,
    )
    reciprocal_out = torch.clamp(
        reciprocal_out, qinfo("qint16").min, qinfo("qint16").max
    )
    intermediate_scale = exp_out_scale * reciprocal_out_scale
    intermediate_res = exp_out.to(torch.int32) * reciprocal_out.to(torch.int32)
    softmax_out = _requantize(
        intermediate_res,
        intermediate_scale,
        torch.zeros_like(intermediate_scale).to(dtype=torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )
    return softmax_out


def _detection_post_process_v1(
    # Tensors
    data: List[Tensor],
    anchor: List[Tensor],
    exp_table: Tensor,
    image_sizes: Tensor,
    # task params
    num_classes: int,
    # shifts
    input_shifts: List[int],
    exp_shift: int,
    # filter params
    box_filter_threshold: int,
    class_offsets: List[int],
    seed: int,
    # decode params
    use_clippings: bool,
    # nms params
    nms_threshold: int,
    nms_margin: int,
    post_nms_top_k: int,
    march: str,
) -> List[Tuple[Tensor, Tensor, Tensor]]:
    num_anchors: List[int] = [int(a.size(1) / 4) for a in anchor]
    anchor_start_idx: List[int] = [0]
    for per_branch_num_anchors in num_anchors:
        anchor_start_idx.append(anchor_start_idx[-1] + per_branch_num_anchors)
    anchor_start_idx = anchor_start_idx[:-1]

    block_sizes: List[Tuple[int, int]] = []

    if march == "bayes":
        for branch_data in data:
            block_sizes.append((branch_data.size(2), branch_data.size(3)))
    else:
        max_input_size = 144 * 4 * 2048
        for num_anchor in num_anchors:
            # per_anchor_size is aligned with 4
            per_anchor_size = math.ceil((4 + num_classes) / 4) * 4
            max_tile_area = math.floor(
                max_input_size
                / (math.ceil(per_anchor_size * num_anchor / 4) * 4)
            )
            max_tile_w = (
                math.ceil(math.floor(math.sqrt(max_tile_area)) / 8) * 8
            )
            max_tile_h = math.floor(max_tile_area / max_tile_w)
            block_sizes.append((max_tile_h, max_tile_w))

    stride_hw: List[Tuple[int, int]] = []
    for per_branch_anchor in anchor:
        stride_hw.append(
            (
                int(
                    (
                        per_branch_anchor[0, 1, 1, 0]
                        - per_branch_anchor[0, 1, 0, 0]
                    ).item()
                ),
                int(
                    (
                        per_branch_anchor[0, 0, 0, 1]
                        - per_branch_anchor[0, 0, 0, 0]
                    ).item()
                ),
            )
        )

    anchor = torch.cat(
        [
            per_branch_anchor[0, :, 0, 0].flatten()
            for per_branch_anchor in anchor
        ]
    ).reshape(-1, 4)

    x1 = anchor[:, 0]
    y1 = anchor[:, 1]
    x2 = anchor[:, 2]
    y2 = anchor[:, 3]

    anchor = torch.stack(
        [y2 - y1, x2 - x1, (y1 + y2) / 2, (x1 + x2) / 2], dim=-1
    )

    shifted_anchor = hF.round(anchor * 4).to(dtype=torch.int32)

    assert shifted_anchor.min() >= 0 and shifted_anchor.max() <= (
        (1 << 16) - 1
    ), "anchor value out of range"

    per_class_idx: List[int] = []
    per_class_threshold: List[int] = []

    batch_size = data[0].size(0)

    ret = torch.ops.changan.gpu_quanti_proposal(
        [d.cpu() for d in data],
        shifted_anchor.cpu(),
        exp_table.cpu(),
        image_sizes.expand(batch_size, 2).float().cpu(),
        num_anchors,
        [num_classes] * len(data),
        input_shifts,
        exp_shift,
        [block_size[0] for block_size in block_sizes],
        [block_size[1] for block_size in block_sizes],
        box_filter_threshold,
        per_class_idx,
        per_class_threshold,
        class_offsets,
        seed,
        anchor_start_idx,
        [s[0] for s in stride_hw],
        [s[1] for s in stride_hw],
        use_clippings,
        False,
        image_sizes[0][0].item(),
        image_sizes[0][1].item(),
        "hw",
        nms_threshold,
        post_nms_top_k,
        nms_margin,
        -1,
        march,
    ).to(device=anchor.device)

    ret_list: List[Tuple[Tensor, Tensor, Tensor]] = []
    for per_image_ret in ret:
        valid = per_image_ret[:, -1] >= 0
        per_image_ret = per_image_ret[valid]
        splited_ret: Tuple[Tensor, Tensor, Tensor] = (
            per_image_ret[:, :4].to(dtype=torch.int16),
            per_image_ret[:, 4].to(dtype=torch.int8),
            per_image_ret[:, 5],
        )
        ret_list.append(splited_ret)

    return ret_list


def _conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    output_padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    groups: int,
    padding_mode: str,
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    convert_ret = _conv_convert_int_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        scale,
        dtype,
        sumin_scale,
        True,
        groups,
        march,
    )

    filters = weight.size()[1] * groups
    kernel_size = (weight.size()[2], weight.size(3))
    if march == "bernoulli":
        (
            bpu_weight,
            bpu_weight_shift,
            bpu_bias,
            bpu_bias_shift,
            bpu_input_shift,
            bpu_output_shift,
            bpu_sumin_shift,
            dequant_output_scale,
            _,
        ) = convert_ret
        if not (
            torch.all(bpu_input_shift + bpu_weight_shift - bpu_bias_shift >= 0)
            and torch.all(
                bpu_input_shift + bpu_weight_shift - bpu_sumin_shift >= 0
            )
            and torch.all(
                bpu_input_shift + bpu_weight_shift - bpu_output_shift >= 0
            )
        ):
            warnings.warn(
                "Not support bias/sumin right shift or output left shift"
                "on Bernoulli hardware, which may cause accuracy mismatch"
            )
        conv_ret = torch.ops.changan.gpu_scale_quanti_deconvolution_with_shift(
            input,
            bpu_weight,
            bpu_bias,
            sumin if sumin is not None else torch.zeros(1).to(input),
            bpu_input_shift.item(),
            bpu_output_shift.item(),
            bpu_bias_shift,
            bpu_weight_shift,
            bpu_sumin_shift.item(),
            True,  # use_bias
            filters,  # filters
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            activation,
            groups,
            True if sumin is not None else False,  # elementwise_input
            True
            if dtype == "qint32"
            else False,  # disable_output_quantization
            dtype,
            march,
        )
    else:
        """
        int-conv_transpose_2d
        Calculation formula：
        * f <- convolution
        * x <- feature,  w <- weight,  e <- sumin, b <- bias
        * y = saturate8(saturate16(f(x, w) + (b << bias_left_shift) +
            truncate16(e << sumin_left_shift) * sumin_scale))
            >> accu_right_shift) * output_scale >> output_right_shift)

        """
        (
            bpu_weight,
            bpu_bias,
            bpu_bias_lshift,
            bpu_escale,
            bpu_escale_lshift,
            bpu_oscale,
            bpu_accu_rshift,
            bpu_output_rshift,
            dequant_output_scale,
        ) = convert_ret
        conv_ret = torch.ops.changan.gpu_scale_quanti_deconvolution(
            input,
            bpu_weight,
            bpu_bias,
            sumin if sumin is not None else torch.zeros(1).to(input),
            bpu_oscale,
            bpu_accu_rshift,
            bpu_bias_lshift,
            bpu_output_rshift,
            bpu_escale,
            bpu_escale_lshift,
            True,  # use_bias,
            filters,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            activation,
            groups,
            True if sumin is not None else False,  # elementwise_input,
            True
            if dtype == "qint32"
            else False,  # disable_output_quantization
            dtype,  # out_quanti_type,
            march,
        )
    return conv_ret, dequant_output_scale


def _multi_scale_roi_align(
    # input
    features: List[Tensor],
    boxes: List[Tensor],
    # roi_align parameters
    output_size: BroadcastingList2[int],
    spatial_scale: List[float],
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    # rois selection parameters
    canonical_box_size: int,
    canonical_level: int,
    # march
    march: str,
) -> Tensor:
    output_size = _pair(output_size)

    # if no boxes, return empty
    if len(boxes) == 0:
        return torch.empty(
            (0, features[0].shape[1]) + output_size,
            device=features[0].device,
            dtype=features[0].dtype,
        )

    # convert boxes from List[Tensor[L, 4]] to Tensor[M, 5]
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full_like(b[:, :1], i, dtype=dtype, device=device)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)

    num_level_assignments = len(spatial_scale)

    if num_level_assignments == 1:
        return _roi_align_list(
            features[0],
            boxes,
            output_size,
            spatial_scale[0],
            sampling_ratio,
            aligned,
            interpolate_mode,
            march,
        )

    # Bernoulli2 or Bernoulli roi_align only support fixed point box input
    # Bayes roi_align only support float box input
    # No influence when invoking roi_align,
    # BUT affect feature level mapping computation!!!
    if not boxes[0].is_floating_point():
        boxes = [box * 0.25 for box in boxes]

    box_sizes = torch.sqrt(
        torch.cat(
            [
                (each_box[:, 2] - each_box[:, 0])
                * (each_box[:, 3] - each_box[:, 1])
                for each_box in boxes
            ]
        )
    )
    # Eqn.(1) in FPN paper
    mapped_levels = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size) + 1e-8
    )
    levels = -torch.log2(
        torch.tensor(spatial_scale, device=mapped_levels.device)
    )
    mapped_levels = (
        torch.clamp(mapped_levels, min=levels[0], max=levels[-1]).to(
            torch.int64
        )
        - levels[0]
    )
    num_boxes = rois.size(0)
    num_channels = features[0].shape[1]

    dtype, device = features[0].dtype, features[0].device
    N = features[0].size(0)
    result = torch.zeros(
        (num_boxes, num_channels) + output_size, dtype=dtype, device=device
    )

    for level, scale in enumerate(spatial_scale):
        indexs = torch.where(mapped_levels == level)[0]
        rois_per_level = rois[indexs]

        # convert rois:Tensor[L, 5] to box_lists: List[Tensor[M,4]]
        # to avoid jit.script error when invoking roi_align_tensor here
        box_list: List[Tensor] = []
        for i in range(N):
            box_list.append(rois_per_level[rois_per_level[:, 0] == i, 1:])

        result.index_put_(
            (indexs,),
            _roi_align_list(
                features[level],
                box_list,
                output_size,
                scale,
                sampling_ratio,
                aligned,
                interpolate_mode,
                march,
            ),
        )
    return result


def _correlation(
    data1: Tensor,
    data2: Tensor,
    kernel_size: int,
    max_displacement: int,
    stride1: int,
    stride2: int,
    pad_size: int,
    is_multiply: bool,
    scale1: Tensor,
    zero_point1: Tensor,
    dtype1: str,
    scale2: Tensor,
    zero_point2: Tensor,
    dtype2: str,
    inter_scale: Tensor,
    out_scale: Tensor,
    out_zero_point: Tensor,
    out_dtype: str,
    march: str,
) -> Tensor:
    return torch.ops.changan.gpu_scale_quanti_correlation(
        data1,
        data2,
        scale1,
        scale2,
        inter_scale,
        out_scale,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        pad_size,
        is_multiply,
        out_dtype,
        march,
    )


def _mean(
    x: Tensor,
    dim: int,
    x_scale: Tensor,
    x_zero_point: Tensor,
    x_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if march == "bernoulli":
        assert (
            x_dtype == "qint8" and dtype == "qint8"
        ), "only support int8 input and output type"
        c = x.shape[1]
        device = x.device
        m, e = torch.frexp(torch.tensor(1 / c, device=device))
        fake_weight_value = torch.clamp(
            torch.floor(m * 128 + 0.5), -128, 127
        ) * torch.pow(2.0, e - 7)
        weight_scale = (2.0 ** (e - 7)).reshape(1)  # must guarantee dim=1
        weight = torch.full(
            (1, c, 1, 1), fake_weight_value, dtype=torch.float32, device=device
        )

        # use conv to compute
        out, _ = _conv2d(
            x,
            weight,
            torch.zeros(1, dtype=torch.float32).to(device),
            None,
            (1, 1),  # stride
            (0, 0),  # padding
            (1, 1),  # dilation
            1,
            "zeros",
            "",
            x_scale,
            x_zero_point,
            x_dtype,
            weight_scale,
            torch.zeros_like(weight_scale).to(torch.long),
            "qint8",
            torch.ones(1, dtype=torch.float32).to(device),
            torch.zeros(1, dtype=torch.long),
            "qint8",
            None,
            x_zero_point,
            None,
            scale,
            zero_point,
            dtype,
            march,
        )
        return out
    else:
        r = torch.sum(x, dim, True, dtype=torch.int32)
        return _requantize(
            r,
            x_scale,
            x_zero_point,
            "qint32",
            scale * x.shape[dim],
            zero_point,
            dtype,
            march,
        )


def _segment_lut(
    input: Tensor,
    table: Tensor,
    scales: Tensor,
    beta: Tensor,
    left_shift: Tensor,
    right_shift: Tensor,
    max: Tensor,
    is_centrosymmetric: bool,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
):
    return torch.ops.changan.gpu_segment_lut(
        input,
        table,
        scales,
        beta,
        left_shift,
        right_shift,
        max,
        is_centrosymmetric,
        8,
        dtype,
        march,
    )


def _point_pillars_scatter(
    voxel_features: Tensor, coords: Tensor, output_shape: List[int]
) -> Tensor:
    voxel_features = voxel_features.reshape((voxel_features.size(0), -1))

    batch_size = output_shape[0]
    channel_dim = voxel_features.size(1)

    hight = output_shape[-2]
    width = output_shape[-1]

    canvas = torch.zeros(
        batch_size * hight * width,
        channel_dim,
        dtype=voxel_features.dtype,
        device=voxel_features.device,
    )

    index = (
        coords[:, 0] * (hight * width) + coords[:, -2] * width + coords[:, -1]
    )

    canvas[index] = voxel_features

    return canvas.reshape(batch_size, hight, width, channel_dim).permute(
        0, 3, 1, 2
    )


def _channel_shuffle(input: Tensor, groups: int):
    batch_size, channels, height, width = input.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    input = input.contiguous()
    input = input.view(batch_size, groups, channels_per_group, height, width)
    input = input.transpose(1, 2).contiguous()

    input = input.view(batch_size, channels, height, width)

    return input


def _prelu(
    input: Tensor,
    weight: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    # prelu out = max(input, 0) + weight * min(input, 0)

    positive_out = input.clamp_min(0)
    positive_out = _requantize(
        positive_out,
        input_scale,
        input_zero_point,
        input_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )

    negative_out = weight.reshape(-1, 1, 1) * input.clamp_max(0).to(
        torch.int32
    )
    negative_out = _requantize(
        negative_out.to(torch.int32),
        input_scale * weight_scale,
        input_zero_point,
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )

    return positive_out + negative_out


def _window_partition(x: Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size, window_size, W // window_size, window_size, C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def _window_reverse(windows: Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def _linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    conv_input = input.reshape(-1, input.shape[-1], 1, 1)
    conv_weight = weight.reshape(list(weight.shape) + [1, 1])
    if sumin is not None:
        sumin = sumin.reshape(-1, sumin.shape[-1], 1, 1)
    out, dequant_out_scale = _conv2d(
        input=conv_input,
        weight=conv_weight,
        bias=bias,
        sumin=sumin,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        padding_mode="zeros",
        activation=activation,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        input_dtype=input_dtype,
        weight_scale=weight_scale,
        weight_zero_point=torch.zeros_like(weight_scale).to(torch.long),
        weight_dtype=weight_dtype,
        bias_scale=bias_scale,
        bias_zero_point=bias_zero_point,
        bias_dtype=bias_dtype,
        sumin_scale=sumin_scale,
        sumin_zero_point=sumin_zero_point,
        sumin_dtype=sumin_dtype,
        scale=scale,
        zero_point=zero_point,
        dtype=dtype,
        march=march,
    )
    out_shape = list(input.shape)[:-1] + [weight.shape[0]]
    return out.reshape(out_shape), dequant_out_scale


def _rle(input: Tensor, dtype: torch.dtype) -> List[Tensor]:
    assert not input.is_floating_point(), "rle only works on int values"
    assert (
        dtype == torch.int8 or dtype == torch.int16
    ), "Only support torch.int8 or torch.int16 dtype in rle"
    flatten_input = input.flatten(start_dim=1).cpu()

    min = -128 if dtype == torch.int8 else -32768
    max = 127 if dtype == torch.int8 else 32767
    max_num = max - min
    assert (
        input.min() >= min and input.max() <= max
    ), "input data range exceeds {} range".format(dtype)
    result: List[Tensor] = []
    N, len = flatten_input.size()
    for i in range(N):
        num = 0
        src_index = 0
        per_batch_result: List[int] = []
        # process per batch
        while src_index < len:
            repeat = 1
            data = flatten_input[i][src_index].item()

            # get the repeat times of data
            src_index += 1
            while src_index < len and flatten_input[i][src_index] == data:
                src_index += 1
                repeat += 1

            # process repeat times exceed max_num limit
            while repeat > max_num:
                per_batch_result.append(data)
                per_batch_result.append(max_num)
                repeat -= max_num
                num += 1
            per_batch_result.append(data)
            per_batch_result.append(repeat)
            num += 1
        # num may larger than 255 or 65535
        result.append(torch.tensor([num] + per_batch_result))

    return result


def _generate_warp_coord_for_deform_conv(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    device: torch.device,
):
    kernel_coord = torch.arange(
        0, kernel_size * dilation, dilation, dtype=torch.int16, device=device
    )
    conv_coord = torch.arange(
        -padding,
        input_size + padding - kernel_coord[-1],
        stride,
        dtype=torch.int16,
        device=device,
    )
    abs_coord = (
        conv_coord.reshape(-1, 1) + kernel_coord.reshape(1, -1)
    ).flatten()

    return abs_coord


def _deform_conv2d(
    input: Tensor,
    offset: Tensor,
    mask: Optional[Tensor],
    sumin: Optional[Tensor],
    weight: Tensor,
    bias: Tensor,
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    offset_scale: Tensor,
    offset_zero_point: Tensor,
    offset_dtype: str,
    mask_scale: Optional[Tensor],
    mask_zero_point: Optional[Tensor],
    mask_dtype: Optional[str],
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    device = input.device
    batch_size, in_channel, in_h, in_w = (
        input.size(0),
        input.size(1),
        input.size(2),
        input.size(3),
    )
    out_h, out_w = offset.size(2), offset.size(3)
    kernel_size = weight.size(2), weight.size(3)
    groups = in_channel // weight.size(1)
    out_channel = weight.size(0)

    # [(y, x), out_h * kernel_h, out_w * kernel_w]
    base_grid = torch.stack(
        torch.meshgrid(
            _generate_warp_coord_for_deform_conv(
                in_h,
                kernel_size[0],
                stride[0],
                padding[0],
                dilation[0],
                device,
            ),
            _generate_warp_coord_for_deform_conv(
                in_w,
                kernel_size[1],
                stride[1],
                padding[1],
                dilation[1],
                device,
            ),
            # ignore indexing because 1.9.1 do not have this param
            # but in the future the default behaviour will be changed to "xy"
            # indexing="ij",
        ),
        dim=0,
    )

    # Compute coord_shift for warp.
    h = in_h if in_h > out_h else out_h
    w = in_w if in_w > out_w else out_w

    max_coord = h if h > w else w
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = min(coord_shift, 8)
    coord_shift = coord_shift if coord_shift > 0 else 0

    grid_scale = torch.tensor(
        1.0 / (1 << coord_shift),
        dtype=torch.float,
        device=base_grid.device,
    ).reshape(1)

    # [batch_size, offset_group, (y, x), out_h * kernel_h, out_w * kernel_w]
    offset = (
        offset.reshape(
            offset.size(0),
            -1,
            kernel_size[0],
            kernel_size[1],
            2,
            out_h,
            out_w,
        )
        .permute(0, 1, 4, 5, 2, 6, 3)
        .reshape(
            offset.size(0),
            -1,
            2,
            out_h * kernel_size[0],
            out_w * kernel_size[1],
        )
    )
    offset_group = offset.size(1)

    # [batch_size, offset_group, (y, x), out_h * kernel_h, out_w * kernel_w]
    grid = _add(
        base_grid,
        offset,
        torch.ones_like(offset_scale),
        offset_scale,
        offset_zero_point,
        offset_zero_point,
        "qint16",
        offset_dtype,
        grid_scale,
        offset_zero_point,
        "qint16",
        march,
    )

    # [batch_size * offset_group, in_channel,
    # out_h * kernel_h, out_w * kernel_w]
    feature = torch.ops.changan.gpu_quanti_grid_sample(
        input,
        grid,
        "bilinear",
        "zeros",
        True,
        coord_shift,
        march,
    )

    if offset_group > 1:
        feature = feature.reshape(
            batch_size,
            offset_group,
            offset_group,
            in_channel // offset_group,
            out_h * kernel_size[0],
            out_w * kernel_size[1],
        )
        feature = torch.cat(
            [feature[:, i, i, :, :, :] for i in range(offset_group)], dim=1
        )

    output, out_scale = _conv2d(
        feature,
        weight,
        bias,
        None if sumin is None else sumin,
        kernel_size,  # stride
        (0, 0),  # padding
        (1, 1),  # dilation
        groups,
        "zeros",
        activation,
        input_scale,
        input_zero_point,
        input_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        input_scale,  # bias_scale,
        input_zero_point,  # bias_zero_point,
        input_dtype,  # bias_dtype,
        sumin_scale,
        sumin_zero_point,
        sumin_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )

    return output, out_scale


def _voxelization(
    points: Tensor,
    voxel_size: Tensor,
    pc_range: Tensor,
    max_voxels: int,
    max_points_per_voxel: int,
    use_max: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert points(N, >=3) to voxels.

    Args:
        points: (N, ndim), points[:, :3] contain xyz points
            and points[:, 3:] contain other information such as reflectivity.
        voxel_size: (3,), xyz, indicate voxel size.
        pc_range: (6,), indicate voxel range, format:
            [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points: Indicate maximum points contained in a voxel.
        max_voxels: Indicate maximum voxels this function create.
            you should shuffle points before call this function because
            max_voxels may drop some points.

    Returns:
        voxels: (M, max_points, ndim) Only contain points.
        coordinates: (M, 3) coordinates in zyx format.
        num_points_per_voxel: (M,) Number of points in per voxel.
    """

    ndim = 3

    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size  # (x, y, z)
    grid_size = torch.round(grid_size).type(torch.int32)
    voxel_map_shape: List[int] = grid_size.type(torch.int64).tolist()
    voxel_map_shape = voxel_map_shape[::-1]  # (z, y, x)

    voxels = torch.zeros(
        (max_voxels, max_points_per_voxel, points.shape[-1]),
        dtype=points.dtype,
    )
    coors = torch.zeros(
        (max_voxels, 3),
        dtype=torch.int32,
    )
    num_points_per_voxel = torch.zeros((max_voxels,), dtype=torch.int32)

    voxel_num = 0
    voxel_num = torch.ops.changan.voxelization(
        points.cpu(),
        voxel_size.cpu(),
        pc_range.cpu(),
        voxels.cpu(),
        coors.cpu(),
        num_points_per_voxel.cpu(),
        voxel_num,
        max_points_per_voxel,
        max_voxels,
        ndim,
    )

    if use_max:
        # for deploy, use max_voxels
        out_num = max_voxels
    else:
        out_num = voxel_num

    coors = coors[:out_num].to(points.device)
    voxels = voxels[:out_num].to(points.device)
    num_points_per_voxel = num_points_per_voxel[:out_num].to(points.device)
    return (voxels, coors, num_points_per_voxel)


def _point_pillars_preprocess(
    points_list: List[Tensor],
    pc_range: Tensor,
    voxel_size: Tensor,
    max_voxels: int,
    max_points_per_voxel: int,
    use_max: bool,
) -> Tuple[Tensor, Tensor]:

    voxel_lst: List[Tensor] = []
    coors_lst: List[Tensor] = []
    num_points_per_voxel_lst: List[Tensor] = []
    for points in points_list:
        # voxelize per points, for batch_size > 1
        voxels, coors, num_points_per_voxel = _voxelization(
            points,
            voxel_size=voxel_size,
            pc_range=pc_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
            use_max=use_max,
        )
        voxel_lst.append(voxels)
        coors_lst.append(coors)
        num_points_per_voxel_lst.append(num_points_per_voxel)

    voxel_feature = torch.cat(voxel_lst, dim=0)
    num_points_per_voxel = torch.cat(num_points_per_voxel_lst, dim=0)

    # Pad first element of coord according the index in batch_data.
    # Example:
    #   batch_data = [data1, data2], and batch_size = 2,
    #   batch_data.index(data1) = 0, batch_data.index(data2) = 1,
    #   for data1:  coord (z, y, x) --> Pad 0 --> coord (0, z, y, x)
    #   for data2:  coord (z, y, x) --> Pad 1 --> coord (1, z, y, x)
    coors_batch: List[Tensor] = []
    for i, coor in enumerate(coors_lst):
        coor_pad = F.pad(coor, (1, 0), mode="constant", value=float(i))
        coors_batch.append(coor_pad)
    coors_batch = torch.cat(coors_batch, dim=0).long()

    features = _voxel_feature_encoder(
        pc_range=pc_range,
        features=voxel_feature,
        num_points_in_voxel=num_points_per_voxel,
    )

    return features, coors_batch


def _get_paddings_indicator(
    actual_num: Tensor, max_num: int, axis: int = 0
) -> Tensor:
    """Create boolean mask by actual number of a padded tensor.

    This function helps to identify pillars where there's too little data.

    Example:

    actual_num = [[3,3,3,3,3]] (5 pillars, each contains 3 lidar points)
    max_num: 4 (turns to [[0, 1, 2, 3, 4]])
    will return: [[T, T, T, F, F]]

    Args:
        actual_num: (N,M), where N is batch size and M is
            total number of pillars. In certain cases N can be omitted.
        max_num: max number of points allowed in a pillar.
        axis: axis position. Defaults to 0.

    Returns:
        paddings_indicator: indicates where the tensor should be padded.
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape: List[int] = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device
    ).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


def _voxel_feature_encoder(
    pc_range: Tensor, features: Tensor, num_points_in_voxel: Tensor
) -> Tensor:

    pc_start = pc_range[:3]
    pc_len = pc_range[3:6] - pc_range[:3]
    features[:, :, :3] = features[:, :, :3] - pc_start
    features[:, :, :3] = features[:, :, :3] / pc_len

    # The feature decorations were calculated without regard to whether
    # pillar was empty. Need to ensure that empty pillars remain set to
    # zeros.
    voxel_count = features.shape[1]
    mask = _get_paddings_indicator(num_points_in_voxel, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask

    features = features.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

    return features
