r""" Functional interface (quantized).
This file should include and only include interface functions. All functions
should not include any sepcific implementation but only call the corresponding
functions in .functional_impl.py. All inferface function defined in this file
should decoratored with @with_march and @script_quantized_fn.
"""
import torch
from changan_plugin_pytorch.march import with_march
from torch import Tensor
from torch.jit.annotations import (
    BroadcastingList2,
    BroadcastingList3,
    List,
    Optional,
    Tuple,
)

from ...utils.script_quantized_fn import script_quantized_fn
from .functional_impl import (
    _add,
    _avg_pool2d,
    _base_grid_generator,
    _cat,
    _channel_shuffle,
    _conv2d,
    _conv3d,
    _conv_convert_int_params,
    _conv_transpose2d,
    _correlation,
    _dequantize,
    _detection_post_process_v1,
    _filter,
    _get_multi_table_params_impl,
    _grid_sample,
    _grid_sample_norm_grid,
    _interpolate,
    _linear,
    _lut,
    _masked_fill,
    _matmul,
    _max,
    _max_pool2d,
    _mean,
    _mul,
    _multi_scale_roi_align,
    _multi_table_fit_impl,
    _pad,
    _point_pillars_scatter,
    _prelu,
    _quantize,
    _requantize,
    _roi_align_list,
    _roi_align_tensor,
    _segment_lut,
    _softmax,
    _sub,
    _sum,
    _window_partition,
    _window_reverse,
    _rle,
    _deform_conv2d,
    _point_pillars_preprocess,
)

__all__ = [
    "quantize",
    "dequantize",
    "requantize",
    "conv2d_convert_int_params",
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
    "interpolate",
    "pad",
    "roi_align_list",
    "roi_align_tensor",
    "cap",
    "add",
    "grid_sample",
    "grid_sample_norm_grid",
    "filter",
    "max",
    "sub",
    "lut",
    "get_multi_table_params",
    "multi_table_fit",
    "matmul",
    "base_grid_generator",
    "mul",
    "sum",
    "softmax",
    "detection_post_process_v1",
    "conv_transpose2d",
    "multi_scale_roi_align",
    "segment_lut",
    "point_pillars_scatter",
    "channel_shuffle",
    "prelu",
    "window_partition",
    "window_reverse",
    "linear",
    "rle",
    "point_pillars_preprocess",
]


@with_march
@script_quantized_fn
def quantize(
    input: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    ch_axis: int,
    dtype: str,
    march: str,
) -> Tensor:
    """
    scale quanti input

    Arguments:
        input (Tensor[N, C, H, W]): input tensor
        scale(Tensor[1/C]): scale for quantization
        zero_point(Tensor[1/C]): zero point
        dtype(str): quantize type
        march(str): Bpu version

    Returns:
        output (Tensor[N, C, H, W])
    """
    return _quantize(input, scale, zero_point, dtype, march)


@with_march
@script_quantized_fn
def dequantize(
    input: Tensor, scale: Tensor, zero_point: Tensor, ch_axis: int, march: str
) -> Tensor:
    """
    scale dequantization

    Arguments:
        input (Tensor[N, C, H, W]): input tensor
        scale(Tensor[1] or Tensor[C]): scale for quantization
        zero_point(Tensor[1] or Tensor[C]): zero point for quantization

    Returns:
        output (Tensor[N, C, H, W])
    """
    return _dequantize(input, scale, zero_point, ch_axis, march)


@with_march
@script_quantized_fn
def requantize(
    input: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    """
    requantize on bpu.

    Args:
        input (Tensor[N, C, ...]): Input data.
        input_scale (Tensor[1] or Tensor[C]): The scale of input data.
        input_zero_point (Tensor[1] or Tensor[C]):
            The zero point of input data.
        input_quanti_type (str): Quanti type of input data.
        scale (Tensor[1] or Tensor[C]): The scale of output data.
        zero_point (Tensor[1] or Tensor[C]): The zero point of output data.
        type (str): Quanti type of output data.
        march (str): Bpu version.

    Returns:
        Tensor[N, C, ...]: Output.
    """
    return _requantize(
        input,
        input_scale,
        input_zero_point,
        input_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def conv2d_convert_int_params(
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
    """Convert quantized parameters from qat float params.

    Args:
        input_scale (Tensor): Scale of input feature.
        weight (Tensor): Weight in float type.
        weight_scale (Tensor): Scale of weight.
        weight_dtype (str): Quanti type of weight.
        bias (Tensor): Bias in float type.
        bias_scale (Tensor): Bias scale. Only used in Bernoulli.
        bias_dtype(Tensor): Bias dtype. Only used in Bernoulli.
        out_scale (Tensor): Scale of output activations.
        out_dtype (str, optional): Quanti type of output activations.
            Defaults to "qint32".
        input2_scale (Optional[Tensor], optional): Scale of the other input.
            Defaults to None.
        is_conv_transpose2d (bool, optional): Defaults to False.
        groups (int, optional): Convolution groups. Defaults to 1.
        march (str, optional): Bpu march. Defaults to March.BAYES.

    Returns:
        Tuple[ Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor ]:
        For BERNOULLI:
            bpu_weight
            bpu_weight_shift
            bpu_bias
            bpu_bias_shift
            bpu_input_shift
            bpu_output_shift
            bpu_edata_shift
            dequant_output_scale
            Tensor()
        --------------------
        For BERNOULLI2/BAYES:
            bpu_weight
            bpu_bias
            bpu_bias_lshift
            bpu_escale
            bpu_escale_lshift
            bpu_oscale
            bpu_accu_rshift
            bpu_output_rshift
            dequant_output_scale
    """
    return _conv_convert_int_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        out_scale,
        out_dtype,
        input2_scale,
        is_conv_transpose2d,
        groups,
        march,
    )


@with_march
@script_quantized_fn
def conv2d(
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
    """
    scale quanti convolution

    Arguments:
        input: input tensor
        weight: weight tensor
        bias: bias tensor
        sumin: elementwise data tensor
        stride: stride size
        padding: pad size
        dilation: dilation rate
        groups: group number
        padding_mode: padding mode, only support zeros
        activation: activation string, only support relu
        input_scale: scale of input.
        input_zero_point: zero point of input.
        input_dtype: dtype of input.
        weight_scale: weight scale tensor.
        weight_zero_point: weight zero point.
        weight_dtype: weight quant-dtype.
        bias_scale: scale of bias. Only used in Bernoulli.
        bias_zero_point: zero point of bias. Only used in Bernoulli.
        bias_dtype: dtype of bias. Only used in Bernoulli.
        sumin_scale: elementwise-add scale.
        sumin_zero_point: elementwise-add zero_point.
        sumin_dtype: elementwise-add quant-dtype.
        scale: out scale.
        zero_point: out quant zero point.
        dtype: out quant-dtype.
        march: march of bpu

    Returns:
        output
        dequant_out_scale
    """
    return _conv2d(
        input,
        weight,
        bias,
        sumin,
        stride,
        padding,
        dilation,
        groups,
        padding_mode,
        activation,
        input_scale,
        input_zero_point,
        input_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        bias_scale,
        bias_zero_point,
        bias_dtype,
        sumin_scale,
        sumin_zero_point,
        sumin_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def conv3d(
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
    """
    scale quanti convolution

    Arguments:
        input: input tensor
        weight: weight tensor
        bias: bias tensor
        sumin: elementwise data tensor
        stride: stride size
        padding: pad size
        dilation: dilation rate
        groups: group number
        padding_mode: padding mode, only support zeros
        activation: activation string, only support relu
        input_scale: scale of input.
        input_zero_point: zero point of input.
        input_dtype: dtype of input.
        weight_scale: weight scale tensor.
        weight_zero_point: weight zero point.
        weight_dtype: weight quant-dtype.
        bias_scale: scale of bias. Only used in Bernoulli.
        bias_zero_point: zero point of bias. Only used in Bernoulli.
        bias_dtype: dtype of bias. Only used in Bernoulli.
        sumin_scale: elementwise-add scale.
        sumin_zero_point: elementwise-add zero_point.
        sumin_dtype: elementwise-add quant-dtype.
        scale: out scale.
        zero_point: out quant zero point.
        dtype: out quant-dtype.
        march: march of bpu

    Returns:
        output
        dequant_out_scale
    """
    return _conv3d(
        input,
        weight,
        bias,
        sumin,
        stride,
        padding,
        dilation,
        groups,
        padding_mode,
        activation,
        input_scale,
        input_zero_point,
        input_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        bias_scale,
        bias_zero_point,
        bias_dtype,
        sumin_scale,
        sumin_zero_point,
        sumin_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    return_indices: bool,
    ceil_mode: bool,
    # input_scale: Tensor,
    # input_zero_point: Tensor,
    march: str,
) -> Tensor:
    return _max_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
        march,
    )


@with_march
@script_quantized_fn
def avg_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: None,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    return _avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        input_scale,
        input_zero_point,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def interpolate(
    input: Tensor,
    size: Optional[BroadcastingList2[int]],
    scale_factor: Optional[BroadcastingList2[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    march: str,
) -> Tensor:
    return _interpolate(
        input,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        march,
    )


@with_march
@script_quantized_fn
def pad(
    input: Tensor,
    pad: List[int],
    mode: str,
    value: float,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    return _pad(input, pad, mode, value, scale, zero_point, dtype, march)


@with_march
@script_quantized_fn
def masked_fill(
    input: Tensor,
    mask: Tensor,
    value: float,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    return _masked_fill(input, mask, value, scale, zero_point, dtype, march)


@with_march
@script_quantized_fn
def roi_align_list(
    input: Tensor,
    boxes: List[Tensor],
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    march: str,
) -> Tensor:
    """
    Quantized version of Roi align function.
    Same as torchvision.ops.roi_align, except that the output size is
    sampling_ratio times larger and boxes must be List[Tensor].

    Please not that in our underlying implementation the roi is batched into
    Tensor[n, k, 4], which is different with torchvision (Tensor[k, 5], and the
    first element is batch index). So we have to do some extra pre-process
    and post-process.

    For bayes, we expect input rois in float type. But for bernoulli2, rois
    should be int32 and with scale of 0.25 (The output of
    DetectionPostProcessV1).
    """
    return _roi_align_list(
        input,
        boxes,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
        interpolate_mode,
        march,
    )


@with_march
@script_quantized_fn
def roi_align_tensor(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    march: str,
) -> Tensor:
    """
    Quantized version of Roi align function.
    Same as torchvision.ops.roi_align, except that
    we only support sampling_ratio = 1 and Tensor input for boxes.

    Please not that in our underlying implementation the roi is batched into
    Tensor[n, k, 4], which is different with torchvision (Tensor[k, 5], and the
    first element is batch index). So we have to do some extra pre-process
    and post-process.
    """
    return _roi_align_tensor(
        input,
        boxes,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
        interpolate_mode,
        march,
    )


@with_march
@script_quantized_fn
def cap(
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
    """
    Concat on bpu, same as torch.cat

    Args:
        input_list (List[Tensor[N, C, ...]]): List of tensors of the same type.
            Non-empty tensors provided must have the same shape, except in
            the cap dimension.
        dim (int): The dimension over which the tensors are concatenated.
        input_scale_list (List[Tensor[1] or Tensor[C]]): Scale of input.
        input_zero_point_list (List[Tensor[1] or Tensor[C]]):
            Zero point of input.
        input_dtype_list (List[str]): Quant dtype of input.
        scale (Tensor[1]): Output scale.
        zero_point (Tensor[1]): Output zero point.
        dtype (str): Output quant dtype.
        march (str): Bpu version.

    Returns:
        Tensor: [N, C, ...]
    """
    return _cat(
        input_list,
        dim,
        input_scale_list,
        input_zero_point_list,
        input_dtype_list,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def add(
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
    """
    Add on bpu, same as torch.add

    Args:
        x (Tensor): First tensor for add
        y (Tensor): Second tensor for add
        x_scale (Tensor): Scale of first input
        y_scale (Tensor): Scale of second input
        x_zero_point (Tensor): Zero point of first input
        y_zero_point (Tensor): Zero point of Second input
        x_dtype (str): Quant dtype of first input
        y_dtype (str): Quant dtype of second input
        scale (Tensor): Output Scale
        zero_point: (Tensor): Output zero point
        dtype (str): Output quant dtype
        march (str): Bpu version

    Returns:
        Tensor: [N, C, ...]
    """
    return _add(
        x,
        y,
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


@with_march
@script_quantized_fn
def grid_sample(
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
    return _grid_sample(
        input,
        grid,
        mode,
        padding_mode,
        align_corners,
        grid_scale,
        grid_zero_point,
        grid_dtype,
        march,
    )


@with_march
@script_quantized_fn
def grid_sample_norm_grid(
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
    return _grid_sample_norm_grid(
        input,
        grid,
        mode,
        padding_mode,
        align_corners,
        grid_scale,
        grid_zero_point,
        grid_dtype,
        march,
    )


@with_march
@script_quantized_fn
def filter(
    inputs: List[Tensor],
    scales: List[Tensor],
    zero_points: List[Tensor],
    dtypes: List[str],
    threshold: float,
    idx_range: Tuple[int, int],
    march: str,
) -> List[List[Tensor]]:
    return _filter(
        inputs, scales, zero_points, dtypes, threshold, idx_range, march
    )


@with_march
@script_quantized_fn
def max(
    input: Tensor,
    dim: int,
    keepdim: bool,
    group: int,
    march: str,
) -> Tuple[Tensor, Tensor]:
    return _max(input, dim, keepdim, group, march)


@with_march
@script_quantized_fn
def sub(
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
    """
    Sub on bpu, same as torch.sub

    Args:
        x (Tensor): Left value for sub
        y (Tensor): Right value for sub
        x_scale (Tensor): Scale of first input
        x_zero_point (Tensor): Zero point of first input
        x_dtype (str): Quant dtype of first input
        y_scale (Tensor): Scale of second input
        y_zero_point (Tensor): Zero point of Second input
        y_dtype (str): Quant dtype of second input
        scale (Tensor): Output Scale
        zero_point: (Tensor): Output zero point
        dtype (str): Output quant dtype
        march (str): Bpu version

    Returns:
        Tensor: [N, C, ...]
    """
    return _sub(
        x,
        y,
        x_scale,
        x_zero_point,
        x_dtype,
        y_scale,
        y_zero_point,
        y_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def lut(
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
    return _lut(
        data,
        data_scale,
        data_zero_point,
        data_type,
        table,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def get_multi_table_params(
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
    return _get_multi_table_params_impl(
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
        march,
    )


@with_march
@script_quantized_fn
def multi_table_fit(
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
    return _multi_table_fit_impl(
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
        dtype,
        is_symmetric,
        symmetric_k,
        symmetric_b,
        march,
    )


@with_march
@script_quantized_fn
def matmul(
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
    """
    Matmul on bpu, same as torch.matmul

    Args:
        input (Tensor[..., N, K]): First tensor for matmul
        other (Tensor[..., K, M]): Second tensor for matmul
        input_scale (Tensor): Scale of first input
        input_zero_point (Tensor): Zero point of first input
        input_dtype (str): Quant dtype of first input
        other_scale (Tensor): Scale of second input
        other_zero_point (Tensor): Zero point of Second input
        other_dtype (str): Quant dtype of second input
        scale (Tensor): Output Scale
        zero_point: (Tensor): Output zero point
        dtype (str): Output quant dtype
        march (str): Bpu version

    Returns:
        Tensor: [N, C, ...]
    """
    return _matmul(
        input,
        other,
        input_trans,
        other_trans,
        input_scale,
        input_zero_point,
        input_dtype,
        other_scale,
        other_zero_point,
        other_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def base_grid_generator(
    size: BroadcastingList2[int],
    with_third_channel: bool,
    device: torch.device,
    march: str,
) -> Tensor:
    """
    Generate base grid for affine transform or perspective transform.

    Args:
        size (BroadcastingList2[int]): Output size.
        with_third_channel (bool): Whether append the all ones third channel.
        device (torch.device): The device of output tensor.

    Returns:
        Tensor: Base grid.
    """
    return _base_grid_generator(size, with_third_channel, device, march)


@with_march
@script_quantized_fn
def mul(
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
    """
    Mul on bpu, same as torch.mul

    Args:
        input (Tensor): First input for mul
        other (Tensor): Second input for mul
        input_scale (Tensor): Scale of first input
        input_zero_point (Tensor): Zero point of first input
        input_dtype (str): Quant dtype of first input
        other_scale (Tensor): Scale of second input
        other_zero_point (Tensor): Zero point of Second input
        other_dtype (str): Quant dtype of second input
        scale (Tensor): Output Scale
        zero_point: (Tensor): Output zero point
        dtype (str): Output quant dtype
        march (str): Bpu version

    Returns:
        Tensor: [N, C, ...]
    """
    return _mul(
        input,
        other,
        input_scale,
        input_zero_point,
        input_dtype,
        other_scale,
        other_zero_point,
        other_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def sum(
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
    """
    Sum on bpu, same as torch.sum

    Args:
        x (Tensor): Input.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether the output tensor has dim retained or not.
        x_scale (Tensor): Scale of input.
        x_zero_point (Tensor): Zero point of input.
        x_dtype (str): Quantization type of input.
        scale (Tensor): Scale of input.
        zero_point (Tensor): Zero point of input.
        dtype (str): Quantization type of input.
        march (str): Bpu version.

    Returns:
        Tensor
    """
    return _sum(
        x,
        dim,
        keepdim,
        x_scale,
        x_zero_point,
        x_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def softmax(
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

    """
    quantization softmax along channel axis

    Arguments:
        data(Tensor[N, C, H, W]): input tensor.
        data_scale(Tensor): scale of input tensor.
        data_zero_point(Tensor): zero point of input.
        data_type(str): input data type.
        scale(Tensor): output data scale.
        zero_point(Tensor): zero point of output.
        dtype(str): data type of output.
        exp_out_scale(Tensor): scale of exp table.
        exp_out_zero_point(Tensor): zero point of exp table.
        exp_out_dtype(str): data type of exp table.
        reciprocal_out_scale(Tensor): scale of reciprocal table.
        reciprocal_out_zero_point(Tensor): zero point of reciprocal table.
        reciprocal_out_type(str): data type of reciprocal table.
        exp_dense_table(Tensor): dense input table calculated by exp function.
        exp_qint_dense_xmin(Tensor): left boundary of input mapping to dense
            table.
        exp_qint_dense_xmax: Tensor: right boundary of input mapping to dense
            table.
        exp_sparse_table(Tensor): sparse input table calculated by exp function
        exp_qint_sparse_xmin(Tensor): left boundary of input mapping to sparse
            table.
        exp_qint_sparse_xmax(Tensor): right boundary of input mapping to sparse
            table.
        exp_left_line_xmin(Tensor): left boundary of exp left linear fitting.
        exp_left_line_ymin(Tensor): the value of exp function in the left
            bound.
        exp_left_line_xmax(Tensor): right boundary of exp left linear fitting
        exp_left_line_ymax(Tensor): the value of exp function in the right
            bound.
        exp_right_line_xmin(Tensor): left boundary of exp right linear
            fitting.
        exp_right_line_ymin(Tensor): the value of exp function in the left
            boundary of exp right linear fitting.
        exp_right_line_xmax(Tensor): right boundary of exp right linear
            fitting.
        exp_right_line_ymax(Tensor): the value of exp function in the right
            boundary of exp right linear fitting.
        exp_qint_left_constant_xmin(Tensor): qint left boundary of left exp
            constant fitting.
        exp_qint_left_constant_xmax(Tensor): qint right boundary of left exp
            constant fitting.
        exp_qint_right_constant_xmin(Tensor): qint left boundary of right exp
            constant fitting.
        exp_qint_right_constant_xmax(Tensor): qint right boundary of right exp
            constant fitting.
        exp_left_constant_fit_y(Tensor): constant of left exp constant fitting.
        exp_right_constant_fit_y(Tensor): constant of right exp constant
            fitting.
        rescale_shift(Tensor): rescale coefficient of exp out to exp sum
        reciprocal_dense_table(Tensor): a table of reciprocal of dense input.
        reciprocal_qint_dense_xmin(Tensor): qint left boundary of input mapping
            to dense reciprocal table.
        reciprocal_qint_dense_xmax(Tensor): qint right boundary of input
            mapping to dense table.
        reciprocal_sparse_table(Tensor): a table of reciprocal of sparse input.
        reciprocal_qint_sparse_xmin(Tensor): qint left boundary of input
            mapping to sparse reciprocal tabel.
        reciprocal_qint_sparse_xmax(Tensor): qint right boundary of input
            mapping to sparse table.
        reciprocal_left_line_xmin(Tensor): left boundary of reciprocal
            left linear fitting.
        reciprocal_left_line_ymin(Tensor): reciprocal of left boundary
            of reciprocal left linear fitting.
        reciprocal_left_line_xmax(Tensor): right boundary of reciprocal
            left linear fitting.
        reciprocal_left_line_ymax(Tensor): reciprocal of right boundary of
            reciprocal left linear fitting.
        reciprocal_right_line_xmin(Tensor): left boundary of reciprocal right
            linear fitting.
        reciprocal_right_line_ymin(Tensor): reciprocal of left boundary of
            reciprocal right linear fitting
        reciprocal_right_line_xmax(Tensor): right boundary of reciprocal right
            linear fitting.
        reciprocal_right_line_ymax(Tensor): reciprocal of right boundary of
            reciprocal right linear fitting.
        reciprocal_qint_left_constant_xmin(Tensor): qint left boundary of left
            reciprocal linear fitting.
        reciprocal_qint_left_constant_xmax(Tensor): qint right boundary of
            left reciprocal linear fitting.
        reciprocal_qint_right_constant_xmin(Tensor): qint left boundary of
            right reciprocal constant fitting.
        reciprocal_qint_right_constant_xmax(Tensor): qint right boundary of
            right reciprocal constant fitting.
        reciprocal_left_constant_fit_y(Tensor): constant of reciprocal in left
            reciprocal constant fitting.
        reciprocal_right_constant_fit_y(Tensor): constant of reciprocal in
            right reciprocal constant fitting
        march(str): Bpu version.

    Returns:
        output (Tensor[N, C, H, W])
    """
    return _softmax(
        data,
        data_scale,
        data_zero_point,
        data_type,
        scale,
        zero_point,
        dtype,
        exp_out_scale,
        exp_out_zero_point,
        exp_out_type,
        reciprocal_out_scale,
        reciprocal_out_zero_point,
        reciprocal_out_type,
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
        rescale_shift,
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
        march,
    )


@with_march
@script_quantized_fn
def detection_post_process_v1(
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
    return _detection_post_process_v1(
        data,
        anchor,
        exp_table,
        image_sizes,
        num_classes,
        input_shifts,
        exp_shift,
        box_filter_threshold,
        class_offsets,
        seed,
        use_clippings,
        nms_threshold,
        nms_margin,
        post_nms_top_k,
        march,
    )


@with_march
@script_quantized_fn
def conv_transpose2d(
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
    """
    scale quanti deconvolution

    Arguments:
        input: input tensor
        weight: weight tensor
        bias: bias tensor
        sumin: elementwise data tensor
        stride: stride size
        padding: pad size
        output_padding: output_padding size
        dilation: dilation rate
        groups: group number
        padding_mode: padding mode, only support zeros
        activation: activation string, only support relu
        input_scale: scale of input.
        input_zero_point: zero point of input.
        input_dtype: dtype of input.
        weight_scale: weight scale tensor.
        weight_zero_point: weight zero point.
        weight_dtype: weight quant-dtype.
        bias_scale: bias scale tensor. Only used in Bernoulli
        bias_zero_point: bias zero point. Only used in Bernoulli
        bias_dtype: bias dtype. Only used in Bernoulli
        sumin_scale: elementwise-add scale.
        sumin_zero_point: elementwise-add zero_point.
        sumin_dtype: elementwise-add quant-dtype.
        scale: out scale.
        zero_point: out quant zero point.
        dtype: out quant-dtype.
        march: march of bpu

    Returns:
        output
        dequant_out_scale
    """

    # convert to int params
    return _conv_transpose2d(
        input,
        weight,
        bias,
        sumin,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        padding_mode,
        activation,
        input_scale,
        input_zero_point,
        input_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        bias_scale,
        bias_zero_point,
        bias_dtype,
        sumin_scale,
        sumin_zero_point,
        sumin_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def multi_scale_roi_align(
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
    return _multi_scale_roi_align(
        features,
        boxes,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
        interpolate_mode,
        canonical_box_size,
        canonical_level,
        march,
    )


@with_march
@script_quantized_fn
def correlation(
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
    """
    Scale quanti correlation

    Args:
        data1: feature1
        data2: feature2
        kernel_size: kernel_size for correlation
        max_displacement: max_displacement for correlation
        stride1: patch stride of data1
        stride2: patch stride of data2 within neighborhood centered data1
        pad_size: pad size for data1 and data2
        is_multiply: operation type is either multiplication or subduction
        scale1: scale of data1
        zero_point1: zero_point of data1
        dtype1: dtype of data1
        scale2: scale of data2
        zero_point2: zero_point of data2
        dtype2: dtype of data2
        inter_scale: scale of channel sum of elementwise_mul_add inter output
        out_scale: scale of output
        out_zero_point: zero point of output
        out_dtype: dtype of output
        march: march of bpu

    Return:
        out
    """
    return _correlation(
        data1,
        data2,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        pad_size,
        is_multiply,
        scale1,
        zero_point1,
        dtype1,
        scale2,
        zero_point2,
        dtype2,
        inter_scale,
        out_scale,
        out_zero_point,
        out_dtype,
        march,
    )


@with_march
@script_quantized_fn
def mean(
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
    """
    mean on bpu, same as torch.mean

    Args:
        x(Tensor): Input
        dim(int): The dimension to reduce
        x_scale(Tensor): Scale of input
        x_zero_point(Tensor): Zero point of input
        x_dtype(str): Quantization type of input
        scale(Tensor): Scale of output
        zero_point(Tensor): Zero point of output
        dtype(str): Quantization type of output
        march(str): Bpu version

    Returns:
        Tensor
    """
    return _mean(
        x,
        dim,
        x_scale,
        x_zero_point,
        x_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@with_march
@script_quantized_fn
def segment_lut(
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
    """
    Segment Look Up Table, implemented by spliting input range into 8 segments
    which include 2 linear fit segments and 6 LUT segments.

    Args:
        input (Tensor): Input data, only support qint16.
        table (Tensor[6, 64]): Table values for LUT segments.
        scale (Tensor[8]): Scale for input of each segment.
        beta (Tensor[8]): Bias for input of each segment.
        left_shift (Tensor[8]): Left shift for beta of each segment.
        right_shift (Tensor[8]): Right shift of each segment.
        max (Tensor[8]): Max input value of each segment.
        is_centrosymmetric (bool): If True, output of negative input
            is computed by -F(-x).
        input_scale (Tensor): Input scale.
        input_zero_point (Tensor): Input zero point.
        input_dtype (str): Input dtype.
        scale (Tensor): Out scale.
        zero_point (Tensor): Out zero point.
        dtype (str): Output dtype.
        march (str): March of bpu.

    Returns:
        Tensor
    """
    return _segment_lut(
        input,
        table,
        scales,
        beta,
        left_shift,
        right_shift,
        max,
        is_centrosymmetric,
        input_scale,
        input_zero_point,
        input_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@script_quantized_fn
def channel_shuffle(input: Tensor, groups: int):
    return _channel_shuffle(input, groups)


@script_quantized_fn
def point_pillars_scatter(
    voxel_features: Tensor, coords: Tensor, output_shape: List[int]
) -> Tensor:
    """
    Put voxel features into corresponding position and make up a pseudo image.

    Args:
        voxel_features (Tensor):
            [M, ...], dimention after M will be flattened.
        coords (Tensor):
            [M, (n, ..., y, x)], only indices on N, H and W are used.
        output_shape (List[int]): Expected output shape.

    Returns:
        Tensor: The NCHW pseudo image.
    """
    return _point_pillars_scatter(voxel_features, coords, output_shape)


@with_march
@script_quantized_fn
def prelu(
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
    """
    PReLU on BPU.

    Args:
        input(Tensor): input data
        weight(Tensor): weight
        input_scale(Tensor): scale of input
        input_zero_point(Tensor): zero point of input
        input_dtype(str): dtype of input
        weight_scale(Tensor): scale of weight
        weight_zero_point(Tensor): zero point of weight
        weight_dtype(str): dtype of weight
        scale(Tensor): scale of output
        zero_point(Tensor): zero point of output
        dtype(str): dtype of output
        march(str): bpu version
    """
    return _prelu(
        input,
        weight,
        input_scale,
        input_zero_point,
        input_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@script_quantized_fn
def window_partition(x: Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    return _window_partition(x, window_size)


@script_quantized_fn
def window_reverse(windows: Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    return _window_reverse(windows, window_size, H, W)


@with_march
@script_quantized_fn
def linear(
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
    return _linear(
        input,
        weight,
        bias,
        sumin,
        activation,
        input_scale,
        input_zero_point,
        input_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        bias_scale,
        bias_zero_point,
        bias_dtype,
        sumin_scale,
        sumin_zero_point,
        sumin_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@script_quantized_fn
def rle(input: Tensor, dtype: torch.dtype) -> List[Tensor]:
    """
    RLE compression algorithm. Compress the tensor in the format of
    [pair_num, value, num, value, num, ...]

    pair_num: indicates (value, num) pair amount
    value: the compressed data, support int8/16 dtype
    num: amount of the contiguous value. It dtype corresponds to value dtype.
        (int8 value, uint8 num / int16 value, uint16 num)

    Args:
        input(Tensor): The data to be compressed
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
    return _rle(input, dtype)


@with_march
@script_quantized_fn
def deform_conv2d(
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
    """
    quantized deformable convolution2d

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width]):
            input tensor
        offset (Tensor[batch_size,
            offset_groups * kernel_height * kernel_width * 2,
            out_height, out_width]):
            offsets to be applied for each position in the
            convolution kernel.
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
            out_height, out_width]):
            masks to be applied for each position in the convolution kernel.
            Default: None
        sumin: elementwise data tensor
        weight (Tensor[out_channels, in_channels // groups,
            kernel_height, kernel_width]):
            convolution weights
        bias (Tensor[out_channels]):
            optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]):
            distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]):
            height/width of padding of zeroes around each image.
            Default: 0
        dilation (int or Tuple[int, int]):
            the spacing between kernel elements. Default: 1
        activation: activation string, only support relu
        input_scale: scale of input.
        input_zero_point: zero point of input.
        input_dtype: dtype of input.
        offset_scale: scale of offset.
        offset_zero_point: zero point of offset.
        offset_dtype: dtype of offset.
        mask_scale: scale of mask.
        mask_zero_point: zero point of mask.
        mask_dtype: dtype of mask.
        sumin_scale: elementwise-add scale.
        sumin_zero_point: elementwise-add zero_point.
        sumin_dtype: elementwise-add quant-dtype.
        weight_scale: weight scale tensor.
        weight_zero_point: weight zero point.
        weight_dtype: weight quant-dtype.
        scale: out scale.
        zero_point: out quant zero point.
        dtype: out quant-dtype.
        march: march of bpu

    Returns:
        output
        dequant_out_scale
    """
    return _deform_conv2d(
        input,
        offset,
        mask,
        sumin,
        weight,
        bias,
        stride,
        padding,
        dilation,
        activation,
        input_scale,
        input_zero_point,
        input_dtype,
        offset_scale,
        offset_zero_point,
        offset_dtype,
        mask_scale,
        mask_zero_point,
        mask_dtype,
        sumin_scale,
        sumin_zero_point,
        sumin_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )


@script_quantized_fn
def point_pillars_preprocess(
    points_list: List[Tensor],
    pc_range: Tensor,
    voxel_size: Tensor,
    max_voxels: int,
    max_points_per_voxel: int,
    use_max: bool,
) -> Tuple[Tensor, Tensor]:
    return _point_pillars_preprocess(
        points_list,
        pc_range,
        voxel_size,
        max_voxels,
        max_points_per_voxel,
        use_max,
    )
