import math

import torch
import torch.nn.functional as F
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.march import March, get_march, with_march
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import Tensor
from torch.jit.annotations import BroadcastingList2, List, Optional
from torch.nn.modules.utils import _pair


@with_march
def interpolate(
    data: Tensor,
    size: Optional[BroadcastingList2[int]],
    scale_factor: Optional[BroadcastingList2[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    quantized_forward: bool,
    march: str,
):
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
    return torch.ops.changan.quanti_resize(
        data.float(),
        scale.float(),
        zero_point,
        mode,
        align_corners,
        out_height,
        out_width,
        ratio_height,
        ratio_width,
        quantized_forward,
        march,
    )


@with_march
def roi_align_list(
    input: Tensor,
    boxes: List[Tensor],
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    roi_quantized: bool,
    march: str,
) -> Tensor:
    """
    Qat version of Roi align function.
    Same as torchvision.ops.roi_align, except thatthe output size is
    sampling_ratio times larger and boxes must be List[Tensor].

    Please not that in our underlying implementation the roi is batched into
    Tensor[n, k, 4], which is different with torchvision (Tensor[k, 5], and the
    first element is batch index). So we have to do some extra pre-process
    and post-process.
    """
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
            roi_per_image[:, 0] * spatial_scale > input.size(3),
            roi_per_image[:, 1] * spatial_scale > input.size(2),
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

    ret = torch.ops.changan.quanti_roi_resize(
        input.float(),
        batched_rois.float(),
        scale.float(),
        zero_point.float(),
        spatial_scale,
        output_size[0] * sampling_ratio,
        output_size[1] * sampling_ratio,
        aligned,
        interpolate_mode,
        roi_quantized,
        True,
        march,
    )

    ret[illegal_roi_mask] = 0
    return ret[valid_mask]


@with_march
def roi_align_tensor(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    roi_quantized: bool,
    march: str,
) -> Tensor:
    """
    Qat version of Roi align function.
    Same as torchvision.ops.roi_align, except that the output size is
    sampling_ratio times larger and boxes must be a Tensor.

    Please not that in our underlying implementation the roi is batched into
    Tensor[n, k, 4], which is different with torchvision (Tensor[k, 5], and the
    first element is batch index). So we have to do some extra pre-process
    and post-process.
    """
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
        rois_list.append(rois[rois[:, 0] == batch_idx, 1:])
        forward_mapping = torch.cat(
            (forward_mapping, (rois[:, 0] == batch_idx).nonzero()), dim=0
        )

    reverse_mapping = torch.argsort(
        forward_mapping.flatten(), descending=False
    )

    return roi_align_list(
        input,
        rois_list,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
        interpolate_mode,
        scale,
        zero_point,
        dtype,
        roi_quantized,
        march,
    )[reverse_mapping]


@with_march
def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
):
    """
    Given an input and a flow-field grid, computes the output using
    input values and pixel locations from grid.

    Note that the grid required by this function is DIFFERENT from
    torch.nn.functional.grid_sample !!!

    And the gradient of grid is always 0 now.

    Args:
        input (Tensor[N, C, H, W]): Input data.
        grid (Tensor[N, H_out, W_out, (dx, dy)]): Flow-field. This param
            is different with torch.nn.functional.grid_sample. In this
            function, the sample point of output point (x, y) is computed
            by (x + dx, y + dy).
        mode (str, optional): Interpolation mode to calculate output values.
            Only "bilinear" is supported now.
        padding_mode (str, optional): Padding mode for outside grid values.
            Only "zeros" is supported now.
        align_corners ([type], optional): Since the grid format is
            different with torch.nn.functional.grid_sample, this param
            does not have any effect now.
    """
    # Convert from xy to yx.
    grid_yx = torch.stack((grid[..., 1], grid[..., 0]), dim=-1)

    # Compute coord_shift.
    max_coord = max(
        max(input.size(2), input.size(3)), max(grid.size(2), grid.size(3))
    )
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)

    # Coord int16 quantization.
    grid_scale = torch.tensor(
        1.0 / (1 << coord_shift), dtype=torch.float, device=grid.device
    ).reshape(1)
    quant_info = qinfo("qint16")
    grid_yx = torch.ops.changan.scale_quanti(
        grid_yx.float(),
        grid_scale,
        torch.zeros_like(grid_scale).to(dtype=torch.long),
        -1,
        quant_info.min,
        quant_info.max,
        True,
        False,
        "bpu_round",
        march,
    )

    # Convert to absolute grid.
    N, H, W, _ = grid.shape
    base_coord = torch.stack(
        [
            torch.arange(H, device=grid.device)
            .reshape(1, H, 1)
            .expand(N, H, W),
            torch.arange(W, device=grid.device)
            .reshape(1, 1, W)
            .expand(N, H, W),
        ],
        dim=-1,
    )
    absolute_grid = grid_yx + base_coord
    absolute_grid = absolute_grid.permute(0, 3, 1, 2).unsqueeze(1)

    return torch.ops.changan.quanti_grid_sample(
        input.float(),
        absolute_grid,
        scale,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march,
    )


def avg_pool2d(
    input: QTensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[float]] = None,
    padding: BroadcastingList2[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[bool] = None,
    activation_post_process=None,
    force_same_scale: bool = False,
) -> QTensor:
    assert count_include_pad, "count_include_pad must be True"
    assert divisor_override is None, "divisor_override must be None"

    march = get_march()

    if march == March.BERNOULLI:
        hw_reciprocal = 1 / kernel_size[0] / kernel_size[1]
        # avg = accumulator * (int(hw_reciprocal * 2 ** 9) / 2 ** 9)
        divisor_shift = 9
        out = F.avg_pool2d(
            input.as_subclass(torch.Tensor),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            1,
        ) * (
            int(hw_reciprocal * 2 ** divisor_shift)
            * (1.0 / (2 ** divisor_shift))
        )
        force_same_scale = True
    else:
        out = F.avg_pool2d(
            input.as_subclass(torch.Tensor),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    if force_same_scale:
        assert_qtensor_input(input)
        if activation_post_process is not None:
            activation_post_process.disable_observer()
            activation_post_process.set_qparams(input.q_scale())
        else:
            info = qinfo(input.dtype)
            approximate_mode = "floor" if march == "bernoulli" else "bpu_round"
            out = torch.ops.changan.scale_quanti(
                out,
                input.q_scale(),
                input.q_zero_point(),
                -1,
                info.min,
                info.max,
                True,
                False,
                approximate_mode,
                march,
            )
            return QTensor(
                out, input.q_scale(), input.dtype, input.per_channel_axis
            )
    if activation_post_process is not None:
        return activation_post_process(out)

    # activation_post_process is None and not force_same_scale
    return QTensor(out, scale=None, dtype="float32")


@with_march
def grid_sample_norm_grid(
    input: Tensor,
    grid: Tensor,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
):
    """
    Given an input and a flow-field grid, computes the output using
    input values and pixel locations from grid.

    And the gradient of grid is always 0 now.

    Args:
        input (Tensor[N, C, H, W]): Input data.
        grid (Tensor[N, H_out, W_out, (dx, dy)]): Flow-field. This param
            is same as torch.nn.functional.grid_sample.
        mode (str, optional): Interpolation mode to calculate output values.
            Only "bilinear" is supported now.
        padding_mode (str, optional): Padding mode for outside grid values.
            Only "zeros" is supported now.
        align_corners ([type], optional): This param
            does not have any effect now.
    """
    assert align_corners is True, "only support align_corners=True"

    h, w = input.size(2), input.size(3)
    grid_h, grid_w = grid.size(1), grid.size(2)

    # Compute coord_shift.
    max_coord = max(max(h, w), max(grid_h, grid_w))
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)

    # Convert grid from normed to absolute

    grid_x = grid[..., :1]
    grid_y = grid[..., 1:]

    grid_x = (grid_x + 1) * (w - 1) / 2
    grid_y = (grid_y + 1) * (h - 1) / 2

    # Convert from xy to yx.
    grid_yx = torch.cat((grid_y, grid_x), dim=3)

    # Coord int16 quantization.
    grid_scale = torch.tensor(
        1.0 / (1 << coord_shift), dtype=torch.float, device=grid.device
    ).reshape(1)
    quant_info = qinfo("qint16")
    grid_yx = torch.ops.changan.scale_quanti(
        grid_yx.float(),
        grid_scale,
        torch.zeros_like(grid_scale).to(dtype=torch.long),
        -1,
        quant_info.min,
        quant_info.max,
        True,
        False,
        "bpu_round",
        march,
    )
    grid_yx = grid_yx.permute(0, 3, 1, 2).unsqueeze(1)

    return torch.ops.changan.quanti_grid_sample(
        input.float(),
        grid_yx,
        scale,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march,
    )


@with_march
def requantize(
    input: Tensor,
    input_scale: Tensor,
    output_scale: Tensor,
    input_zero_point: Tensor,
    output_zero_point: Tensor,
    vector_dim: int,
    input_dtype: str,
    output_dtype: str,
    march: str,
) -> Tensor:
    """
    emulate requantize on bpu.

    Args:
        input: Input data.
        input_scale: The scale of input data.
        output_scale: The scale of output data.
        input_zero_point: The zero point of input data.
        output_zero_point: The zero point of output data.
        vector_dim: data quantization channel
        input_dtype: Quanti type of input data.
        output_dtype: Quanti type of output data.
        march: Bpu version.

    Returns:
        Tensor[N, C, ...]: Output.
    """
    return torch.ops.changan.scale_requanti(
        input,
        input_scale,
        output_scale,
        input_zero_point,
        output_zero_point,
        vector_dim,
        input_dtype,
        output_dtype,
        False,  # pre_rshift_with_round
        True,  # post_rshift_with_round
        march,
    )
