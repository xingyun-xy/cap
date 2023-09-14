import math

import torch
from changan_plugin_pytorch.dtype import qinfo
from torch import Tensor
from torch.jit.annotations import Tuple


def linear_transform(
    k: Tensor, b: Tensor, max_shift: int
) -> Tuple[int, int, int]:
    bits = 14
    k = torch.clamp(k, -(2 ** bits - 1), (2 ** bits - 1))
    km, ke = torch.frexp(k)
    shift = bits - ke.item()
    if shift > max_shift:
        retk = 0
        bm, be = torch.frexp(b)
        shift = bits - be.item()
        shift = min(shift, max_shift)
        retb = torch.round(bm * (2 ** bits - 1))
        return int(retk), int(retb.item()), int(shift)
    else:
        retk = torch.round(km * (2 ** bits - 1))
        retb = torch.round(b * (2 ** shift))
        retb = torch.clamp(retb, -(2 ** 30), (2 ** 30) - 1)
        retb = int(retb.item())
        # TODO: better solution
        i = 0
        while retb > qinfo("qint16").max or retb < qinfo("qint16").min:
            i += 1
            retb = retb >> 1
        retb = retb << i
        return int(retk.item()), int(retb), int(shift)


def _get_lut_params(
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    qxmin: Tensor,
    qxmax: Tensor,
) -> Tuple[int, int, int, int, int, int]:
    num_entries = 256
    fscale = torch.tensor([0.0], device=data_scale.device)
    fbias = torch.tensor([0.0], device=data_scale.device)
    if qxmax != qxmin:
        fscale = (num_entries - 1) / (qxmax - qxmin)
        fbias = -(num_entries - 1) * qxmin / (qxmax - qxmin)
    scale, bias, pshift = linear_transform(fscale, fbias, 31)
    itplt_shift = 4
    left_shift = 0
    m, k = torch.frexp(torch.tensor(float(bias)))
    right_shift_bias = bias
    if k.item() > 15:
        left_shift = k.item() - 15
        right_shift_bias = int(float(bias) / 2 ** left_shift)
    return scale, bias, right_shift_bias, left_shift, pshift, itplt_shift


@torch.jit.script
def _lut_int8_to_int8(
    table: Tensor,
    data: Tensor,
    march: str,
) -> Tensor:
    """
    Look up int8 table for int8 input on bpu.

    Args:
        table (Tensor): Table for looking up
        data (Tensor): Input converted to index
    """
    if data.numel() == 0:
        return data.to(torch.int8)
    index = data.to(torch.long).contiguous() + 128
    return torch.take(table, index).to(torch.int8)


@torch.jit.script
def _lut_with_int16(
    table: Tensor,
    data: Tensor,
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    qxmin: Tensor,
    qxmax: Tensor,
    num_entries: int,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    """
    Look up table on bpu.

    Args:
        table (Tensor): Table for looking up
        data (Tensor): Input converted to index
        data_scale (Tensor): Scale of input
        data_zero_point (Tensor): zero point of data
        data_type (str): data type
        qxmin (Tensor): Left border of real fitting interval
        qxmax (Tensor): Right border of real fitting interval
        num_entries (int): Number of elements in the table
        scale (Tensor): scale of output
        dtype (str): Quant dtype of output
        zero_point: (Tensor): Output zero point
        march (str): Bpu version

    Returns:
        Tensor
    """
    if data.numel() == 0:
        return data.to(torch.int16)
    (
        scale,
        bias,
        right_shift_bias,
        left_shift,
        pshift,
        itplt_shift,
    ) = _get_lut_params(data_scale, data_zero_point, data_type, qxmin, qxmax)
    res = torch.ops.changan.gpu_quanti_lut(
        data, table, dtype, scale, bias, pshift, itplt_shift
    )
    return res


def _get_line_fit_params(
    xmin: Tensor,
    ymin: Tensor,
    xmax: Tensor,
    ymax: Tensor,
    data_scale: Tensor,
    scale: Tensor,
) -> Tuple[int, int, int, int, int]:
    """
    xmin, ymin, xmax, ymax is tensor of float
    float number: y = x * slope + bias
    qint_y * out_scale = qint_x * data_scale * slope + bias
    qint_y = qint_x * slope * data_scale / out_scale + bias / out_scale
    qint_slope = slope * data_scale / out_scale
    qint_bias = bias / out_scale
    """
    assert xmax >= xmin, "xmax must greater than or equal than xmin"
    slope = torch.tensor([0.0], device=data_scale.device)
    if xmin < xmax:
        slope = (ymax - ymin) / (xmax - xmin)
    bias = ymax - slope * xmax
    slope = slope * data_scale / scale
    bias = bias / scale
    slope, bias, pshift = linear_transform(slope, bias, 32)
    m, k = torch.frexp(torch.tensor(float(bias)))
    left_shift = 0
    right_shift_bias = bias
    if k.item() > 15:
        left_shift = k.item() - 15
        right_shift_bias = int(float(bias) / 2 ** left_shift)
    # bias for plugin, right_shift_bias for compiler
    return slope, bias, right_shift_bias, left_shift, pshift


@torch.jit.script
def _get_multi_table_params(
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
):
    """
    Providing params of line fit, constant fit and table fit for compiler

    Data type of line fit xmin and xmax is tensor of float,
    because use float boundaries and float y=activation_func(boundaries) to
    calculate float slope and bias then transform them into qint slope and
    bias. But xmin xmax of table fit is tensor of int, because qint data
    need to map to index between [0, 255], so qint boundary and 256 is used
    to calculate float slope and bias, then transform them into qint slope
    and bias
    """
    # get left constant params
    # because slope in constant fit is 0 and bias is the constant value,
    # so just send two same boundaries to xmin and xmax to calculate slope,
    # here we use 0
    (
        left_constant_slope,
        left_constant_bias,
        left_constant_right_shift_bias,
        left_constant_left_shift,
        left_constant_pshift,
    ) = _get_line_fit_params(
        torch.tensor(0),
        left_constant_fit_y,
        torch.tensor(0),
        left_constant_fit_y,
        data_scale,
        scale,
    )

    # use 0 to fill the missing value to make the line fit params is
    # the same as table params
    left_constant_params = (
        left_constant_slope,
        left_constant_right_shift_bias,
        left_constant_left_shift,
        left_constant_pshift,
        0,
        0,
    )
    # get left constant params
    (
        right_constant_slope,
        right_constant_bias,
        right_constant_right_shift_bias,
        right_constant_left_shift,
        right_constant_pshift,
    ) = _get_line_fit_params(
        torch.tensor(0),
        right_constant_fit_y,
        torch.tensor(0),
        right_constant_fit_y,
        data_scale,
        scale,
    )
    right_constant_params = (
        right_constant_slope,
        right_constant_right_shift_bias,
        right_constant_left_shift,
        right_constant_pshift,
        0,
        0,
    )
    # get left linear fit params
    (
        left_line_fit_slope,
        left_line_fit_bias,
        left_line_fit_right_shift_bias,
        left_line_fit_left_shift,
        left_line_fit_pshift,
    ) = _get_line_fit_params(
        left_line_xmin,
        left_line_ymin,
        left_line_xmax,
        left_line_ymax,
        data_scale,
        scale,
    )
    left_line_params = (
        left_line_fit_slope,
        left_line_fit_right_shift_bias,
        left_line_fit_left_shift,
        left_line_fit_pshift,
        # left qint boundary of left linear fitting
        int(torch.round(left_line_xmin / data_scale).to(torch.int32).item()),
        # right qint boundary of left linear fitting
        int(torch.round(left_line_xmax / data_scale).to(torch.int32).item()),
    )
    # get right linear fit params
    (
        right_line_fit_slope,
        right_line_fit_bias,
        right_line_fit_right_shift_bias,
        right_line_fit_left_shift,
        right_line_fit_pshift,
    ) = _get_line_fit_params(
        right_line_xmin,
        right_line_ymin,
        right_line_xmax,
        right_line_ymax,
        data_scale,
        scale,
    )
    right_line_params = (
        right_line_fit_slope,
        right_line_fit_right_shift_bias,
        right_line_fit_left_shift,
        right_line_fit_pshift,
        # left qint boundary of right linear fitting
        int(torch.round(right_line_xmin / data_scale).to(torch.int32).item()),
        # right qint boundary of right linear fitting
        int(torch.round(right_line_xmax / data_scale).to(torch.int32).item()),
    )
    # get dense table params
    # dense_right_shift_bias is needed by compiler
    # dense_bias is needed by pytorch plugin
    (
        dense_scale,
        dense_bias,
        dense_right_shift_bias,
        dense_left_shift,
        dense_pshift,
        dense_itplt_shift,
    ) = _get_lut_params(
        data_scale,
        data_zero_point,
        data_type,
        qint_dense_xmin,
        qint_dense_xmax,
    )
    dense_table_params = (
        dense_scale,
        dense_bias,
        dense_right_shift_bias,
        dense_left_shift,
        dense_pshift,
        dense_itplt_shift,
    )
    # get sparse table params
    (
        sparse_scale,
        sparse_bias,
        sparse_right_shift_bias,
        sparse_left_shift,
        sparse_pshift,
        sparse_itplt_shift,
    ) = _get_lut_params(
        data_scale,
        data_zero_point,
        data_type,
        qint_sparse_xmin,
        qint_sparse_xmax,
    )
    sparse_table_params = (
        sparse_scale,
        sparse_bias,
        sparse_right_shift_bias,
        sparse_left_shift,
        sparse_pshift,
        sparse_itplt_shift,
    )
    multi_table_params = {
        "left_constant": left_constant_params,
        "right_constant": right_constant_params,
        "left_line": left_line_params,
        "right_line": right_line_params,
        "dense_table": dense_table_params,
        "sparse_table": sparse_table_params,
    }
    return multi_table_params


@torch.jit.script
def line_fit_interval(
    data: Tensor,
    xmin: Tensor,
    ymin: Tensor,
    xmax: Tensor,
    ymax: Tensor,
    data_scale: Tensor,
    scale: Tensor,
    march: str,
) -> Tensor:
    if data.numel() == 0:
        return data.to(torch.int32)
    slope, bias, right_shift_bias, left_shift, pshift = _get_line_fit_params(
        xmin, ymin, xmax, ymax, data_scale, scale
    )
    res = torch.ops.changan.gpu_quanti_line_fit(
        data, "qint16", slope, bias, pshift
    )
    return res.to(torch.int32)


@torch.jit.script
def table_mapping_interval(
    data: Tensor,
    table: Tensor,
    data_scale: Tensor,
    data_type: str,
    data_zero_point: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    xmin: Tensor,
    xmax: Tensor,
    march: str,
) -> Tensor:
    input_types = {torch.int8: "qint8", torch.int32: "qint16"}
    table = table.to(torch.int32)
    out = _lut_with_int16(
        table,
        data,
        data_scale,
        data_zero_point,
        input_types[data.dtype],
        xmin,
        xmax,
        256,
        scale,
        zero_point,
        "qint16",
        march,
    )
    return out.to(torch.int32)


def _clamp(
    xmin: float,
    xmax: float,
    data_scale: float,
    min_qint_boundary: int,
    max_qint_boundary: int,
):
    left_boundary = min(
        max(math.ceil(xmin / data_scale), min_qint_boundary), max_qint_boundary
    )
    right_boundary = min(
        max(math.ceil(xmax / data_scale), min_qint_boundary), max_qint_boundary
    )
    return left_boundary, right_boundary


def clamp_boundary(
    dense_xmin: float,
    dense_xmax: float,
    sparse_xmin: float,
    sparse_xmax: float,
    left_line_xmin: float,
    left_line_xmax: float,
    right_line_xmin: float,
    right_line_xmax: float,
    data_scale: float,
    min_qint_boundary: int,
    max_qint_boundary: int,
):
    qint_dense_xmin, qint_dense_xmax = _clamp(
        dense_xmin,
        dense_xmax,
        data_scale,
        min_qint_boundary,
        max_qint_boundary,
    )
    qint_sparse_xmin, qint_sparse_xmax = _clamp(
        sparse_xmin,
        sparse_xmax,
        data_scale,
        min_qint_boundary,
        max_qint_boundary,
    )
    qint_left_line_xmin, qint_left_line_xmax = _clamp(
        left_line_xmin,
        left_line_xmax,
        data_scale,
        min_qint_boundary,
        max_qint_boundary,
    )
    qint_right_line_xmin, qint_right_line_xmax = _clamp(
        right_line_xmin,
        right_line_xmax,
        data_scale,
        min_qint_boundary,
        max_qint_boundary,
    )
    qint_right_constant_xmin, qint_left_constant_xmax = _clamp(
        right_line_xmax,
        left_line_xmin,
        data_scale,
        min_qint_boundary,
        max_qint_boundary,
    )
    qint_left_constant_xmin = min_qint_boundary
    qint_right_constant_xmax = max_qint_boundary
    res = [
        qint_dense_xmin,
        qint_dense_xmax,
        qint_sparse_xmin,
        qint_sparse_xmax,
        qint_left_line_xmin,
        qint_left_line_xmax,
        qint_right_line_xmin,
        qint_right_line_xmax,
        qint_left_constant_xmin,
        qint_left_constant_xmax,
        qint_right_constant_xmin,
        qint_right_constant_xmax,
    ]
    return res


def _multi_table_fit(
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
    march: str,
    is_symmetric: bool = False,
) -> Tensor:
    """
    Data type of parameters begin with qint is tensor of int,
    and line fit parameters is tensor of float.Different data type
    is caused by different calculation of slope and bias for mapping
    input of line fit or table fit to qint activation function result
    For more details, see _get_multi_table_params interface above
    """
    if data_type == "qint16":
        data = data.to(torch.int32)
    input_types = {torch.int8: "qint8", torch.int32: "qint16"}
    out = torch.zeros_like(data, dtype=torch.int32)

    # left constant fit
    mask = torch.logical_and(
        data.to(torch.int32) >= qint_left_constant_xmin,
        data.to(torch.int32) <= qint_left_constant_xmax,
    )
    left_constant_fit_input = torch.masked_select(data, mask)
    left_constant_fit_res = line_fit_interval(
        left_constant_fit_input,
        torch.scalar_tensor(0),
        left_constant_fit_y,
        torch.scalar_tensor(0),
        left_constant_fit_y,
        data_scale,
        scale,
        march,
    )
    out.masked_scatter_(mask, left_constant_fit_res)
    mask = torch.logical_and(
        data.to(torch.int32) <= qint_right_constant_xmax,
        data.to(torch.int32) >= qint_right_constant_xmin,
    )
    right_constant_fit_input = torch.masked_select(data, mask)
    right_constant_fit_res = line_fit_interval(
        right_constant_fit_input,
        torch.scalar_tensor(0),
        right_constant_fit_y,
        torch.scalar_tensor(0),
        right_constant_fit_y,
        data_scale,
        scale,
        march,
    )
    out.masked_scatter_(mask, right_constant_fit_res)

    # right line fitting interval
    # we get right_line_xmin by
    # right_line_xmin = qint_right_line_xmin * data_scale,
    # if there is no loss of accuracy,
    # res = right_line_xmin / data_scale should be a tensor of int which
    # equal to qint_right_line_xmin but in fact res is a tensor of float like
    # tensor([8.9999]), after floor operation is will be tensor([8.0]),
    # result to fail to be adjacent to the left boundary of right constant fit
    # so use round to restore res to be qint_right_line_xmin
    mask = torch.logical_and(
        data.to(torch.int32)
        >= torch.round(right_line_xmin / data_scale).to(torch.int32),
        data.to(torch.int32)
        <= torch.round(right_line_xmax / data_scale).to(torch.int32),
    )
    right_x = torch.masked_select(data, mask)
    right_y = line_fit_interval(
        data=right_x,
        xmin=right_line_xmin,
        ymin=right_line_ymin,
        xmax=right_line_xmax,
        ymax=right_line_ymax,
        data_scale=data_scale,
        scale=scale,
        march=march,
    )
    out.masked_scatter_(mask, right_y)

    # left line fitting interval
    mask = torch.logical_and(
        data.to(torch.int32)
        >= torch.round(left_line_xmin / data_scale).to(torch.int32),
        data.to(torch.int32)
        <= torch.round(left_line_xmax / data_scale).to(torch.int32),
    )
    left_x = torch.masked_select(data, mask)
    left_y = line_fit_interval(
        data=left_x,
        xmin=left_line_xmin,
        ymin=left_line_ymin,
        xmax=left_line_xmax,
        ymax=left_line_ymax,
        data_scale=data_scale,
        scale=scale,
        march=march,
    )
    out.masked_scatter_(mask, left_y)

    # sparse table mapping interval
    mask = torch.logical_and(
        data.to(torch.int32) >= qint_sparse_xmin,
        data.to(torch.int32) <= qint_sparse_xmax,
    )
    sparse_x = torch.masked_select(data, mask)
    sparse_y = table_mapping_interval(
        data=sparse_x,
        table=sparse_table,
        data_scale=data_scale,
        data_type=input_types[sparse_x.dtype],
        data_zero_point=data_zero_point,
        scale=scale,
        zero_point=zero_point,
        dtype=dtype,
        xmin=qint_sparse_xmin,
        xmax=qint_sparse_xmax,
        march=march,
    )
    out.masked_scatter_(mask, sparse_y)

    # dense table mapping interval
    mask = torch.logical_and(
        data.to(torch.int32) >= qint_dense_xmin,
        data.to(torch.int32) <= qint_dense_xmax,
    )
    dense_x = torch.masked_select(data, mask)
    dense_y = table_mapping_interval(
        data=dense_x,
        table=dense_table,
        data_scale=data_scale,
        data_type=input_types[sparse_x.dtype],
        data_zero_point=data_zero_point,
        scale=scale,
        zero_point=zero_point,
        dtype=dtype,
        xmin=qint_dense_xmin,
        xmax=qint_dense_xmax,
        march=march,
    )
    out.masked_scatter_(mask, dense_y)
    return out
