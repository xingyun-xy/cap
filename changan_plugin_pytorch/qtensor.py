import sys
from collections import defaultdict
from numbers import Real

import torch
from torch import Tensor
from torch.jit.annotations import BroadcastingList2, List, Optional
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from changan_plugin_pytorch.dtype import QuantDType, qinfo
from changan_plugin_pytorch.march import March, get_march

__all__ = ["QTensor"]


def qtensor_allow_float_operation(enabled: bool):
    """
    Whether allow to directly use QTensor as input of float operations.
    The default behaviour is False.
    """
    assert isinstance(enabled, bool)

    # we must re-generate the whole dict to change the default behaviour,
    # because a_default_dict["non exist key"] will add "non exist key" to
    # the dict, and modify a_default_dict.default_factory will not change
    # the return value of a_defaultdict["non exist key"]
    QTensor.set_dispatcher(_get_dispatcher(enabled))


def _unsupported(func, types, args, kwargs):
    msg = "function {} is not implemented for QTensor".format(func)
    raise NotImplementedError(msg)


def _qtensor_to_float(data):
    if isinstance(data, QTensor):
        return data.dequantize()
    elif isinstance(data, (list, tuple)):
        return type(data)(_qtensor_to_float(d) for d in data)
    elif isinstance(data, dict):
        ret = {}
        for k, v in data.items():
            ret[k] = _qtensor_to_float(v)
        return ret
    else:
        return data


def _call_on_float(func, types, args, kwargs):
    """Call func on dequantized Tensor"""
    tensor_args = _qtensor_to_float(args)
    tensor_kwargs = _qtensor_to_float(kwargs)
    if tensor_kwargs is None:
        tensor_kwargs = {}
    return func(*tensor_args, **tensor_kwargs)


def _qtensor_to_tensor(data):
    if isinstance(data, QTensor):
        return data.as_subclass(Tensor)
    elif isinstance(data, (list, tuple)):
        return type(data)(_qtensor_to_tensor(d) for d in data)
    elif isinstance(data, dict):
        ret = {}
        for k, v in data.items():
            ret[k] = _qtensor_to_tensor(v)
        return ret
    else:
        return data


def _call_on_tensor(func, types, args, kwargs):
    """Directly call func on the underlying Tensor"""
    tensor_args = _qtensor_to_tensor(args)
    tensor_kwargs = _qtensor_to_tensor(kwargs)
    if tensor_kwargs is None:
        tensor_kwargs = {}
    return func(*tensor_args, **tensor_kwargs)


def _compare_call_on_tensor(func, types, args, kwargs):
    input1 = args[0]
    input2 = args[1]
    assert (
        input1.dtype == input2.dtype
    ), "expeted same dtype, but get one {} and another".format(
        input1.dtype, input2.dtype
    )
    from changan_plugin_pytorch.nn.quantized.functional import requantize

    quantized_input1 = input1.int_repr()
    quantized_input2 = input2.int_repr()
    if input1.q_scale() == input2.q_scale():
        res = func(
            quantized_input1.as_subclass(Tensor),
            quantized_input2.as_subclass(Tensor),
        )
    elif input1.q_scale() > input2.q_scale():
        bigger_scale = input1.q_scale()
        requantize_input2 = requantize(
            quantized_input2,
            input2.q_scale(),
            input2.q_zero_point(),
            input2.dtype,
            bigger_scale,
            input1.q_zero_point(),
            input1.dtype,
        )
        res = func(
            quantized_input1.as_subclass(Tensor),
            requantize_input2.as_subclass(Tensor),
        )
    else:
        bigger_scale = input2.q_scale()
        requantize_input1 = requantize(
            quantized_input1,
            input1.q_scale(),
            input1.q_zero_point(),
            input1.dtype,
            bigger_scale,
            input2.q_zero_point(),
            input2.dtype,
        )
        res = func(
            requantize_input1.as_subclass(Tensor),
            quantized_input2.as_subclass(Tensor),
        )
    return QTensor(res, torch.tensor([1.0], device=input1.device), "qbool")


def _wrap_ret(func, types, args, kwargs):
    """Call func as on Tensor and wrap return value
    as QTensor use input scale and dtype"""
    ret = _call_on_tensor(func, types, args, kwargs)
    return QTensor(ret, args[0].scale, args[0].dtype, args[0].per_channel_axis)


def _wrap_rets(func, types, args, kwargs):
    """Call func as on Tensor and wrap return values
    as QTensor use input scale and dtype"""
    rets = _call_on_tensor(func, types, args, kwargs)
    return type(rets)(
        QTensor(ret, args[0].scale, args[0].dtype, args[0].per_channel_axis)
        for ret in rets
    )


def _wrap_first(func, types, args, kwargs):
    """Call func as on Tensor and wrap the first return value
    as QTensor use input scale and dtype"""
    rets = _call_on_tensor(func, types, args, kwargs)
    return (
        QTensor(
            rets[0], args[0].scale, args[0].dtype, args[0].per_channel_axis
        ),
    ) + rets[1:]


def _assert_scale_close(func, types, args, kwargs):
    """Call func as on Tensor and check the scale of inputs"""
    assert torch.allclose(args[0].scale, args[1].scale)
    rets = _call_on_tensor(func, types, args, kwargs)
    return rets


def qtensor_function(f):
    def wrapped_func(func, types, args, kwargs):
        return f(*args, **kwargs)

    return wrapped_func


@qtensor_function
def _qtensor_avg_pool2d(
    input,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[float]] = None,
    padding: BroadcastingList2[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[bool] = None,
):
    if input.is_quantized:
        from changan_plugin_pytorch.nn.quantized.functional import (
            avg_pool2d as quantized_avg_pool2d,
        )

        kernel_size = _pair(kernel_size)
        stride = kernel_size if stride is None else _pair(stride)
        padding = _pair(padding)

        ret, scale = quantized_avg_pool2d(
            input.as_subclass(Tensor),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )

        return QTensor(ret, scale, input.dtype, input.per_channel_axis)
    else:
        from changan_plugin_pytorch.nn.qat.functional import (
            avg_pool2d as qat_avg_pool2d,
        )

        return qat_avg_pool2d(
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            None,
            True,
        )


@qtensor_function
def _qtensor_pad(
    input,
    pad: List[int],
    mode: str = "constant",
    value: float = 0,
):
    if input.is_quantized:
        from changan_plugin_pytorch.nn.quantized.functional import (
            pad as quantized_pad,
        )

        res = quantized_pad(
            input.as_subclass(Tensor),
            pad,
            mode,
            value,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )
    else:
        if mode == "constant":
            from changan_plugin_pytorch.nn.quantized.functional import quantize

            value = float(
                quantize(
                    torch.tensor([float(value)], device=input.device),
                    input.q_scale(),
                    input.q_zero_point(),
                    -1,
                    input.dtype,
                )[0]
                * input.q_scale(),
            )

        res = torch.nn.functional.pad(
            input.as_subclass(Tensor),
            pad,
            mode,
            value,
        )

    scale = input.q_scale()
    return QTensor(res, scale, input.dtype, input.per_channel_axis)


@qtensor_function
def _qtensor_masked_fill(
    input,
    mask: Tensor,
    value: float,
):
    assert (
        mask.dtype == torch.bool
    ), "mask is expected to be BoolTensor, but got {} instead.".format(
        mask.dtype
    )
    assert input.q_scale().numel() == 1, (
        "only per-tensor scale is supported, "
        + "and expecting scale shape to be (1,), "
        + "but got {} instead".format(input.q_scale().shape)
    )
    if input.is_quantized:
        from changan_plugin_pytorch.nn.quantized.functional import (
            masked_fill as quantized_masked_fill,
        )

        res = quantized_masked_fill(
            input.as_subclass(Tensor),
            mask,
            value,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )
    else:
        from changan_plugin_pytorch.nn.quantized.functional import quantize

        filled_value = float(
            quantize(
                torch.tensor([float(value)], device=input.device),
                input.q_scale(),
                input.q_zero_point(),
                -1,
                input.dtype,
            )[0]
            * input.q_scale()
        )

        res = torch.masked_fill(
            input.as_subclass(Tensor),
            mask,
            filled_value,
        )

    return QTensor(res, input.q_scale(), input.dtype, input.per_channel_axis)


@qtensor_function
def _qtensor_interpolate(
    input,
    size: Optional[BroadcastingList2[int]] = None,
    scale_factor: Optional[BroadcastingList2[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
):
    from changan_plugin_pytorch.nn.qat.functional import (
        interpolate as qat_interploate,
    )
    from changan_plugin_pytorch.nn.quantized.functional import (
        interpolate as quantized_interploate,
    )

    size = _pair(size) if size else None
    scale_factor = _pair(scale_factor) if scale_factor else None

    if input.is_quantized:
        ret = quantized_interploate(
            input.as_subclass(Tensor),
            size,
            scale_factor,
            mode,
            align_corners,
            recompute_scale_factor,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )
    else:
        ret = qat_interploate(
            input.as_subclass(Tensor),
            size,
            scale_factor,
            mode,
            align_corners,
            recompute_scale_factor,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
            True,
        )
        march = get_march()

        info = qinfo(input.dtype)
        approximate_mode = "floor" if march == March.BERNOULLI else "bpu_round"
        ret = torch.ops.changan.scale_quanti(
            ret,
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
    return QTensor(ret, input.q_scale(), input.dtype, input.per_channel_axis)


@qtensor_function
def _qtensor_ones_like(input, **kwargs):
    if "dtype" not in kwargs:
        kwargs["dtype"] = torch.float32
    return torch.ones_like(input.as_subclass(Tensor), **kwargs)


@qtensor_function
def _qtensor_zeros_like(input, **kwargs):
    if "dtype" not in kwargs:
        kwargs["dtype"] = torch.float32
    return torch.zeros_like(input.as_subclass(Tensor), **kwargs)


@qtensor_function
def _qtensor_affine_grid(
    theta, size: List[int], align_corners: Optional[bool] = None
):
    device = theta.device
    INT16_MAX = (1 << 15) - 1

    if theta.is_quantized:
        from changan_plugin_pytorch.nn.quantized.functional import (
            matmul,
            requantize,
        )

        N, C, H, W = size

        x = (
            torch.linspace(
                -INT16_MAX,
                INT16_MAX,
                W,
                dtype=torch.int16,
                device=device,
            )
            .unsqueeze(0)
            .expand(H, W)
        )
        y = (
            torch.linspace(
                -INT16_MAX,
                INT16_MAX,
                H,
                dtype=torch.int16,
                device=device,
            )
            .unsqueeze(-1)
            .expand(H, W)
        )
        ones = torch.full((H, W), INT16_MAX, dtype=torch.int16, device=device)

        if not align_corners:
            x = (x * ((W - 1) / W)).round().to(dtype=torch.int16)
            y = (y * ((H - 1) / H)).round().to(dtype=torch.int16)

        base_grid = (
            torch.stack([x, y, ones], dim=-1).unsqueeze(0).expand(N, H, W, 3)
        )
        base_grid_scale = torch.tensor(
            [1 / INT16_MAX], dtype=torch.float32, device=device
        )
        grid_scale = torch.tensor(
            [2 / INT16_MAX], dtype=torch.float32, device=device
        )

        theta = theta.reshape(N, 1, 2, 3)

        if theta.dtype != "qint16":
            theta = QTensor(
                requantize(
                    theta.as_subclass(Tensor),
                    theta.q_scale(),
                    theta.q_zero_point(),
                    theta.dtype,
                    theta.q_scale() / (1 << 8),
                    theta.q_zero_point(),
                    "qint16",
                ),
                theta.q_scale() / (1 << 8),
                "qint16",
            )

        grid = matmul(
            base_grid.reshape(N, 1, H * W, 3),
            theta.as_subclass(Tensor),
            False,
            True,
            base_grid_scale,
            theta.q_zero_point(),
            "qint16",
            theta.q_scale(),
            theta.q_zero_point(),
            "qint16",
            grid_scale,
            theta.q_zero_point(),
            "qint16",
        ).reshape(N, H, W, 2)

        return QTensor(grid, grid_scale, "qint16")

    else:
        grid = F.affine_grid(theta.as_subclass(Tensor), size, align_corners)
        scale = torch.tensor(
            [2 / INT16_MAX], dtype=torch.float32, device=device
        )
        zero_point = torch.zeros(1, dtype=torch.long, device=device)

        march = get_march()

        info = qinfo("qint16")
        approximate_mode = "floor" if march == March.BERNOULLI else "bpu_round"

        return QTensor(
            torch.ops.changan.scale_quanti(
                grid,
                scale,
                zero_point,
                -1,
                info.min,
                info.max,
                True,
                False,
                approximate_mode,
                march,
            ),
            scale,
            "qint16",
        )


@qtensor_function
def _qtensor_grid_sample(
    input,
    grid,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
):
    assert isinstance(grid, QTensor)

    if input.is_quantized:
        from changan_plugin_pytorch.nn.quantized.functional import (
            grid_sample_norm_grid as quantized_grid_sample,
        )

        ret = quantized_grid_sample(
            input.as_subclass(Tensor),
            grid.as_subclass(Tensor),
            mode,
            padding_mode,
            align_corners,
            grid.q_scale(),
            grid.q_zero_point(),
            grid.dtype,
        )
    else:
        from changan_plugin_pytorch.nn.qat.functional import (
            grid_sample_norm_grid as qat_grid_sample,
        )

        ret = qat_grid_sample(
            input.as_subclass(Tensor),
            grid.as_subclass(Tensor),
            mode,
            padding_mode,
            align_corners,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )

    return QTensor(ret, input.q_scale(), input.dtype, input.per_channel_axis)


@qtensor_function
def _qtensor_mul_scalar(input, other, *, out=None):
    assert out is None
    assert isinstance(other, (int, float))
    other_scale = abs(other)
    other_data = 0 if other == 0 else other / other_scale

    if input.is_quantized:
        # if 0 or 1, directly return QTensor, avoid extra 'mul' on HW
        if other_data == 0:
            r = torch.zeros_like(input.as_subclass(Tensor))
        elif other_data == 1:
            r = input.as_subclass(Tensor)
        else:
            from changan_plugin_pytorch.nn.quantized.functional import mul

            r = mul(
                input.as_subclass(Tensor),
                torch.tensor([[[[other_data]]]]).to(input.as_subclass(Tensor)),
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
                torch.tensor([other_scale], dtype=torch.float32).to(
                    input.device
                ),
                input.q_zero_point(),
                input.dtype,
                input.q_scale() * other_scale,
                input.q_zero_point(),
                input.dtype,
            )
        return QTensor(
            r,
            (input.q_scale() * other_scale).clamp_min(
                torch.finfo(torch.float32).eps
            ),
            input.dtype,
            input.per_channel_axis,
        )
    else:
        return QTensor(
            input.as_subclass(Tensor) * other,
            (input.q_scale() * other_scale).clamp_min(
                torch.finfo(torch.float32).eps
            ),
            input.dtype,
            input.per_channel_axis,
        )


@qtensor_function
def _qtensor_clamp(input, min=None, max=None, *, out=None):
    # min and max could be Number, Tensor and None
    # if min or max is constant Tensor, float and qat results may diff much
    # in the case min < max < input or input < min < max because clamp to input
    # range operation in fake quant of min and max
    info = qinfo(input.dtype)
    if input.is_quantized:
        from changan_plugin_pytorch.nn.quantized.functional import (
            quantize,
            requantize,
        )

        if isinstance(min, QTensor):
            min = requantize(
                min.int_repr(),
                min.q_scale(),
                min.q_zero_point(),
                min.dtype,
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
            )
        elif isinstance(min, Tensor):
            min = quantize(
                min,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                input.dtype,
            )
        elif isinstance(min, Real):
            # keep min type
            min = (
                torch.clamp(
                    torch.floor(min / input.q_scale() + 0.5),
                    info.min,
                    info.max,
                )
                .to(info._storage_type)
                .item()
            )
        else:
            assert min is None, "Only support min type: Number, Tensor, None"

        if isinstance(max, QTensor):
            max = requantize(
                max.int_repr(),
                max.q_scale(),
                max.q_zero_point(),
                max.dtype,
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
            )
        elif isinstance(max, Tensor):
            max = quantize(
                max,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                input.dtype,
            )
        elif isinstance(max, Real):
            # keep max type
            max = (
                torch.clamp(
                    torch.floor(max / input.q_scale() + 0.5),
                    info.min,
                    info.max,
                )
                .to(info._storage_type)
                .item()
            )
        else:
            assert max is None, "Only support max type: Number, Tensor, None"

        r = torch.clamp(input.int_repr(), min, max)
        return QTensor(r, input.q_scale(), input.dtype, input.per_channel_axis)
    else:
        if isinstance(min, QTensor):
            min = min.as_subclass(Tensor)
        elif isinstance(min, Tensor):
            min = torch.ops.changan.scale_quanti(
                min,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                info.min,
                info.max,
                True,
                False,
                "bpu_round",
                get_march(),
            )
        elif isinstance(min, Real):
            # keep min type
            min = (
                torch.clamp(
                    torch.floor(min / input.q_scale() + 0.5),
                    info.min,
                    info.max,
                )
                * input.q_scale()
            ).item()
        else:
            assert min is None, "Only support min type: Number, Tensor, None"

        if isinstance(max, QTensor):
            max = max.as_subclass(Tensor)
        elif isinstance(max, Tensor):
            max = torch.ops.changan.scale_quanti(
                max,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                info.min,
                info.max,
                True,
                False,
                "bpu_round",
                get_march(),
            )
        elif isinstance(max, Real):
            # keep max type
            max = (
                torch.clamp(
                    torch.floor(max / input.q_scale() + 0.5),
                    info.min,
                    info.max,
                )
                * input.q_scale()
            ).item()
        else:
            assert max is None, "Only support max type: Number, Tensor, None"

        return QTensor(
            torch.clamp(input.as_subclass(Tensor), min, max),
            input.q_scale(),
            input.dtype,
            input.per_channel_axis,
        )


@qtensor_function
def _qtensor_channel_shuffle(input, groups: int):
    from changan_plugin_pytorch.nn.functional import channel_shuffle

    return channel_shuffle(input, groups)


@qtensor_function
def _qtensor_pixel_shuffle(input, upscale_factor: int):
    out = F.pixel_shuffle(input.as_subclass(Tensor), upscale_factor)
    return QTensor(out, input.q_scale(), input.dtype, input.per_channel_axis)


@qtensor_function
def _qtensor_pixel_unshuffle(input, downscale_factor: int):
    out = F.pixel_unshuffle(input.as_subclass(Tensor), downscale_factor)
    return QTensor(out, input.q_scale(), input.dtype, input.per_channel_axis)


_CALL_ON_TENSOR = [
    torch._C._TensorBase.argmax,
    torch._C._TensorBase.argmin,
    torch._C._TensorBase.device.__get__,
    torch._C._TensorBase.dim,
    torch._C._TensorBase.get_device,
    torch._C._TensorBase.is_cuda.__get__,
    torch._C._TensorBase.is_contiguous,
    torch._C._TensorBase.numel,
    torch._C._TensorBase.requires_grad.__get__,
    torch._C._TensorBase.requires_grad.__set__,
    torch._C._TensorBase.reshape,
    torch._C._TensorBase.shape.__get__,
    torch._C._TensorBase.size,
    torch._C._TensorBase.ndim.__get__,
    torch.argmax,
    torch.argmin,
    Tensor.backward,
    Tensor.grad.__get__,
    Tensor.grad_fn.__get__,
]


_COMPARE_CALL_ON_TENSOR = [
    torch._C._TensorBase.eq,
    torch._C._TensorBase.gt,
    torch._C._TensorBase.greater,
    torch._C._TensorBase.greater_equal,
    torch._C._TensorBase.ge,
    torch._C._TensorBase.lt,
    torch._C._TensorBase.less,
    torch._C._TensorBase.le,
    torch._C._TensorBase.less_equal,
    torch.eq,
    torch.gt,
    torch.greater,
    torch.greater_equal,
    torch.ge,
    torch.less,
    torch.le,
    torch.less_equal,
    torch.lt,
]


# In dictionary order
_WRAP_RET = [
    torch._C._TensorBase.__getitem__,
    torch._C._TensorBase.contiguous,
    torch._C._TensorBase.detach,
    torch._C._TensorBase.expand,
    torch._C._TensorBase.flatten,
    torch._C._TensorBase.permute,
    torch._C._TensorBase.repeat,
    torch._C._TensorBase.reshape,
    torch._C._TensorBase.roll,
    torch._C._TensorBase.squeeze,
    torch._C._TensorBase.tile,
    torch._C._TensorBase.transpose,
    torch._C._TensorBase.unsqueeze,
    torch._C._TensorBase.view,
    torch.flatten,
    torch.permute,
    torch.reshape,
    torch.roll,
    torch.squeeze,
    torch.tile,
    torch.transpose,
    torch.unsqueeze,
]

_WRAP_RETS = [torch.split, torch.Tensor.split]

_WRAP_FIRST = [
    torch.max,
    torch._C._TensorBase.max,
    torch.min,
    torch._C._TensorBase.min,
]


# In dictionary order
_FUNC_MAPPING = {
    torch._C._TensorBase.clamp: _qtensor_clamp,
    torch._C._TensorBase.clip: _qtensor_clamp,
    torch._C._TensorBase.masked_fill: _qtensor_masked_fill,
    torch._C._TensorBase.mul: _qtensor_mul_scalar,
    torch.clamp: _qtensor_clamp,
    torch.clip: _qtensor_clamp,
    torch.masked_fill: _qtensor_masked_fill,
    torch.mul: _qtensor_mul_scalar,
    torch.ones_like: _qtensor_ones_like,
    torch.zeros_like: _qtensor_zeros_like,
    # functional
    F.avg_pool2d: _qtensor_avg_pool2d,
    F.channel_shuffle: _qtensor_channel_shuffle,
    F.grid_sample: _qtensor_grid_sample,
    F.interpolate: _qtensor_interpolate,
    F.affine_grid: _qtensor_affine_grid,
    F.grid_sample: _qtensor_grid_sample,
    F.pixel_shuffle: _qtensor_pixel_shuffle,
    F.pixel_unshuffle: _qtensor_pixel_unshuffle,
    F.pad: _qtensor_pad,
}


def _get_dispatcher(allow_float_operation: bool = False) -> dict:
    """
    Generate a map from torch function to implementation on QTensor.
    If you need to support more torch functions in QTensor,
    please extend this map.
    """
    dispatcher = defaultdict(
        lambda: _call_on_float
        if allow_float_operation
        else lambda: _unsupported
    )
    for f in _CALL_ON_TENSOR:
        dispatcher[f] = _call_on_tensor
    for f in _COMPARE_CALL_ON_TENSOR:
        dispatcher[f] = _compare_call_on_tensor
    for f in _WRAP_RET:
        dispatcher[f] = _wrap_ret
    for f in _WRAP_RETS:
        dispatcher[f] = _wrap_rets
    for f in _WRAP_FIRST:
        dispatcher[f] = _wrap_first
    for k, v in _FUNC_MAPPING.items():
        dispatcher[k] = v

    return dispatcher


class QTensor(Tensor):
    _QTENSOR_DISPATCHER: dict = _get_dispatcher()

    def __new__(cls, data, scale, dtype, per_channel_axis=-1):
        """Generate a QTensor with quantized data.

        Args:
            data (Tensor): Quantized int data or float data from fake quanti.
            scale (Tensor): Scale.
            dtype (str): Quantize type.
            per_channel_axis (int, optional): The channel axis for per channel
                quantized data, -1 for per tensor quanti. Defaults to -1.

        Returns:
            QTensor
        """
        if scale is not None and scale.numel() > 1:
            assert per_channel_axis > -1, (
                "Please specify per_channel_axis "
                + "for per channel quantized QTensor"
                + "receive scale: {}".format(scale)
            )
        if (
            per_channel_axis > -1
            and not torch.jit.is_scripting()
            and not torch._C._get_tracing_state()
        ):
            assert scale.numel() == data.size(per_channel_axis), (
                "Invalid scale size for per channel quantized QTensor, "
                + "data shape is {} but scale shape is {} (ch_axis={})".format(
                    data.shape, scale.shape, per_channel_axis
                )
            )

        instance = data.as_subclass(cls)
        instance.scale = scale
        instance.zero_point = (
            None
            if scale is None
            else torch.zeros_like(scale, dtype=torch.long)
        )
        # we cannot rewrite Tensor.dtype
        instance.qtype = (
            QuantDType(dtype) if not isinstance(dtype, QuantDType) else dtype
        )
        instance.per_channel_axis = per_channel_axis

        return instance

    @property
    def dtype(self) -> str:
        """Quanti type"""
        return self.qtype

    @property
    def is_quantized(self) -> bool:
        """Is True if the Tensor is quantized, False otherwise."""
        return not self.as_subclass(Tensor).is_floating_point()

    def dequantize(self) -> Tensor:
        """Return the dequantized float Tensor

        Returns:
            Tensor
        """
        if self.is_quantized:
            from .nn.quantized.functional import dequantize

            return dequantize(
                self.as_subclass(Tensor),
                self.scale,
                self.zero_point,
                self.per_channel_axis,
            )
        else:
            return self.as_subclass(Tensor)

    def int_repr(self) -> Tensor:
        """
        Return the quantized int Tensor

        Returns:
            Tensor
        """
        if self.is_quantized:
            return self.as_subclass(Tensor)
        else:
            from .nn.quantized.functional import quantize

            return quantize(
                self.as_subclass(Tensor),
                self.scale,
                self.zero_point,
                -1,
                self.qtype,
            )

    def qscheme(self):
        """Returns the quantization scheme of a given QTensor."""
        if self.scale.numel() == 1:
            if self.zero_point.prod() == 0:
                return torch.per_tensor_symmetric
            else:
                return torch.per_tensor_affine
        else:
            if self.zero_point.prod() == 0:
                return torch.per_channel_symmetric
            else:
                return torch.per_channel_affine

    def q_scale(self) -> Tensor:
        """
        Given a Tensor quantized by linear(affine) quantization,
        returns the scale of the underlying quantizer().

        Returns:
            Tensor
        """
        return self.scale

    def q_zero_point(self) -> Tensor:
        """
        Given a Tensor quantized by linear(affine) quantization,
        returns the zero_point of the underlying quantizer().

        Returns:
            Tensor
        """
        return self.zero_point

    def q_per_channel_scales(self) -> Tensor:
        """
        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a Tensor of scales of the underlying quantizer.
        It has the number of elements that matches the
        corresponding dimensions (from q_per_channel_axis) of the tensor.

        Returns:
            Tensor
        """
        return self.scale

    def q_per_channel_zero_points(self) -> Tensor:
        """
        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a tensor of zero_points of the underlying quantizer.
        It has the number of elements that matches the
        corresponding dimensions (from q_per_channel_axis) of the tensor.

        Returns:
            Tensor
        """
        return self.zero_point

    def q_per_channel_axis(self) -> int:
        """
        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns the index of dimension on which per-channel
        quantization is applied.

        Returns:
            int
        """
        return self.per_channel_axis

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        wrapped_func = cls._QTENSOR_DISPATCHER[func]
        try:
            return wrapped_func(func, types, args, kwargs)
        except Exception as e:
            raise type(e)(
                e.message
                if hasattr(e, "message")
                else "" + "\n when calling function {}".format(func)
            ).with_traceback(sys.exc_info()[2])

    @classmethod
    def set_dispatcher(cls, new_dispatcher: dict):
        cls._QTENSOR_DISPATCHER = new_dispatcher

    def __repr__(self):
        return (
            "QTensor(\n  data = {},\n  scale = {},\n  zero_point = {},\n  "
            + "dtype = {},\n  per_channel_axis = {},\n  is_quantized = {}\n)"
        ).format(
            self.as_subclass(Tensor),
            self.scale,
            self.zero_point,
            self.dtype,
            self.per_channel_axis,
            self.is_quantized,
        )
