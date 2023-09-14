import warnings

import torch
from changan_plugin_pytorch.dtype import qinfo, qint8, qint16
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn,
)
from torch import Tensor
from torch.nn import Module

from .functional import lut, segment_lut


def _arange(start, stop, step, device=None, output_length=None):
    if isinstance(start, Tensor):
        device = start.device
        start = start.item()
    if isinstance(stop, Tensor):
        device = stop.device
        stop = stop.item()
    if isinstance(step, Tensor):
        device = step.device
        step = step.item()
    if step == 0:
        return torch.full((output_length,), start, device=device)
    else:
        return torch.arange(start, stop, step, device=device)


def _generate_single_table(func, input_scale, output_scale):
    x = _arange(-128, 128, 1, input_scale.device)
    y = func(x * input_scale)

    info = qinfo(qint8)
    return (
        (y / output_scale + 0.5)
        .floor()
        .clamp(info.min, info.max)
        .to(info._storage_type)
    )


def _get_linear_kb_by_points(x1, x2, y1, y2):
    diffx = x1 - x2
    diffy = y1 - y2
    if diffx == 0:
        k = torch.zeros_like(x1)
    else:
        k = diffy / diffx
    b = y1 - x1 * k
    return k, b


def _get_linear_kb(func, x1, x2):
    return _get_linear_kb_by_points(x1, x2, func(x1), func(x2))


def _convert_linear_kb(k, b, input_scale, output_scale):
    # (x / input_scale * int_k + (int_b << left_shift)) >> right_shift
    #     = (x * k + b) / output_scale
    # x / input_scale * int_k + (int_b << left_shift)
    #     = (x * k / output_scale + b / output_scale) << right_shift
    # int_k / input_scale = (k / output_scale) << right_shift
    # int_b << left_shift = (b / output_scale) << right_shift
    # int_k >> right_shift = (k / output_scale * input_scale)
    # int_b << left_shift = (b / output_scale) << right_shift
    int_k, neg_right_shift = torch.ops.changan.toqint(
        k / output_scale * input_scale,  # x
        16,  # qbits
        16,  # max_bits
        False,  # allow_left_shift
        True,  # allow_right_shift
    )
    right_shift = -neg_right_shift

    # limit int_b << left_shift to int31
    max_right_shift = max(
        30 - torch.log2((b / output_scale).abs() + 1).ceil(), 0
    )
    if right_shift > max_right_shift:
        int_k = (int_k >> (right_shift - max_right_shift)).to(
            dtype=torch.int32
        )
        right_shift[:] = max_right_shift

    int_b, left_shift = torch.ops.changan.toqint(
        b / output_scale * (1 << right_shift.item()),  # x
        16,  # qbits
        31,  # max_bits
        True,  # allow_left_shift
        False,  # allow_right_shift
    )
    return int_k, int_b, left_shift, right_shift


def _generate_table(
    func, xmin, xmax, output_scale, output_dtype=qint16, table_length=64
):
    step = (xmax - xmin) / (table_length - 1)
    x = _arange(xmin, xmax + step * 0.5, step=step, output_length=table_length)
    y = func(x)

    info = qinfo(output_dtype)
    return (y / output_scale).clamp(info.min, info.max).to(info._storage_type)


def _get_optimized_dividing_points(
    func, input_float_min, input_float_max, strategy, accu_divide_num=256
):
    if strategy == "curvature":
        # use curvature to decide dividing points
        step = (input_float_max - input_float_min) / accu_divide_num
        x = _arange(
            input_float_min,
            input_float_max + step * 0.5,
            step,
            output_length=accu_divide_num + 1,
        )
        y = func(x)
        dy = (y[1:] - y[:-1]) / step
        ddy = (dy[1:] - dy[:-1]) / step
        if ddy.isnan().sum() > 0:
            raise ValueError
        ddy = ddy.abs()
        ddy = torch.cat([ddy, ddy[-1:]])
        accumulate = torch.cumsum(ddy, dim=0)
        segment_idx = accumulate.div(accumulate[-1] / 6, rounding_mode="floor")

        dividing_points = input_float_min.reshape(1)
        for i, p in zip(segment_idx + 1, x[1:]):
            if i > 6:
                break
            if i > dividing_points.numel():
                # constraint segment range not smaller than step
                dividing_points = torch.cat([dividing_points, p.reshape(1)])

        dividing_points = torch.cat(
            [dividing_points, x[-1:], input_float_max.reshape(1)]
        )

        return dividing_points

    elif strategy == "evenly":
        step = (input_float_max - input_float_min) / 6
        return _arange(
            input_float_min,
            input_float_max + step * 1.5,
            step,
            output_length=8,
        )
    else:
        raise ValueError("Unsupported strategy")


class SegmentLUT(Module):
    out_scale: Tensor
    out_zero_point: Tensor

    _QAT_MODULE = qat.SegmentLUT

    def __init__(
        self,
        simulated_func,
        is_centrosymmetric=False,
        dividing_points=None,
        input_range=None,
        auto_divide_strategy="evenly",
        inverse_func=None,
        output_scale=None,
        output_dtype=qint16,
        device=torch.device("cpu"),
    ):
        super(SegmentLUT, self).__init__()

        assert output_dtype in (qint8, qint16)

        output_scale = output_scale.clone().detach().to(device=device)

        self.simulated_func = simulated_func
        self.is_centrosymmetric = is_centrosymmetric
        if dividing_points:
            self.dividing_points = torch.tensor(dividing_points, device=device)
        else:
            self.dividing_points = None
        if input_range:
            self.input_range = torch.tensor(input_range, device=device)
        else:
            self.input_range = None
        self.auto_divide_strategy = auto_divide_strategy
        # only used for monotonically decreasing function to
        # generate input range
        self.inverse_func = inverse_func
        self.output_dtype = output_dtype
        self.idx_bits = 8

        self.register_buffer(
            "out_scale",
            torch.tensor(output_scale, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "out_zero_point", torch.ones(1, dtype=torch.long, device=device)
        )

        self.handle_load_state_dict()

    def handle_load_state_dict(self):
        def _allow_miss_key_hook(
            state_dict: dict,
            prefix: str,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            ignore_keys = [
                "single_table",
                "table",
                "scale",
                "beta",
                "left_shift",
                "right_shift",
                "max",
            ]

            msg = (
                "LUT params are generated on the fly instead of saved in "
                + "buffers since version 0.14.7, please update the "
                + "quantized ckpt to avoid this warning"
            )

            have_unexpected_keys = False
            for ignore_key in ignore_keys:
                if prefix + ignore_key in state_dict:
                    state_dict.pop(prefix + ignore_key)
                    have_unexpected_keys = True
            if have_unexpected_keys:
                warnings.warn(msg)

        self._register_load_state_dict_pre_hook(_allow_miss_key_hook)

    @script_quantized_fn
    def _init_single_table_params(
        self,
        input_scale,
    ):
        return _generate_single_table(
            self.simulated_func, input_scale, self.out_scale
        )

    @script_quantized_fn
    def _init_multi_table_params(
        self,
        input_scale: Tensor,
        input_dtype,
    ):
        device = input_scale.device

        table = torch.zeros((6, 64), dtype=torch.int16, device=device)
        alpha = torch.zeros(8, dtype=torch.int16, device=device)
        beta = torch.zeros(8, dtype=torch.int16, device=device)
        left_shift = torch.zeros(8, dtype=torch.int8, device=device)
        right_shift = torch.zeros(8, dtype=torch.int8, device=device)

        info = qinfo(input_dtype)
        # get input min max
        if self.input_range is None:
            input_float_min = info.min * input_scale
            input_float_max = info.max * input_scale
            # generate monotonically decreasing function input range
            # for function f(x):
            #   input_range: [qin_min * s_in, qin_max * s_in]
            #   output_range: [f(qin_max * s_in), f(qin_min * s_in)]
            #                =[qout_min * s_out, qout_max * s_out]
            #   so: input_min = f^{-1}(qout_max * s_out)
            #       input_max = qin_max * s_in
            # Note: this range is the theoretically upper bound of input range.
            #   If want more precise result, try to specify input range by
            #   parameter input_range
            if self.inverse_func is not None:
                input_float_min = self.inverse_func(
                    qinfo(self.output_dtype).max * self.out_scale
                )
        else:
            input_float_min, input_float_max = self.input_range
        if self.is_centrosymmetric and input_float_min < 0:
            input_float_min[:] = 0

        # generate dividing_points
        if self.dividing_points is None:
            dividing_points = _get_optimized_dividing_points(
                self.simulated_func,
                input_float_min,
                input_float_max,
                self.auto_divide_strategy,
            )
        else:
            dividing_points = self.dividing_points

        # generate int params
        segment_max = (
            (dividing_points / input_scale)
            .round()
            .clamp(info.min, info.max)
            .to(torch.int16)
        )
        segment_max[-1] = info.max

        # must recompute dividing points according to max !
        dividing_points = segment_max * input_scale

        # params for left linear fit
        if input_float_min == dividing_points[0]:
            k, b = _get_linear_kb(
                self.simulated_func,
                input_float_min,
                (dividing_points[1] - dividing_points[0]) / 63
                + dividing_points[0],
            )
        else:
            k, b = _get_linear_kb(
                self.simulated_func, input_float_min, dividing_points[0]
            )
        int_k, int_b, lshift, rshift = _convert_linear_kb(
            k, b, input_scale, self.out_scale
        )
        alpha[0] = int_k
        beta[0] = int_b
        left_shift[0] = lshift
        right_shift[0] = rshift

        # params for right linear fit
        if input_float_max == dividing_points[-1]:
            k, b = _get_linear_kb(
                self.simulated_func,
                (dividing_points[-2] - dividing_points[-1]) / 63
                + dividing_points[-1],
                input_float_max,
            )
        else:
            k, b = _get_linear_kb(
                self.simulated_func,
                dividing_points[-1],
                input_float_max,
            )
        int_k, int_b, lshift, rshift = _convert_linear_kb(
            k, b, input_scale, self.out_scale
        )
        alpha[-1] = int_k
        beta[-1] = int_b
        left_shift[-1] = lshift
        right_shift[-1] = rshift

        # params for segment lut
        for i in range(1, 7):
            xmin = dividing_points[i - 1]
            xmax = dividing_points[i]
            table[i - 1].copy_(
                _generate_table(
                    self.simulated_func,
                    xmin,
                    xmax,
                    self.out_scale,
                    self.output_dtype,
                )
            )
            k, b = _get_linear_kb_by_points(xmin, xmax, 0, 63)
            int_k, int_b, lshift, rshift = _convert_linear_kb(
                k, b, input_scale, 1.0 / (1 << self.idx_bits)
            )
            alpha[i] = int_k
            beta[i] = int_b
            left_shift[i] = lshift
            right_shift[i] = rshift

        return table, alpha, beta, left_shift, right_shift, segment_max

    def forward(self, input: QTensor):
        if input.dtype == qint8 and self.output_dtype == qint8:
            ret = lut(
                input.as_subclass(Tensor),
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
                self._init_single_table_params(input.q_scale()),
                self.out_scale,
                self.out_zero_point,
                self.output_dtype,
            )
        else:
            assert (
                input.dtype == qint16
            ), "SegmentLUT do not support qint8 input with qint16 output"
            (
                table,
                alpha,
                beta,
                left_shift,
                right_shift,
                segment_max,
            ) = self._init_multi_table_params(input.q_scale(), input.dtype)
            ret = segment_lut(
                input.as_subclass(Tensor),
                table,
                alpha,
                beta,
                left_shift,
                right_shift,
                segment_max,
                self.is_centrosymmetric,
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
                self.out_scale,
                self.out_zero_point,
                self.output_dtype,
            )
        return QTensor(ret, self.out_scale, self.output_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        assert (
            mod.activation_post_process
        ), "qat mod  must have activation_post_process"

        return cls(
            simulated_func=mod.simulated_func,
            is_centrosymmetric=mod.is_centrosymmetric,
            dividing_points=mod.dividing_points,
            input_range=mod.input_range,
            auto_divide_strategy=mod.auto_divide_strategy,
            inverse_func=mod.inverse_func,
            output_scale=mod.activation_post_process.scale,
            output_dtype=mod.activation_post_process.dtype,
            device=mod.activation_post_process.scale.device,
        )
