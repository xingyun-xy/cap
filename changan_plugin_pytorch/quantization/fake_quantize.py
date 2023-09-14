"""
same as torch.quantization.FakeQuantize.

"""
import re
from typing import Sequence, Union

import torch
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.qtensor import QTensor
from torch.quantization.fake_quantize import (
    FakeQuantizeBase,
    _is_per_channel,
)
from torch.quantization.observer import NoopObserver, _with_args

from .misc import pow_quantization, set_qparam
from .observer import (
    CalibObserver,
    FixedScaleObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

saturate_grad = True


def _is_affine(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_affine, torch.per_channel_affine]


class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale  # noqa



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating
      point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enabled` controls the application of fake quantization
      on tensors, note that statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with
      fake-quantization, the allowable values is qint8 and qint16. The values
      of quant_min and quant_max should be chosen to be consistent with the
      dtype


    Args:
        observer (module): Module for observing statistics on input tensors
                           and calculating scale and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        channel_len (int): Size of data at channel dim.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on
                           the input tensor and provides a method to calculate
                           scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=None,
        quant_max=None,
        saturate=None,
        in_place=False,
        channel_len=1,
        **observer_kwargs,
    ):
        assert (
            saturate is None or type(saturate) == bool
        ), "saturate should be None or bool type"
        assert type(in_place) == bool, "in_place should be bool type"
        assert type(channel_len) == int, "channel_len should be int type"
        assert (
            channel_len >= 1
        ), "channel_len should greater than or equal to 1"

        super(FakeQuantize, self).__init__()
        # use flags rather than buffer to avoid cuda to cpu copy and
        # speed up forward
        self._fake_quant_enabled = True
        self._observer_enabled = True

        self.activation_post_process = observer(
            quant_min=quant_min,
            quant_max=quant_max,
            **observer_kwargs,
        )
        # get quant minmax from observer where they are properly configured
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )

        self.is_per_channel = _is_per_channel(self.qscheme)
        self.is_affine = _is_affine(self.qscheme)

        if self.is_per_channel:
            scale_len = channel_len
        else:
            scale_len = 1

        self.register_buffer(
            "scale", torch.ones(scale_len, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.zeros(scale_len, dtype=torch.long)
        )

        if observer == FixedScaleObserver:
            fixed_scale, fixed_zero_point = self.calculate_qparams()
            self.set_qparams(fixed_scale, fixed_zero_point)

        self.march = get_march()

        self.approximate_mode = "bpu_round"
        if self.march == March.BERNOULLI and not self.is_per_channel:
            self.approximate_mode = "floor"

        if saturate is not None:
            self.saturate = saturate
        else:
            self.saturate = saturate_grad
        self.in_place = in_place

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0
        self._fake_quant_enabled = enabled

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0
        self._observer_enabled = enabled

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        v = state_dict.get(prefix + "observer_enabled", None)
        if v is not None:
            self._observer_enabled = v[0].item() == 1  # use item to get a bool

        v = state_dict.get(prefix + "fake_quant_enabled", None)
        if v is not None:
            self._fake_quant_enabled = v[0].item() == 1

        super(FakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_qparams(
        self,
        scale: Union[torch.Tensor, Sequence, float],
        zero_point: Union[torch.Tensor, Sequence, int] = None,
    ):
        """default symmetric"""
        set_qparam(scale, self.scale, "scale")
        if zero_point is not None:
            set_qparam(zero_point, self.zero_point, "zero_point")
        else:
            self.zero_point.copy_(torch.zeros_like(self.zero_point))

    def forward(self, X):
        # only update scale when training
        if self._observer_enabled and self.training:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            self.scale = _scale
            # if _zero_point is not None:
            #     self.zero_point = _zero_point

        if self._fake_quant_enabled:
            X = torch.ops.changan.scale_quanti(
                X,
                self.scale,
                self.zero_point,
                self.ch_axis,
                self.quant_min,
                self.quant_max,
                self.saturate,
                self.in_place,
                self.approximate_mode,
                self.march,
            )

        # return qtensor type
        return QTensor(
            data=X,
            scale=self.scale,
            dtype=self.dtype,
            per_channel_axis=self.ch_axis,
        )

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return "fake_quant_enabled={}, observer_enabled={},\
            quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, \
        scale={}, zero_point={}".format(
            self.fake_quant_enabled,
            self.observer_enabled,
            self.quant_min,
            self.quant_max,
            self.dtype,
            self.qscheme,
            self.ch_axis,
            self.scale,
            self.zero_point,
        )


class CalibFakeQuantize(FakeQuantizeBase):
    def __init__(
        self,
        channel_len=1,
        dtype="qint8",
        ch_axis=-1,
        observer=CalibObserver,
        **observer_kwargs,
    ):
        assert dtype in (
            "qint8",
            "qint16",
        ), f"unsupported dtype: {dtype}"
        super(CalibFakeQuantize, self).__init__()
        self._observer_enabled = True
        self.activation_post_process = observer(**observer_kwargs)
        if observer in (NoopObserver,):
            self.scale_len = channel_len
        else:
            self.scale_len = 1
        self.register_buffer(
            "scale", torch.ones(self.scale_len, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.zeros(self.scale_len, dtype=torch.long)
        )
        self.channel_len = channel_len
        self.dtype = dtype
        self.ch_axis = ch_axis

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0
        self._observer_enabled = enabled

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        v = state_dict.get(prefix + "observer_enabled", None)
        if v is not None:
            self._observer_enabled = v[0].item() == 1  # use item to get a bool

        super(CalibFakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, X):
        if isinstance(self.activation_post_process, FixedScaleObserver):
            (
                _scale,
                _zero_point,
            ) = self.activation_post_process.calculate_qparams()
            self.set_qparams(_scale, _zero_point)
        elif self._observer_enabled and self.scale_len == 1:
            self.activation_post_process(X.detach())
            fmax = torch.max(torch.abs(X))
            scale = 2 * fmax / (qinfo(self.dtype).max - qinfo(self.dtype).min)
            if pow_quantization():
                min_valid_shift = 1
                max_valid_shift = 14
                shift = torch.floor((-1) * torch.log2(scale))
                shift = torch.clamp(shift, min_valid_shift, max_valid_shift)
                scale = 1 / 2 ** shift
            self.scale.copy_(scale.detach())
        else:
            pass
        return QTensor(
            data=X,
            scale=self.scale,
            dtype=self.dtype,
            per_channel_axis=self.ch_axis,
        )

    def calculate_qparams(self, **kwargs):
        pass

    def set_qparams(
        self,
        scale,
        zero_point=None,
    ):
        """default symmetric"""
        set_qparam(scale, self.scale, "scale")
        if zero_point is not None:
            set_qparam(zero_point, self.zero_point, "zero_point")
        else:
            self.zero_point.copy_(torch.zeros_like(self.zero_point))

    with_args = classmethod(_with_args)


default_8bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="qint8",
)

per_channel_8bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint8",
    ch_axis=1,
)
"""Int8 quantization config with MovingAverageMinMaxObserver for feature"""

default_weight_8bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint8",
    ch_axis=0,
)

default_4bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="qint4",
)
default_uint4_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="quint4",
)

default_weight_4bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint4",
    ch_axis=0,
)
default_16bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="qint16",
)
default_weight_16bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint16",
    ch_axis=0,
)
default_calib_fake_quant = CalibFakeQuantize.with_args(
    observer=CalibObserver,
    num_bits=8,
    axis=None,
    unsigned=False,
    num_bins=2048,
    skip_zeros=False,
    method="percentile",
    percentile=99.99,
)

default_weight_calib_fake_quant = CalibFakeQuantize.with_args(
    observer=NoopObserver,
    ch_axis=0,
)


def _is_fake_quant_script_module(mod):
    """Returns true if given mod is an instance of FakeQuantize script module."""  # noqa
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like
        # '__torch__.torch.quantization.fake_quantize.___torch_mangle_2.FakeQuantize' # noqa
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return (
            name
            == "changan_plugin_pytorch.quantization.fake_quantize.FakeQuantize"
        )  # noqa
    return False


def disable_fake_quant(mod):
    if type(mod) == FakeQuantize or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    if type(mod) == FakeQuantize or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()


def disable_observer(mod):
    if type(mod) == FakeQuantize or _is_fake_quant_script_module(mod):
        mod.disable_observer()


def enable_observer(mod):
    if type(mod) == FakeQuantize or _is_fake_quant_script_module(mod):
        mod.enable_observer()
