from typing import Sequence, Union

import torch
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn.parameter import Parameter
from torch.quantization.observer import _with_args

from .misc import set_qparam
from .observer import MovingAverageMinMaxObserver
from .observer import FixedScaleObserver


class PACTFakeQuantize(torch.quantization.FakeQuantizeBase):
    r"""This is an extension of the FakeQuantize module in fake_quantize.py
    which support learning of the alpha,which is an activation
    clipping parameter to  find the right quantization scale.
    When using symmetric quantization ,scale can be calculated by

    scale = alpha / (float(quant_max - quant_min) / 2)

    Args:
        observer (module): Module for observing statistics on input tensors
                           and calculating scale and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        alpha(float): An activation clipping parameter
        channel_len (int): Size of data at channel dim,default is 1
        observer_kwargs (optional): Arguments for the observer module
    """

    def __init__(
        self,
        observer,
        quant_min=0,
        quant_max=255,
        alpha=6.0,
        channel_len=1,
        **observer_kwargs,
    ):
        super(PACTFakeQuantize, self).__init__()
        assert (
            quant_min < quant_max
        ), "quant_min must be strictly less than quant_max."
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.activation_post_process = observer(**observer_kwargs)
        assert (
            qinfo(self.activation_post_process.dtype).min <= quant_min
        ), "quant_min out of bound"
        assert (
            quant_max <= qinfo(self.activation_post_process.dtype).max
        ), "quant_max out of bound"
        if observer == FixedScaleObserver:
            fixed_scale, fixed_zero_point = self.calculate_qparams()
            self.register_buffer("scale", fixed_scale)
            self.register_buffer("zero_point", fixed_zero_point)
        else:
            self.register_buffer(
                "scale", torch.tensor([1.0], dtype=torch.float)
            )
            self.register_buffer(
                "zero_point", torch.tensor([0], dtype=torch.long)
            )
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        self.is_symmetric_quant = True
        self.alpha = Parameter(torch.tensor([alpha]))
        if self.qscheme not in (
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        ):
            self.is_symmetric_quant = False
            self.n_alpha = Parameter(torch.tensor([-alpha]))

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

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
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            X = torch.where(X > self.alpha, self.alpha, X)
            self.activation_post_process.max_val.data.fill_(self.alpha.data[0])
            if X.min() < 0:
                if self.is_symmetric_quant:
                    X = torch.where(X < -self.alpha, -self.alpha, X)
                    self.activation_post_process.min_val.data.fill_(
                        -self.alpha[0].data
                    )
                else:
                    X = torch.where(X < self.n_alpha, self.n_alpha, X)
                    self.activation_post_process.min_val.data.fill_(
                        self.n_alpha[0].data
                    )
            else:
                self.activation_post_process.min_val.data.fill_(0.0)

            (
                _scale,
                _zero_point,
            ) = self.activation_post_process.calculate_qparams()
            assert self.scale.shape == _scale.shape, (
                "mismatched shape when update scale {} vs {}".format(
                    self.scale.shape, _scale.shape
                )
                + ". Please set or check channel_len param in qconfig"
            )
            self.scale.copy_(_scale)
            if _zero_point is not None:
                assert self.zero_point.shape == _zero_point.shape, (
                    "mismatched shape when update zero_point {} vs {}".format(
                        self.zero_point.shape, _zero_point.shape
                    )
                    + ". Please set or check channel_len param in qconfig"
                )
                self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            X = torch.fake_quantize_per_tensor_affine(
                X,
                self.scale.item(),
                self.zero_point.item(),
                self.quant_min,
                self.quant_max,
            )

        # return qtensor type
        return QTensor(
            data=X,
            scale=self.scale,
            dtype=self.dtype,
        )

    with_args = classmethod(_with_args)


default_8bit_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("qint8").min,
    quant_max=qinfo("qint8").max,
    dtype="qint8",
    qscheme=torch.per_tensor_symmetric,
)

default_4bit_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("qint4").min,
    quant_max=qinfo("qint4").max,
    dtype="qint4",
    qscheme=torch.per_tensor_symmetric,
)

default_uint4_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("quint4").min,
    quant_max=qinfo("quint4").max,
    dtype="quint4",
    qscheme=torch.per_tensor_symmetric,
)

default_16bit_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("qint16").min,
    quant_max=qinfo("qint16").max,
    dtype="qint16",
    qscheme=torch.per_tensor_symmetric,
)
