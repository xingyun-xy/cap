from typing import Optional

import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.utils import _triple

from .functional import conv3d

__all__ = ["Conv3d", "ConvReLU3d", "ConvAdd3d", "ConvAddReLU3d"]


class Conv3d(Module):
    _QAT_MODULE = (qat.Conv3d,)

    weight: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor
    bias: Tensor
    bias_scale: Tensor
    bias_zero_point: Tensor
    out_scale: Tensor
    out_zero_point: Tensor

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_dtype=None,
        bias_dtype=None,
        out_dtype=None,
    ):
        super(Conv3d, self).__init__()
        if padding_mode != "zeros":
            raise NotImplementedError(
                "Currently only zero-padding is supported by quantized conv"
            )
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.factory_kwargs = {"device": device}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight_dtype = weight_dtype
        self.bias_dtype = bias_dtype or "qint32"
        self.out_dtype = out_dtype

        self._init_buffer()
        self._set_activation()

    def _set_activation(self):
        self.activation = ""

    def _init_buffer(self):
        self.register_buffer(
            "weight",
            torch.zeros(
                (
                    self.out_channels,
                    self.in_channels // self.groups,
                    *self.kernel_size,
                ),
                dtype=torch.float32,
                **self.factory_kwargs,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(
                self.out_channels, dtype=torch.float32, **self.factory_kwargs
            ),
        )
        self.register_buffer(
            "weight_zero_point",
            torch.zeros(
                self.out_channels, dtype=torch.int64, **self.factory_kwargs
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                self.out_channels, dtype=torch.float32, **self.factory_kwargs
            ),
        )
        self.register_buffer(
            "bias_scale",
            torch.ones(
                self.out_channels, dtype=torch.float32, **self.factory_kwargs
            ),
        )
        self.register_buffer(
            "bias_zero_point",
            torch.zeros(
                self.out_channels, dtype=torch.int64, **self.factory_kwargs
            ),
        )

        self.register_buffer(
            "out_scale",
            torch.ones([1], dtype=torch.float32, **self.factory_kwargs),
        )
        self.register_buffer(
            "out_zero_point",
            torch.ones([1], dtype=torch.int64, **self.factory_kwargs),
        )

    def forward(self, input1: QTensor, input2: Optional[QTensor] = None):
        out, dequant_out_scale = conv3d(
            input=input1.int_repr(),
            weight=self.weight,
            bias=self.bias,
            sumin=input2.int_repr() if input2 is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=self.padding_mode,
            activation=self.activation,
            input_scale=input1.q_scale(),
            input_zero_point=input1.q_zero_point(),
            input_dtype=input1.dtype,
            weight_scale=self.weight_scale,
            weight_zero_point=self.weight_zero_point,
            weight_dtype=self.weight_dtype,
            bias_scale=self.bias_scale,
            bias_zero_point=self.bias_zero_point,
            bias_dtype=self.bias_dtype,
            sumin_scale=input2.q_scale() if input2 is not None else None,
            sumin_zero_point=input2.q_zero_point()
            if input2 is not None
            else None,
            sumin_dtype=input2.dtype if input2 is not None else None,
            scale=self.out_scale,
            zero_point=self.out_zero_point,
            dtype=self.out_dtype,
        )
        if self.out_dtype == "qint32":
            return QTensor(
                out, dequant_out_scale, self.out_dtype, per_channel_axis=1
            )
        else:
            return QTensor(
                out,
                self.out_scale,
                self.out_dtype,
            )

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) in cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + [modc.__name__ for modc in cls._QAT_MODULE]
        )
        weight_post_process = mod.weight_fake_quant
        activation_post_process = mod.activation_post_process
        out_dtype = (
            activation_post_process.dtype
            if activation_post_process is not None
            else "qint32"
        )
        # construct
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
            mod.weight.device,
            None,
            weight_post_process.dtype,
            None,
            out_dtype,
        )
        # buffer
        qconv.weight.copy_(mod.weight)
        qconv.weight_scale.resize_as_(weight_post_process.scale)
        qconv.weight_scale.copy_(weight_post_process.scale)

        if mod.bias is not None:
            qconv.bias.copy_(mod.bias)
        if out_dtype != "qint32":
            qconv.out_scale.resize_as_(activation_post_process.scale)
            qconv.out_scale.copy_(activation_post_process.scale)
        return qconv


class ConvReLU3d(Conv3d):
    _QAT_MODULE = (qat.ConvReLU3d, qat.ConvReLU63d)

    def _set_activation(self):
        self.activation = "relu"


class ConvAdd3d(Conv3d):
    _QAT_MODULE = (qat.ConvAdd3d,)

    def add(self, input1, input2):
        return self.__call__(input1, input2)


class ConvAddReLU3d(ConvAdd3d, ConvReLU3d):
    _QAT_MODULE = (qat.ConvAddReLU3d, qat.ConvAddReLU63d)
