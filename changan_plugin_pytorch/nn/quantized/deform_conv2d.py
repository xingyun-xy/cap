import math
from typing import Optional, Tuple
from torch.nn.modules.utils import _pair
import torch
from changan_plugin_pytorch.dtype import qint8, qint16, qint32
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.nn import qat
from torch import Tensor

from .functional import add, cap, conv2d, grid_sample, deform_conv2d

__all__ = [
    "DeformConv2d",
    "DeformConvReLU2d",
    "DeformConvAdd2d",
    "DeformConvAddReLU2d",
]


class DeformConv2d(torch.nn.Module):

    _QAT_MODULE = (qat.DeformConv2d,)
    weight: torch.Tensor
    bias: torch.Tensor
    weight_scale: torch.Tensor
    weight_zero_point: torch.Tensor
    bias_scale: torch.Tensor
    bias_zero_point: torch.Tensor
    out_scale: torch.Tensor
    out_zero_point: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        weight_dtype=qint8,
        out_dtype=qint8,
    ):
        super(DeformConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight_dtype = weight_dtype
        self.out_dtype = out_dtype

        self.activation = ""

        self._init_buffer()

    def _init_buffer(self):
        self.register_buffer(
            "bias", torch.zeros(self.out_channels, dtype=torch.float32)
        )
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_channels,
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
        )

        self.register_buffer(
            "weight_scale", torch.ones(self.out_channels, dtype=torch.float32)
        )
        self.register_buffer(
            "weight_zero_point",
            torch.zeros(self.out_channels, dtype=torch.long),
        )

        # only used in bernoulli
        self.register_buffer(
            "bias_scale", torch.ones(self.out_channels, dtype=torch.float32)
        )
        self.register_buffer(
            "bias_zero_point",
            torch.zeros(self.out_channels, dtype=torch.int64),
        )

        self.register_buffer("out_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer(
            "out_zero_point", torch.zeros(1, dtype=torch.long)
        )

    def forward(
        self,
        input: QTensor,
        offset: QTensor,
        mask: Optional[QTensor] = None,
        other: Optional[QTensor] = None,
    ):
        ret = deform_conv2d(
            input.as_subclass(Tensor),
            offset.as_subclass(Tensor),
            None if mask is None else mask.as_subclass(Tensor),
            None if other is None else other.as_subclass(Tensor),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.activation,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
            offset.q_scale(),
            offset.q_zero_point(),
            offset.dtype,
            None if mask is None else mask.q_scale(),
            None if mask is None else mask.q_zero_point(),
            None if mask is None else mask.dtype,
            None if other is None else other.q_scale(),
            None if other is None else other.q_zero_point(),
            None if other is None else other.dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_dtype,
            self.out_scale,
            self.out_zero_point,
            self.out_dtype,
        )

        if self.out_dtype == qint32:
            return QTensor(ret[0], ret[1], self.out_dtype, per_channel_axis=1)
        else:
            return QTensor(ret[0], self.out_scale, self.out_dtype)

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
            else qint32
        )

        qmod = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            weight_post_process.dtype,
            out_dtype,
        )

        qmod.weight.copy_(mod.weight)
        if mod.bias is not None:
            qmod.bias.copy_(mod.bias)

        qmod.weight_scale.copy_(weight_post_process.scale)
        qmod.weight_zero_point.copy_(weight_post_process.zero_point)
        if out_dtype != qint32:
            qmod.out_scale.copy_(activation_post_process.scale)
            qmod.out_zero_point.copy_(activation_post_process.zero_point)
        return qmod


class DeformConvReLU2d(DeformConv2d):

    _QAT_MODULE = (
        qat.DeformConvReLU2d,
        qat.DeformConvReLU62d,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        weight_dtype=qint8,
        out_dtype=qint8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight_dtype,
            out_dtype,
        )
        self.activation = "relu"


class DeformConvAdd2d(DeformConv2d):

    _QAT_MODULE = (qat.DeformConvAdd2d,)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        weight_dtype=qint8,
        out_dtype=qint8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight_dtype,
            out_dtype,
        )

    def add(self, input1, input2):
        return self.__call__(*input1, other=input2)


class DeformConvAddReLU2d(DeformConv2d):

    _QAT_MODULE = (
        qat.DeformConvAddReLU2d,
        qat.DeformConvAddReLU62d,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        weight_dtype=qint8,
        out_dtype=qint8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight_dtype,
            out_dtype,
        )
        self.activation = "relu"

    def add(self, input1, input2):
        return self.__call__(*input1, other=input2)
