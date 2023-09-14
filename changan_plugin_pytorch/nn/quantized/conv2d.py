"""
quantized conv
"""
import warnings

import changan_plugin_pytorch
import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qat_mode import get_qat_mode, QATMode
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module
from torch.nn.modules.utils import _pair

from .functional import conv2d

__all__ = ["Conv2d", "ConvReLU2d", "ConvAdd2d", "ConvAddReLU2d"]


def _bernoulli_bias_quantization(qconv, bias):
    m, e = torch.frexp(torch.clamp(bias, -128, 127))
    qconv.bias_scale.copy_(2.0 ** (e - 7))
    qconv.bias.copy_(
        torch.clamp(torch.floor(m * 2 ** 7), -128, 127)
        * qconv.bias_scale.to(m.device)
    )


def _fuse_bn_to_wscale(qconv, mod, march, weight_post_process):
    bn_var_rsqrt = torch.rsqrt(mod.bn.running_var + mod.bn.eps)
    fused_weight = mod.weight * (mod.bn.weight * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(mod.weight.shape) - 1)
    )
    qconv.weight.copy_(fused_weight)
    if march == March.BERNOULLI:
        max_weight = torch.max(
            torch.abs(torch.flatten(fused_weight, start_dim=1)), dim=1
        ).values
        quantized_weight = torch.floor(
            max_weight / weight_post_process.scale + 0.5
        )
        m, e = torch.frexp(quantized_weight)
        assert weight_post_process.dtype in ("qint8", "qint16")
        if weight_post_process.dtype == "qint8":
            bits = 7
        else:
            bits = 15
        shift = torch.where(e > bits, e - bits, e - e)
        wscale = weight_post_process.scale * 2 ** shift
    else:
        wscale = (
            weight_post_process.scale * torch.abs(mod.bn.weight) * bn_var_rsqrt
        )
    qconv.weight_scale.copy_(wscale)

    if mod.bias is not None:
        fused_bias = (
            mod.bias - mod.bn.running_mean
        ) * mod.bn.weight * bn_var_rsqrt + mod.bn.bias
        qconv.bias.copy_(fused_bias)
        if march == March.BERNOULLI:
            _bernoulli_bias_quantization(qconv, fused_bias)
    else:
        fused_bias = (
            (-1) * mod.bn.running_mean * mod.bn.weight * bn_var_rsqrt
            + mod.bn.bias
        )
        if march == March.BERNOULLI:
            _bernoulli_bias_quantization(qconv, fused_bias)
        else:
            qconv.bias.copy_(fused_bias)


def _fuse_bn_to_weight(qconv, module):
    fused_weight, fused_bias = torch.nn.utils.fuse_conv_bn_weights(
        module.weight,
        module.bias,
        module.bn.running_mean,
        module.bn.running_var,
        module.bn.eps,
        module.bn.weight,
        module.bn.bias,
    )
    qconv.weight.copy_(fused_weight)
    qconv.bias.copy_(fused_bias)


class Conv2d(Module):
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `qint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale

    See :class:`~torch.nn.Conv2d` for other attributes.
    """
    _QAT_MODULE = (qat.Conv2d, qat.ConvBN2d)

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
        out_dtype="qint8",
    ):
        super(Conv2d, self).__init__()
        if padding_mode != "zeros":
            raise NotImplementedError(
                "Currently only zero-padding is supported by quantized conv"
            )
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.out_dtype = out_dtype
        self.weight_dtype = "qint8"
        self.bias_dtype = (
            "qint8" if get_march() == March.BERNOULLI else "qint32"
        )

        self._init_buffer()

    def _init_buffer(self):
        weight_shape = [self.out_channels, self.in_channels // self.groups]
        self.register_buffer(
            "weight",
            torch.zeros(
                weight_shape + list(self.kernel_size), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "bias", torch.zeros(self.out_channels, dtype=torch.float32)
        )
        self.register_buffer(
            "bias_scale", torch.ones(self.out_channels, dtype=torch.float32)
        )  # only used in bernoulli
        self.register_buffer(
            "bias_zero_point",
            torch.zeros(self.out_channels, dtype=torch.int64),
        )  # only used in bernoulli
        self.register_buffer(
            "weight_scale", torch.ones(self.out_channels, dtype=torch.float32)
        )
        self.register_buffer("out_scale", torch.ones([1], dtype=torch.float32))

    def forward(self, input1, input2=None):
        default_zero_point = input1.q_zero_point()
        out, dequant_out_scale = conv2d(
            input=input1.int_repr(),
            weight=self.weight,
            bias=self.bias,
            sumin=input2.int_repr() if input2 is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=self.padding_mode,
            activation=self.activation if hasattr(self, "activation") else "",
            input_scale=input1.q_scale(),
            input_zero_point=default_zero_point,
            input_dtype=input1.dtype,
            weight_scale=self.weight_scale,
            weight_zero_point=torch.zeros_like(self.weight_scale).to(
                torch.long
            ),
            weight_dtype=self.weight_dtype,
            bias_scale=self.bias_scale,
            bias_zero_point=self.bias_zero_point,
            bias_dtype=self.bias_dtype,
            sumin_scale=input2.q_scale() if input2 is not None else None,
            sumin_zero_point=default_zero_point
            if input2 is not None
            else None,
            sumin_dtype=input2.dtype if input2 is not None else None,
            scale=self.out_scale,
            zero_point=default_zero_point,
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
                per_channel_axis=-1 if self.out_scale.numel() == 1 else 1,
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
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ", groups={groups}"
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
            out_dtype,
        )
        # buffer
        qconv.weight.copy_(mod.weight)
        qconv.weight_scale.resize_as_(weight_post_process.scale)
        qconv.weight_scale.copy_(weight_post_process.scale)
        qconv.weight_dtype = weight_post_process.dtype

        march = get_march()

        if mod.bias is not None:
            qconv.bias.copy_(mod.bias)
            if march == March.BERNOULLI:
                qconv.bias_scale.resize_as_(mod.bias_scale)
                qconv.bias_scale.copy_(mod.bias_scale)
        if hasattr(mod, "bn") and isinstance(
            mod.bn, torch.nn.modules.batchnorm._BatchNorm
        ):
            qat_mode = get_qat_mode()
            assert qat_mode in [QATMode.WithBN, QATMode.WithBNReverseFold], (
                "Only support {} and {} ".format(
                    QATMode.WithBN, QATMode.WithBNReverseFold
                )
                + "mode if still contain BN when qat finished"
            )
            if qat_mode == QATMode.WithBN:
                _fuse_bn_to_wscale(qconv, mod, march, weight_post_process)
            else:
                _fuse_bn_to_weight(qconv, mod)
        if out_dtype != "qint32":
            qconv.out_scale.resize_as_(activation_post_process.scale)
            qconv.out_scale.copy_(activation_post_process.scale)
        return qconv


class ConvReLU2d(Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _QAT_MODULE = (
        qat.ConvReLU2d,
        qat.ConvReLU62d,
        qat.ConvBNReLU2d,
        qat.ConvBNReLU62d,
    )

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        out_dtype="qint8",
    ):
        super(ConvReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            out_dtype=out_dtype,
        )
        self.activation = "relu"

    @classmethod
    def from_float(cls, mod):
        if isinstance(mod, qat.ConvReLU62d) and get_march() == March.BERNOULLI:
            warnings.warn(
                "Bernoulli hardware only support relu operation."
                "Using 'ConvReLU62d' may cause qat and quantized "
                "accuracy mismatch in Bernoulli"
            )
        return super(ConvReLU2d, cls).from_float(mod)


class ConvAdd2d(Conv2d):
    r"""
    A ConvAdd2d module is a fused module of Conv2d and add

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _QAT_MODULE = (qat.ConvAdd2d, qat.ConvBNAdd2d)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        out_dtype="qint8",
    ):
        super(ConvAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            out_dtype=out_dtype,
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    @classmethod
    def from_float(cls, mod):
        return super(ConvAdd2d, cls).from_float(mod)


class ConvAddReLU2d(Conv2d):
    r"""
    A ConvAddReLU2d module is a fused module of Conv2d, add and ReLU

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _QAT_MODULE = (
        qat.ConvAddReLU2d,
        qat.ConvAddReLU62d,
        qat.ConvBNAddReLU2d,
        qat.ConvBNAddReLU62d,
    )

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        out_dtype="qint8",
    ):
        super(ConvAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            out_dtype=out_dtype,
        )
        self.activation = "relu"

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    @classmethod
    def from_float(cls, mod):
        if (
            isinstance(mod, qat.ConvAddReLU62d)
            and get_march() == March.BERNOULLI
        ):
            warnings.warn(
                "Bernoulli hardware only support relu operation."
                "Using 'ConvAddReLU62d' may cause qat and quantized "
                "accuracy mismatch in Bernoulli"
            )
        return super(ConvAddReLU2d, cls).from_float(mod)
