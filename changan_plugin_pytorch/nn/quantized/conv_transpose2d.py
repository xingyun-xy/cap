"""
quantized conv_transpose
"""
import warnings

import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor

from .conv2d import Conv2d
from .functional import conv_transpose2d

__all__ = ["ConvTranspose2d"]


class ConvTranspose2d(Conv2d):

    _QAT_MODULE = (qat.ConvTranspose2d,)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        bias,
        padding_mode="zeros",
        out_dtype="qint8",
    ):
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            out_dtype,
        )
        self.output_padding = output_padding
        self.bias_dtype = (
            "qint8" if get_march() == March.BERNOULLI else "qint32"
        )

    def _init_buffer(self):
        weight_shape = [self.in_channels, self.out_channels // self.groups]
        self.register_buffer(
            "weight",
            torch.zeros(
                weight_shape + list(self.kernel_size), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "weight_scale", torch.ones(self.out_channels, dtype=torch.float32)
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
        self.register_buffer("out_scale", torch.ones([1], dtype=torch.float32))

    def forward(self, input1, input2=None):
        default_zero_point = input1.q_zero_point()
        out, dequant_out_scale = conv_transpose2d(
            input=input1.int_repr(),
            weight=self.weight,
            bias=self.bias,
            sumin=input2.int_repr() if input2 is not None else None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
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
            return QTensor(out, self.out_scale, self.out_dtype)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
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
        qdeconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.output_padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
            out_dtype,
        )
        # buffer
        qdeconv.weight.copy_(mod.weight)
        qdeconv.weight_scale.resize_as_(weight_post_process.scale)
        qdeconv.weight_scale.copy_(weight_post_process.scale)
        qdeconv.weight_dtype = weight_post_process.dtype

        if mod.bias is not None:
            qdeconv.bias.copy_(mod.bias)
            if get_march() == March.BERNOULLI:
                qdeconv.bias_scale.resize_as_(mod.bias_scale)
                qdeconv.bias_scale.copy_(mod.bias_scale)
        if out_dtype != "qint32":
            qdeconv.out_scale.resize_as_(activation_post_process.scale)
            qdeconv.out_scale.copy_(activation_post_process.scale)
        return qdeconv


class ConvTransposeReLU2d(ConvTranspose2d):

    _QAT_MODULE = (qat.ConvTransposeReLU2d, qat.ConvTransposeReLU62d)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        bias,
        padding_mode="zeros",
        out_dtype="qint8",
    ):
        super(ConvTransposeReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            padding_mode,
            out_dtype,
        )
        self.activation = "relu"

    @classmethod
    def from_float(cls, mod):
        if (
            isinstance(mod, qat.ConvTransposeReLU62d)
            and get_march() == March.BERNOULLI
        ):
            warnings.warn(
                "Bernoulli hardware only support relu operation."
                "Using 'ConvTransposeReLU62d' may cause qat and quantized "
                "accuracy mismatch in Bernoulli"
            )
        return super(ConvTransposeReLU2d, cls).from_float(mod)


class ConvTransposeAdd2d(ConvTranspose2d):

    _QAT_MODULE = (qat.ConvTransposeAdd2d,)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        bias,
        padding_mode="zeros",
        out_dtype="qint8",
    ):
        super(ConvTransposeAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            padding_mode,
            out_dtype,
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    @classmethod
    def from_float(cls, mod):
        return super(ConvTransposeAdd2d, cls).from_float(mod)


class ConvTransposeAddReLU2d(ConvTranspose2d):

    _QAT_MODULE = (qat.ConvTransposeAddReLU2d, qat.ConvTransposeAddReLU62d)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        bias,
        padding_mode="zeros",
        out_dtype="qint8",
    ):
        super(ConvTransposeAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            padding_mode,
            out_dtype,
        )
        self.activation = "relu"

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    @classmethod
    def from_float(cls, mod):
        if (
            isinstance(mod, qat.ConvTransposeAddReLU62d)
            and get_march() == March.BERNOULLI
        ):
            warnings.warn(
                "Bernoulli hardware only support relu operation."
                "Using 'ConvTransposeAddReLU62d' may cause qat and quantized "
                "accuracy mismatch in Bernoulli"
            )
        return super(ConvTransposeAddReLU2d, cls).from_float(mod)
