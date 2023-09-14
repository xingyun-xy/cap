# linear(in_features, out_features):
# linear_out = matmul(l_input, transpose(l_weight))
# l_input size: (l_input_n, in_features)
# l_weight size: (out_features, in_features)
# linear_out size: (l_input_n, out_features)

# use h=1, w=1, stride=1, kernel_size=1 conv to implement linear:
# conv input size: (l_input_n, in_features, 1, 1)
# conv weight size: (out_features, in_features, 1, 1)
# conv out size: (l_input_n, out_features, 1, 1)

import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn

from .conv2d import Conv2d
from .functional import linear

__all__ = [
    "Linear",
    "LinearReLU",
    "LinearAdd",
    "LinearAddReLU",
]


class Linear(nn.Module):
    _QAT_MODULE = (qat.Linear,)

    def __init__(self, out_dtype, in_features, out_features):
        super(Linear, self).__init__()

        self.register_buffer(
            "weight",
            torch.zeros([out_features, in_features], dtype=torch.float32),
        )
        self.register_buffer(
            "weight_scale", torch.ones(out_features, dtype=torch.float32)
        )
        self.register_buffer(
            "bias", torch.zeros(out_features, dtype=torch.float32)
        )
        self.register_buffer(
            "bias_scale", torch.ones(out_features, dtype=torch.float32)
        )
        self.register_buffer("out_scale", torch.ones([1], dtype=torch.float32))
        self.out_dtype = out_dtype

    def forward(self, input1, input2=None):
        default_zero_point = input1.q_zero_point()
        out, dequant_out_scale = linear(
            input=input1.int_repr(),
            weight=self.weight,
            bias=self.bias,
            sumin=input2.int_repr() if input2 is not None else None,
            activation=self.activation if hasattr(self, "activation") else "",
            input_scale=input1.q_scale(),
            input_zero_point=input1.q_zero_point(),
            input_dtype=input1.dtype,
            weight_scale=self.weight_scale,
            weight_zero_point=torch.zeros_like(self.weight_scale).to(
                torch.long
            ),
            weight_dtype=self.weight_dtype,
            bias_scale=self.bias_scale,
            bias_zero_point=default_zero_point,
            bias_dtype="qint32",
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
                out,
                dequant_out_scale,
                self.out_dtype,
                per_channel_axis=out.ndim - 1,
            )
        else:
            return QTensor(
                out,
                self.out_scale,
                self.out_dtype,
                per_channel_axis=-1,
            )

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
        qlinear = cls(
            out_dtype,
            mod.in_features,
            mod.out_features,
        )
        qlinear.weight.copy_(mod.weight)
        qlinear.weight_scale.resize_as_(weight_post_process.scale)
        qlinear.weight_scale.copy_(weight_post_process.scale)
        qlinear.weight_dtype = weight_post_process.dtype
        if mod.bias is not None:
            qlinear.bias.copy_(mod.bias)
        if out_dtype != "qint32":
            qlinear.out_scale.resize_as_(activation_post_process.scale)
            qlinear.out_scale.copy_(activation_post_process.scale)
        return qlinear


class LinearReLU(Linear):
    _QAT_MODULE = (qat.LinearReLU, qat.LinearReLU6)

    def __init__(self, out_dtype, in_features, out_features):
        super(LinearReLU, self).__init__(out_dtype, in_features, out_features)
        self.activation = "relu"

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU, cls).from_float(mod)


class LinearAdd(Linear):
    _QAT_MODULE = (qat.LinearAdd,)

    def __init__(self, out_dtype, in_features, out_features):
        super(LinearAdd, self).__init__(out_dtype, in_features, out_features)

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    @classmethod
    def from_float(cls, mod):
        return super(LinearAdd, cls).from_float(mod)


class LinearAddReLU(LinearAdd):
    _QAT_MODULE = (qat.LinearAddReLU, qat.LinearAddReLU6)

    def __init__(self, out_dtype, in_features, out_features):
        super(LinearAddReLU, self).__init__(
            out_dtype, in_features, out_features
        )
        self.activation = "relu"

    @classmethod
    def from_float(cls, mod):
        return super(LinearAddReLU, cls).from_float(mod)
