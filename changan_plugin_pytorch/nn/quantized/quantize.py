import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor

from .functional import quantize


class Quantize(torch.nn.Module):
    r"""Quantizes an incoming tensor

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor
     `ch_axis`: channel dim index for per channel quant
     `quant_min`: min quantized value
     `quant_max`: max quantized value

    Attributes:
      `scale`, `zero_point`, `dtype`, `ch_axis`, `quant_min`, `quant_max`

    """
    _QAT_MODULE = qat.QuantStub

    def __init__(
        self, scale, zero_point, ch_axis, quant_min, quant_max, dtype
    ):
        super(Quantize, self).__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point.to(dtype=torch.long))
        self.ch_axis = ch_axis
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.dtype = dtype

    def forward(self, x):
        if isinstance(x, QTensor):
            assert torch.all(
                x.q_scale() == self.scale
            ), "input scale must be the same as op's"
            return x
        if x.is_floating_point():
            out = quantize(x, self.scale, self.zero_point, -1, self.dtype)
            return QTensor(out, self.scale, self.dtype, self.ch_axis)
        else:
            return QTensor(x, self.scale, self.dtype, self.ch_axis)

    def extra_repr(self):
        return "scale={}, zero_point={}, dtype={}".format(
            self.scale, self.zero_point, self.dtype
        )

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        assert hasattr(mod, "activation_post_process")
        scale, zero_point = (
            mod.activation_post_process.scale,
            mod.activation_post_process.zero_point,
        )
        return cls(
            scale.detach(),
            zero_point.detach(),
            mod.activation_post_process.ch_axis,
            mod.activation_post_process.quant_min,
            mod.activation_post_process.quant_max,
            mod.activation_post_process.dtype,
        )


class DeQuantize(torch.nn.Module):
    r"""DeQuantizes an incoming tensor"""
    _QAT_MODULE = qat.DeQuantStub

    def __init__(self):
        super(DeQuantize, self).__init__()

    def forward(self, x):
        return x.dequantize()

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        return cls()
