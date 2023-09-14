import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module

from .functional import correlation


class Correlation(Module):

    _QAT_MODULE = qat.Correlation

    def __init__(
        self,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        pad_size,
        is_multiply,
        out_dtype="qint8",
    ):
        super(Correlation, self).__init__()
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.pad_size = pad_size
        self.is_multiply = is_multiply
        self.out_dtype = out_dtype
        self.register_buffer("scale", torch.tensor([1], dtype=torch.float32))
        self.register_buffer(
            "inter_scale", torch.tensor([1], dtype=torch.float32)
        )

        assert (
            kernel_size > 0 and kernel_size % 2
        ), "Only support positive odd kernel_size"
        assert (
            max_displacement >= 0
        ), "Only support non-negative max_displacement"
        assert stride1 > 0, "Only support positive stride1"
        assert stride2 > 0, "Only support positive stride2"
        assert pad_size >= 0, "Only support non-negative pad_size"
        assert is_multiply, "Only support multiplication now"

    def forward(self, data1, data2):
        out = correlation(
            data1.int_repr(),
            data2.int_repr(),
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.pad_size,
            self.is_multiply,
            data1.q_scale(),
            data1.q_zero_point(),
            data1.dtype,
            data2.q_scale(),
            data2.q_zero_point(),
            data2.dtype,
            self.inter_scale,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(out, self.scale, self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        quantized_mod = cls(
            mod.kernel_size,
            mod.max_displacement,
            mod.stride1,
            mod.stride2,
            mod.pad_size,
            mod.is_multiply,
            mod.activation_post_process.dtype,
        )
        quantized_mod.scale.copy_(mod.activation_post_process.scale)
        quantized_mod.inter_scale.copy_(mod.inter_post_process.scale)
        return quantized_mod
