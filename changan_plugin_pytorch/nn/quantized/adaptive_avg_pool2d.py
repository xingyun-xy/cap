import numpy as np
import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module

from .functional import avg_pool2d


class AdaptiveAvgPool2d(Module):
    r"""quantize version"""
    _QAT_MODULE = qat.AdaptiveAvgPool2d

    def __init__(
        self,
        output_size,
        out_dtype="qint8",
    ):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        self.out_dtype = out_dtype
        self.register_buffer("scale", torch.tensor([1], dtype=torch.float32))

    def forward(self, x):
        input_size = torch.Tensor((x.shape[2], x.shape[3])).cpu().numpy()
        strides = np.floor(input_size / self.output_size).astype(np.int32)
        kernels = (input_size - (self.output_size - 1) * strides).astype(
            np.int32
        )
        out, out_scale = avg_pool2d(
            x.int_repr(),
            kernel_size=kernels,
            stride=strides,
            padding=(0, 0),
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=None,
            input_scale=x.q_scale(),
            input_zero_point=x.q_zero_point(),
            input_dtype=x.dtype,
            scale=self.scale,
            zero_point=torch.zeros_like(self.scale).to(torch.long),
            dtype=self.out_dtype,
        )
        return QTensor(out, out_scale, self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )

        activation_post_process = mod.activation_post_process
        out_dtype = (
            activation_post_process.dtype
            if activation_post_process is not None
            else "qint32"
        )
        qpool = cls(
            mod.output_size,
            out_dtype,
        )

        if out_dtype != "qint32":
            qpool.scale.copy_(activation_post_process.scale)
        return qpool
