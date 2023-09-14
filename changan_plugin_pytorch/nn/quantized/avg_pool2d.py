import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module
from torch.nn.modules.utils import _pair

from .functional import avg_pool2d


class AvgPool2d(Module):
    r"""quantize version"""
    _QAT_MODULE = qat.AvgPool2d

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        out_dtype="qint8",
    ):
        super(AvgPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else kernel_size
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.out_dtype = out_dtype
        self.register_buffer("scale", torch.tensor([1], dtype=torch.float32))

    def forward(self, x):
        out, out_scale = avg_pool2d(
            x.int_repr(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
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
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.ceil_mode,
            mod.count_include_pad,
            mod.divisor_override,
            out_dtype,
        )

        if out_dtype != "qint32":
            qpool.scale.copy_(activation_post_process.scale)
        return qpool
