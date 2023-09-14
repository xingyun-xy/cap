from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module
from torch.nn.modules.utils import _pair

from .functional import max_pool2d


class MaxPool2d(Module):
    r"""quantize version"""
    _QAT_MODULE = qat.MaxPool2d

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        out_dtype="qint8",
    ):
        super(MaxPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else None
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        # self.out_dtype = out_dtype

    def forward(self, x):
        out = max_pool2d(
            x.int_repr(),
            kernel_size=self.kernel_size,
            stride=self.stride
            if self.stride is not None
            else self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
        return QTensor(out, x.q_scale(), x.dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        qpool = cls(
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.return_indices,
            mod.ceil_mode,
        )
        return qpool
