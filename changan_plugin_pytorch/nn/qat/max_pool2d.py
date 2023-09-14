import torch
import torch.nn.functional as F
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
from torch.nn.modules.utils import _pair


class MaxPool2d(nn.MaxPool2d):
    r"""qat version"""
    _FLOAT_MODULE = nn.MaxPool2d

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        qconfig=None,
    ) -> None:
        super(MaxPool2d, self).__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.kernel_size = _pair(self.kernel_size)
        self.qconfig = qconfig
        assert self.return_indices is False, "return_indices must be False"
        assert self.dilation == 1, "dilation must be 1"

    def forward(self, input: QTensor) -> QTensor:
        out = F.max_pool2d(
            input.as_subclass(torch.Tensor),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        return QTensor(out, input.q_scale(), input.dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_pool = cls(
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            return_indices=mod.return_indices,
            ceil_mode=mod.ceil_mode,
            qconfig=qconfig,
        )
        return qat_pool
