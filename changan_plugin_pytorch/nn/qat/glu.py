import torch
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn

from .segment_lut import SegmentLUT


class GLU(nn.Module):
    r"""qat version"""
    _FLOAT_MODULE = nn.GLU

    def __init__(self, dim, qconfig):
        super(GLU, self).__init__()
        self.qconfig = qconfig
        self.dim = dim
        self.sigmoid = SegmentLUT(
            torch.sigmoid,
            False,
            None,
            qconfig=qconfig,
        )
        from changan_plugin_pytorch.nn.quantized.functional_modules import (
            FloatFunctional,
        )

        self.mul = FloatFunctional(qconfig=self.qconfig)

    def forward(self, input: QTensor):
        if input.shape[self.dim] % 2 != 0:
            raise RuntimeError(
                f"Halving dimension must be even, but dimension {self.dim} "
                "is size {input.shape[self.dim]}"
            )
        half_data1, half_data2 = input.split(
            split_size=input.shape[self.dim] // 2, dim=self.dim
        )
        sigmoid_res = self.sigmoid(half_data2)
        ret = self.mul.mul(half_data1, sigmoid_res)
        return ret

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
        qat_glu = cls(dim=mod.dim, qconfig=mod.qconfig)
        return qat_glu
