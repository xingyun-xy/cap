import torch
from changan_plugin_pytorch.qtensor import QTensor
from .segment_lut import SegmentLUT
from ..pow import Pow, exponent_assign_and_check


class Pow(torch.nn.Module):
    r"""qat version of pow"""
    _FLOAT_MODULE = Pow

    def __init__(self, exponent, qconfig):
        super(Pow, self).__init__()
        self.qconfig = qconfig
        self.exponent = (
            exponent if isinstance(exponent, int) else exponent.item()
        )
        from changan_plugin_pytorch.nn.quantized import FloatFunctional

        if self.exponent == 2:
            self.mul = FloatFunctional(qconfig=qconfig)
        else:
            self.pow = SegmentLUT(
                lambda x: torch.pow(x, self.exponent),
                False,
                None,
                qconfig=qconfig,
            )

    def forward(self, input: QTensor, exponent):
        exponent_assign_and_check(self, exponent)
        if self.exponent == 2:
            return self.mul.mul(input, input)
        else:
            return self.pow(input)

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
        qat_mod = cls(
            exponent=mod.exponent,
            qconfig=mod.qconfig,
        )
        return qat_mod
