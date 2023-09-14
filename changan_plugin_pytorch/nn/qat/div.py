import torch
from torch import nn
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.dtype import qint16
from .segment_lut import SegmentLUT
from ..div import Div as float_div


class Div(nn.Module):
    """qat version of div module"""

    _FLOAT_MODULE = float_div

    def __init__(self, qconfig=None):
        super(Div, self).__init__()
        assert qconfig is not None, "qconfig must be provided"
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.activation_post_process = self.qconfig.activation()

    def forward(self, x, y):
        r = torch.div(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        return self.activation_post_process(r)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        if get_march() == March.BAYES:
            return SegmentLUTDiv.from_float(mod)
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_div = cls(qconfig=qconfig)
        return qat_div


class SegmentLUTDiv(nn.Module):
    _FLOAT_MODULE = float_div

    def __init__(self, qconfig):
        super(SegmentLUTDiv, self).__init__()
        self.qconfig = qconfig
        assert qconfig is not None, "qconfig must be provided"
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        from changan_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )
        from changan_plugin_pytorch.nn.quantized.functional_modules import (
            FloatFunctional,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)
        self.reciprocal = SegmentLUT(
            torch.reciprocal,
            True,
            None,
            input_range=None,
            auto_divide_strategy="curvature",
            inverse_func=torch.reciprocal,
            qconfig=int16_qconfig,
        )
        self.mul = FloatFunctional(qconfig=qconfig)

    def propagate_qconfig(self, qconfig):
        from changan_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)
        self.qconfig = qconfig
        self.reciprocal.qconfig = int16_qconfig
        self.mul.qconfig = qconfig

    def forward(self, x, y):
        return self.mul.mul(x, self.reciprocal(y))

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
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_div = cls(qconfig=qconfig)
        return qat_div
