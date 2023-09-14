import torch
from torch import nn
import torch.nn.functional as F
from changan_plugin_pytorch.march import March, get_march
from .segment_lut import SegmentLUT


class LeakyReLU(nn.Module):
    r"""Applies the element-wise function

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    _FLOAT_MODULE = nn.LeakyReLU

    def __init__(
        self, negative_slope: float = 1e-2, inplace: bool = False, qconfig=None
    ):
        super(LeakyReLU, self).__init__()
        self.qconfig = qconfig
        assert qconfig, "qconfig must be provided for QAT module"
        assert self.qconfig.activation, "qconfig activation must be provided"
        self.lut = SegmentLUT(
            lambda x: F.leaky_relu(x, self.negative_slope, self.inplace),
            False,
            [0, 0, 0, 0, 0, 0, 0, 0],
            qconfig=qconfig,
        )
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return self.lut(input)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args:
            mod: a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name___
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qat_mod = cls(
            negative_slope=mod.negative_slope,
            inplace=mod.inplace,
            qconfig=mod.qconfig,
        )
        return qat_mod
