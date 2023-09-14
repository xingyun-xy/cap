import torch
from changan_plugin_pytorch.march import March, get_march
from torch import nn
from torch.nn import functional as F

from .segment_lut import SegmentLUT


class GELU(nn.GELU):
    """
    Apply GELU operation to input array
    """

    _FLOAT_MODULE = nn.GELU

    def __init__(self, qconfig=None):
        super(GELU, self).__init__()
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"

        if get_march() == March.BAYES:
            self.lut = SegmentLUT(
                F.gelu,
                False,
                [-5, -3, -2, -1, -0.75, 0, 5.5, float("inf")],
                qconfig=qconfig,
            )
            self.activation_post_process = None
        else:
            self.lut = None
            self.activation_post_process = self.qconfig.activation()

    def forward(self, input):
        if isinstance(self.lut, SegmentLUT):
            return self.lut(input)
        return self.activation_post_process(
            F.gelu(input.as_subclass(torch.Tensor))
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args:
            'mod' a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qat_mod = cls(qconfig=mod.qconfig)
        return qat_mod
