import torch
import torch.nn.functional as F
from torch import nn
from changan_plugin_pytorch.march import get_march, March
from .segment_lut import SegmentLUT


class SiLU(nn.SiLU):
    """
    Apply silu operation to input array
    """

    _FLOAT_MODULE = nn.SiLU

    def __init__(self, qconfig=None):
        super(SiLU, self).__init__()
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.activation_post_process = self.qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            F.silu(input.as_subclass(torch.Tensor))
        )

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
        if get_march() == March.BAYES:
            qat_silu = SegmentLUT(
                F.silu,
                False,
                # dividing_points=[-6, -4, -3, -1.3, -1, 0, 1, 6],
                auto_divide_strategy="curvature",
                qconfig=mod.qconfig,
            )
        else:
            qat_silu = cls(qconfig=mod.qconfig)
        return qat_silu
