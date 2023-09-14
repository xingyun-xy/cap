import torch
from torch import nn
from changan_plugin_pytorch.march import get_march, March
from .segment_lut import SegmentLUT


class Tanh(nn.Tanh):
    """
    Apply tanh operation to input array
    """

    _FLOAT_MODULE = nn.Tanh

    def __init__(self, qconfig=None):
        super(Tanh, self).__init__()
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.activation_post_process = self.qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            torch.tanh(input.as_subclass(torch.Tensor))
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
        qconfig = mod.qconfig
        if get_march() == March.BAYES:
            qat_tanh = SegmentLUT(torch.tanh, True, qconfig=qconfig)
        else:
            qat_tanh = cls(qconfig=qconfig)
        return qat_tanh
