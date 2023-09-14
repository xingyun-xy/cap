import torch.nn.functional as F
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn


class ReLU(nn.ReLU):
    """
    Apply relu operation to input array

    Args:
        inplace: If set to ``True``, will do this operation in-place.
            Default: ``False``
    """

    _QAT_MODULE = qat.ReLU

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace)

    def forward(self, input):
        return QTensor(
            F.relu(input.int_repr(), self.inplace),
            input.q_scale(),
            input.dtype,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        quantized_relu = cls(inplace=mod.inplace)
        return quantized_relu
