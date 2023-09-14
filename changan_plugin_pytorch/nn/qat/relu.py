import torch
import torch.nn.functional as F
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import nn


class ReLU(nn.ReLU):
    """
    Apply relu operation to input array

    Args:
        inplace: If set to ``True``, will do this operation in-place.
            Default: ``False``
    """

    _FLOAT_MODULE = nn.ReLU

    def __init__(self, inplace=False, qconfig=None):
        super(ReLU, self).__init__(inplace)
        self.qconfig = qconfig

    def forward(self, input):
        assert_qtensor_input(input)

        return QTensor(
            F.relu(input.as_subclass(torch.Tensor), self.inplace),
            input.q_scale(),
            input.dtype,
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
        qat_relu = cls(inplace=mod.inplace, qconfig=qconfig)
        return qat_relu
