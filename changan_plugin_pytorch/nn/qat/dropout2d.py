import torch
import torch.nn.functional as F
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import nn


class Dropout2d(nn.Dropout2d):
    """
    Apply dropout operation to input array

    Args:
        p: Probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place.
            Default: ``False``
    """

    _FLOAT_MODULE = nn.Dropout2d

    def __init__(self, p=0.5, inplace=False, qconfig=None):
        super(Dropout2d, self).__init__(p, inplace)
        self.qconfig = qconfig

    def forward(self, input):
        assert_qtensor_input(input)

        return QTensor(
            F.dropout2d(
                input.as_subclass(torch.Tensor),
                self.p,
                self.training,
                self.inplace,
            ),
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
        qconfig = mod.qconfig
        qat_dropout = cls(p=mod.p, inplace=mod.inplace, qconfig=qconfig)
        return qat_dropout
