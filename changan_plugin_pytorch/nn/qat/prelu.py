import torch
import torch.nn.functional as F
from torch import nn


class PReLU(nn.PReLU):
    r"""Applies the element-wise function:

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are
            legitimate: 1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Attributes:
        weight (Tensor): the learnable weights of shape(:attr:`num_parameters`)
    """

    _FLOAT_MODULE = nn.PReLU

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, qconfig=None
    ):
        super(PReLU, self).__init__(num_parameters, init)
        self.qconfig = qconfig
        assert (
            qconfig and qconfig.activation is not None
        ), "qconfig and activation must be provided for PReLU QAT module"
        from ...quantization.qconfig import replace_qconfig_dtype

        self.activation_post_process = self.qconfig.activation()
        self.weight_fake_quanti = replace_qconfig_dtype(
            qconfig, "qint16"
        ).activation()

    def forward(self, input):
        out = F.prelu(
            input.as_subclass(torch.Tensor),
            self.weight
            if self.num_parameters == 1
            else self.weight_fake_quanti(self.weight).dequantize(),
        )
        return self.activation_post_process(out)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: 'mod' a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat." + cls.__name__,
            +".from_float only works for " + cls._FLOAT_MODULE.__name__,
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qat_prelu = cls(
            num_parameters=mod.num_parameters,
            qconfig=mod.qconfig,
        )
        qat_prelu.weight = mod.weight
        return qat_prelu
