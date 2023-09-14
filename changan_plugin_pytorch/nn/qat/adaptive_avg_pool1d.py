import torch
import torch.nn.functional as F
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn

from .adaptive_avg_pool2d import AdaptiveAvgPool2d


class AdaptiveAvgPool1d(AdaptiveAvgPool2d):
    r"""qat version"""
    _FLOAT_MODULE = nn.AdaptiveAvgPool1d

    def __init__(
        self,
        output_size,
        qconfig=None,
    ) -> None:
        super(AdaptiveAvgPool1d, self).__init__(output_size, qconfig)

    def forward(self, input: QTensor) -> QTensor:
        self._check_output_size(input.shape[2], 2)
        out = F.adaptive_avg_pool1d(
            input.as_subclass(torch.Tensor),
            self.output_size[0],
        )
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        return QTensor(out, scale=None, dtype="float32")

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
        qconfig = mod.qconfig
        qat_pool = cls(
            mod.output_size,
            qconfig=qconfig,
        )
        return qat_pool
