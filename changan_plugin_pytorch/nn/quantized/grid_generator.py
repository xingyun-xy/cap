import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
from torch.nn.modules.utils import _pair

from .functional import base_grid_generator


class BaseGridGenerator(nn.Module):
    _QAT_MODULE = qat.BaseGridGenerator

    def __init__(self, size, with_third_channel):
        super(BaseGridGenerator, self).__init__()
        self.size = _pair(size)
        self.with_third_channel = with_third_channel
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))

    def forward(self):
        base_grid = base_grid_generator(
            self.size, self.with_third_channel, self.scale.device
        )

        return QTensor(base_grid, self.scale, "qint16")

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )

        quantized_mod = cls(
            size=mod.size, with_third_channel=mod.with_third_channel
        )
        return quantized_mod
