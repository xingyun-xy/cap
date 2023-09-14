import torch
from changan_plugin_pytorch.nn import grid_generator as float_grid_generator
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
from torch.nn.modules.utils import _pair


class BaseGridGenerator(nn.Module):
    _FLOAT_MODULE = float_grid_generator.BaseGridGenerator

    def __init__(
        self,
        size,
        with_third_channel,
        qconfig=None,
    ):
        super(BaseGridGenerator, self).__init__()
        self.size = _pair(size)
        self.with_third_channel = with_third_channel
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.qconfig = qconfig

    def forward(self):
        from changan_plugin_pytorch.nn.quantized.functional import (
            base_grid_generator,
        )

        ret = base_grid_generator(
            self.size, self.with_third_channel, self.scale.device
        )

        return QTensor(ret.to(dtype=torch.float), self.scale, "qint16")

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

        qat_mod = cls(
            size=mod.size,
            with_third_channel=mod.with_third_channel,
            qconfig=mod.qconfig,
        )
        return qat_mod
