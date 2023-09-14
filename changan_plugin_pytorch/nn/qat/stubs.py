import torch
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn


class QuantStub(nn.Module):
    r"""Quantize stub module

    Args:
        scale: Pass a number to use as fixed scale.
        zero_point: Pass a number to use as fixed zero_point.
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    _FLOAT_MODULE = torch.quantization.QuantStub

    def __init__(self, scale=None, zero_point=0, qconfig=None):
        super(QuantStub, self).__init__()
        assert qconfig, "qconfig must be provided for QAT module"
        self.scale = scale
        self.zero_point = zero_point
        self.qconfig = qconfig
        self.activation_post_process = self.qconfig.activation()
        if self.scale is not None:
            self.activation_post_process.disable_observer()
            self.activation_post_process.set_qparams(
                self.scale, self.zero_point
            )

    def forward(self, x):
        """when training with one net, but bpu inference with multi-subnet,
        qtensor is allowed as input
        """
        if isinstance(x, QTensor):
            assert (
                self.scale is None
            ), "scale canot be fixed when input type is QTensor"
            assert (
                self.activation_post_process.dtype == x.dtype
            ), "input dtype must be the same as qconfig"
            self.activation_post_process.disable_observer()
            self.activation_post_process.disable_fake_quant()
            self.activation_post_process.set_qparams(x.scale)
            return self.activation_post_process(x.as_subclass(torch.Tensor))
        return self.activation_post_process(x)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        from changan_plugin_pytorch.quantization import QuantStub

        cls._FLOAT_MODULE = (torch.quantization.QuantStub, QuantStub)

        assert type(mod) in cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + [modc.__name__ for modc in cls._FLOAT_MODULE]
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_stub = cls(
            scale=getattr(mod, "scale", None),
            zero_point=getattr(mod, "zero_point", None),
            qconfig=qconfig,
        )
        return qat_stub


class DeQuantStub(torch.quantization.DeQuantStub):
    r"""Dequantize stub module"""
    _FLOAT_MODULE = torch.quantization.DeQuantStub

    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x):
        """QTensor -> Tensor"""
        return x.as_subclass(torch.Tensor)

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
        return cls()
