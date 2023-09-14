import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t
from torch.nn.modules.utils import _ntuple, _pair, _quadruple

from .functional import pad

__all__ = [
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ZeroPad2d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
]


class _ConstantPadNd(torch.nn.Module):
    __constants__ = ["padding", "value"]
    value: float

    def __init__(self, value):
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, data):
        return QTensor(
            pad(
                data.int_repr(),
                self.padding,
                "constant",
                self.value,
                data.q_scale(),
                data.q_zero_point(),
                data.dtype,
            ),
            data.q_scale(),
            data.dtype,
        )

    def __repr__(self):
        return "{} (padding={}, value={})".format(
            self.__class__.__name__, self.padding, self.value
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        assert hasattr(
            mod, "activation_post_process"
        ), "qat._ConstantPadNd must have activation_post_process"

        quantized_mod = cls(padding=mod.padding, value=mod.value)

        return quantized_mod


class ConstantPad1d(_ConstantPadNd):
    """
    Quantized version of torch.nn.ConstantPad1d.
    """

    _QAT_MODULE = qat.ConstantPad1d

    def __init__(self, padding, value):
        super(ConstantPad1d, self).__init__(value)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    """
    Quantized version of torch.nn.ConstantPad2d.
    """

    _QAT_MODULE = qat.ConstantPad2d

    def __init__(self, padding, value):
        super(ConstantPad2d, self).__init__(value)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    """
    Quantized version of torch.nn.ConstantPad3d.
    """

    _QAT_MODULE = qat.ConstantPad3d

    def __init__(self, padding, value):
        super(ConstantPad3d, self).__init__(value)
        self.padding = _ntuple(6)(padding)


class ZeroPad2d(ConstantPad2d):
    """
    Quantized version of torch.nn.ZeroPad2d.
    """

    _QAT_MODULE = qat.ZeroPad2d
    padding: _size_4_t

    def __init__(self, padding: _size_4_t) -> None:
        super(ZeroPad2d, self).__init__(padding, 0.0)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        assert hasattr(
            mod, "activation_post_process"
        ), "qat._ConstantPadNd must have activation_post_process"

        quantized_mod = cls(padding=mod.padding)

        return quantized_mod


class _ReplicationPadNd(torch.nn.Module):
    __constants__ = ["padding"]

    def __init__(self):
        super(_ReplicationPadNd, self).__init__()

    def forward(self, data: QTensor) -> QTensor:
        return QTensor(
            pad(
                data.int_repr(),
                self.padding,
                "replicate",
                0,
                data.q_scale(),
                data.q_zero_point(),
                data.dtype,
            ),
            data.q_scale(),
            data.dtype,
        )

    def extra_repr(self) -> str:
        return "{}".format(self.padding)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )

        quantized_mod = cls(padding=mod.padding)
        return quantized_mod


class ReplicationPad1d(_ReplicationPadNd):
    """
    Quantized version of torch.nn.ReplicationPad1d.
    """

    _QAT_MODULE = qat.ReplicationPad1d
    padding: _size_2_t

    def __init__(self, padding: _size_2_t) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    """
    Quantized version of torch.nn.ReplicationPad2d.
    """

    _QAT_MODULE = qat.ReplicationPad2d
    padding: _size_4_t

    def __init__(self, padding: _size_4_t) -> None:
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    """
    Quantized version of torch.nn.ReplicationPad3d.
    """

    _QAT_MODULE = qat.ReplicationPad3d
    padding: _size_6_t

    def __init__(self, padding: _size_6_t) -> None:
        super(ReplicationPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)
