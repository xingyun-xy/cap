import torch
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t
from torch.nn.modules.utils import _ntuple, _pair, _quadruple

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

    def __init__(self, value, qconfig):
        super(_ConstantPadNd, self).__init__()
        self.value = value

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        assert self.qconfig.activation, (
            "activation_post_process must included "
            + "in qconfig for qat.ConstantPadNd"
        )
        self.activation_post_process = self.qconfig.activation()
        self.activation_post_process.disable_observer()

    def forward(self, data):
        assert_qtensor_input(data)

        self.activation_post_process.set_qparams(data.q_scale())
        return self.activation_post_process(
            torch.nn.functional.pad(
                data.as_subclass(torch.Tensor),
                self.padding,
                "constant",
                value=self.value,
            )
        )

    def __repr__(self):
        return "{} (padding={}, value={})".format(
            self.__class__.__name__, self.padding, self.value
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
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qat_mod = cls(
            padding=mod.padding, value=mod.value, qconfig=mod.qconfig
        )
        return qat_mod


class ConstantPad1d(_ConstantPadNd):
    """
    Qat version of torch.nn.ConstantPad1d.
    """

    _FLOAT_MODULE = torch.nn.ConstantPad1d
    padding: _size_2_t

    def __init__(self, padding, value, qconfig):
        super(ConstantPad1d, self).__init__(value, qconfig)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    """
    Qat version of torch.nn.ConstantPad2d.
    """

    _FLOAT_MODULE = torch.nn.ConstantPad2d
    padding: _size_4_t

    def __init__(self, padding, value, qconfig):
        super(ConstantPad2d, self).__init__(value, qconfig)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    """
    Qat version of torch.nn.ConstantPad3d.
    """

    _FLOAT_MODULE = torch.nn.ConstantPad3d
    padding: _size_6_t

    def __init__(self, padding, value, qconfig):
        super(ConstantPad3d, self).__init__(value, qconfig)
        self.padding = _ntuple(6)(padding)


class ZeroPad2d(ConstantPad2d):
    """
    Qat version of torch.nn.ZeroPad2d.
    """

    _FLOAT_MODULE = torch.nn.ZeroPad2d
    padding: _size_4_t

    def __init__(self, padding, qconfig) -> None:
        super(ZeroPad2d, self).__init__(padding, 0.0, qconfig)

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

        qat_mod = cls(padding=mod.padding, qconfig=mod.qconfig)
        return qat_mod


class _ReplicationPadNd(torch.nn.Module):
    __constants__ = ["padding"]

    def forward(self, data):
        return QTensor(
            torch.nn.functional.pad(
                data.as_subclass(torch.Tensor), self.padding, "replicate"
            ),
            data.q_scale(),
            data.dtype,
        )

    def extra_repr(self) -> str:
        return "{}".format(self.padding)

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

        qat_mod = cls(padding=mod.padding)
        qat_mod.qconfig = mod.qconfig

        return qat_mod


class ReplicationPad1d(_ReplicationPadNd):
    """
    Qat version of torch.nn.ReplicationPad1d.
    """

    _FLOAT_MODULE = torch.nn.ReplicationPad1d
    padding: _size_2_t

    def __init__(self, padding: _size_2_t) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    """
    Qat version of torch.nn.ReplicationPad2d.
    """

    _FLOAT_MODULE = torch.nn.ReplicationPad2d
    padding: _size_4_t

    def __init__(self, padding: _size_4_t) -> None:
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    """
    Qat version of torch.nn.ReplicationPad3d.
    """

    _FLOAT_MODULE = torch.nn.ReplicationPad3d
    padding: _size_6_t

    def __init__(self, padding: _size_6_t) -> None:
        super(ReplicationPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)
