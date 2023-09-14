from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
from torch.nn.modules.utils import _pair

from .functional import avg_pool2d


class AvgPool2d(nn.AvgPool2d):
    r"""qat version"""
    _FLOAT_MODULE = nn.AvgPool2d

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        qconfig=None,
    ) -> None:
        super(AvgPool2d, self).__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        assert self.count_include_pad is True, "count_include_pad must be True"
        assert self.divisor_override is None, "divisor_override must be None"
        assert qconfig, "qconfig must be provided for QAT module"
        self.kernel_size = _pair(self.kernel_size)
        self.qconfig = qconfig
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation()

    def forward(self, input: QTensor) -> QTensor:
        return avg_pool2d(
            input,
            self.kernel_size,
            _pair(self.stride) if self.stride else None,
            _pair(self.padding),
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
            self.activation_post_process,
            False,
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
        qconfig = mod.qconfig
        qat_pool = cls(
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            ceil_mode=mod.ceil_mode,
            count_include_pad=mod.count_include_pad,
            divisor_override=mod.divisor_override,
            qconfig=qconfig,
        )
        return qat_pool
