import torch
import torch.nn.intrinsic as nni
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import intrinsic
from changan_plugin_pytorch.qtensor import QTensor
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_3_t
import changan_plugin_pytorch as hz

__all__ = [
    "Conv3d",
    "ConvReLU3d",
    "ConvAdd3d",
    "ConvAddReLU3d",
    "ConvReLU63d",
    "ConvAddReLU63d",
]


class Conv3d(nn.Conv3d):
    _FLOAT_MODULE = nn.Conv3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        qconfig=None,
    ) -> None:
        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight(channel_len=out_channels)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation(
                channel_len=out_channels
            )

    def _conv_forward(self, input: QTensor):
        return super(Conv3d, self)._conv_forward(
            input.as_subclass(Tensor),
            self.weight_fake_quant(self.weight).as_subclass(Tensor),
            self.bias,
        )

    def forward(self, input: QTensor) -> QTensor:
        out = self._conv_forward(input)

        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

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
        if type(mod) == nni.ConvReLU3d:
            mod = mod[0]
        elif (
            type(mod) == intrinsic.ConvAdd3d
            or type(mod) == intrinsic.ConvReLU63d
            or type(mod) == intrinsic.ConvAddReLU3d
            or type(mod) == intrinsic.ConvAddReLU63d
        ):
            mod = mod.conv
        qconfig = mod.qconfig
        qat_conv3d = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            device=None,
            dtype=None,
            qconfig=qconfig,
        )
        qat_conv3d.weight.copy_(mod.weight)
        if qat_conv3d.bias is not None:
            qat_conv3d.bias.copy_(mod.bias)
        return qat_conv3d


class ConvReLU3d(Conv3d):
    _FLOAT_MODULE = nni.ConvReLU3d
    _version: int = 2

    def forward(self, input: QTensor):
        out = self._conv_forward(input)

        if self.activation_post_process is not None:

            out = hz.nn.qat.compatible_ops.relu(out)
            return self.activation_post_process(out)
        else:
            out = F.relu(out, inplace=True)
            return QTensor(out, None, "float32")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version == 1:
            hz.qat_mode.tricks.relu6 = True

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class ConvReLU63d(Conv3d):
    _FLOAT_MODULE = intrinsic.ConvReLU63d

    def forward(self, input: QTensor):
        out = self._conv_forward(input)
        out = F.relu6(out, inplace=True)

        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")


class ConvAdd3d(Conv3d):
    _FLOAT_MODULE = intrinsic.ConvAdd3d

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def forward(self, input1: QTensor, input2: QTensor):
        out = self._conv_forward(input1)
        out = out + input2.as_subclass(Tensor)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")


class ConvAddReLU3d(ConvAdd3d):
    _FLOAT_MODULE = intrinsic.ConvAddReLU3d
    _version: int = 2

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def forward(self, input1, input2):
        out = self._conv_forward(input1)
        out = out + input2.as_subclass(Tensor)
        if self.activation_post_process is not None:

            out = hz.nn.qat.compatible_ops.relu(out)
            return self.activation_post_process(out)
        else:
            out = F.relu(out, inplace=True)
            return QTensor(out, None, "float32")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version == 1:
            hz.qat_mode.tricks.relu6 = True

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class ConvAddReLU63d(ConvAdd3d):
    _FLOAT_MODULE = intrinsic.ConvAddReLU63d

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def forward(self, input1, input2):
        out = self._conv_forward(input1)
        out = out + input2.as_subclass(Tensor)
        out = F.relu6(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")
