"""
fused conv2d+add+relu modules
"""
import torch
import torch.nn.functional as F
import torch.nn.intrinsic as nni
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import intrinsic
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
import changan_plugin_pytorch as hz

__all__ = [
    "Conv2d",
    "ConvReLU2d",
    "ConvAdd2d",
    "ConvAddReLU2d",
    "ConvReLU62d",
    "ConvAddReLU62d",
]


class Conv2d(nn.Conv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight(channel_len=out_channels)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation(
                channel_len=out_channels
            )

        if get_march() == March.BERNOULLI:
            self.register_buffer(
                "bias_scale", torch.ones(out_channels, dtype=torch.float32)
            )

    def _get_weight_for_fake_quant(self, fused_weight=None):
        return self.weight

    def _fake_quant_bias(self):
        if get_march() is not March.BERNOULLI or self.bias is None:
            return self.bias
        else:
            m, e = torch.frexp(torch.clamp(self.bias, -128, 127))
            self.bias_scale.copy_(2.0 ** (e - 7))
            return torch.ops.changan.scale_quanti(
                self.bias,
                self.bias_scale,
                torch.zeros(len(self.bias_scale), dtype=torch.int64).to(
                    self.bias.device
                ),
                0,
                -128,
                127,
                True,
                False,
                "floor",
                March.BERNOULLI,
            )

    def _conv_forward(self, input, weight, bias):
        if isinstance(input, QTensor) and input.q_scale().numel() > 1:
            assert (
                self.in_channels == self.out_channels
                and self.in_channels == self.groups
            ), "only depthwise conv2d support per channel quanti input"

        return super(Conv2d, self)._conv_forward(
            input.as_subclass(torch.Tensor),
            weight.as_subclass(torch.Tensor),
            bias,
        )

    def forward(self, input):
        out = self._conv_forward(
            input,
            self.weight_fake_quant(self.weight),
            self._fake_quant_bias(),
        )
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
        if type(mod) == nni.ConvReLU2d:
            mod = mod[0]
        elif (
            type(mod) == intrinsic.ConvAdd2d
            or type(mod) == intrinsic.ConvReLU62d
            or type(mod) == intrinsic.ConvAddReLU2d
            or type(mod) == intrinsic.ConvAddReLU62d
        ):
            mod = mod.conv
        qconfig = mod.qconfig
        qat_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv


class ConvReLU2d(Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nni.ConvReLU2d
    _version: int = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def forward(self, input):
        out = self._conv_forward(
            input,
            self.weight_fake_quant(self.weight),
            self._fake_quant_bias(),
        )

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

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU2d, cls).from_float(mod)


class ConvReLU62d(Conv2d):
    r"""
    A ConvReLU62d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def forward(self, input):
        out = self._conv_forward(
            input,
            self.weight_fake_quant(self.weight),
            self._fake_quant_bias(),
        )
        out = F.relu6(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU62d, cls).from_float(mod)


class ConvAdd2d(Conv2d):
    _FLOAT_MODULE = intrinsic.ConvAdd2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def forward(self, input1, input2):
        out = self._conv_forward(
            input1,
            self.weight_fake_quant(self.weight),
            self._fake_quant_bias(),
        )
        out = out + input2.as_subclass(torch.Tensor)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

    @classmethod
    def from_float(cls, mod):
        return super(ConvAdd2d, cls).from_float(mod)


class ConvAddReLU2d(ConvAdd2d):
    _FLOAT_MODULE = intrinsic.ConvAddReLU2d
    _version: int = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def forward(self, input1, input2):
        out = self._conv_forward(
            input1,
            self.weight_fake_quant(self.weight),
            self._fake_quant_bias(),
        )
        out = out + input2.as_subclass(torch.Tensor)

        if self.activation_post_process is not None:
            # default use relu6
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

    @classmethod
    def from_float(cls, mod):
        return super(ConvAddReLU2d, cls).from_float(mod)


class ConvAddReLU62d(ConvAdd2d):
    _FLOAT_MODULE = intrinsic.ConvAddReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvAddReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def forward(self, input1, input2):
        out = self._conv_forward(
            input1,
            self.weight_fake_quant(self.weight),
            self._fake_quant_bias(),
        )
        out = out + input2.as_subclass(torch.Tensor)
        out = F.relu6(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

    @classmethod
    def from_float(cls, mod):
        return super(ConvAddReLU62d, cls).from_float(mod)
