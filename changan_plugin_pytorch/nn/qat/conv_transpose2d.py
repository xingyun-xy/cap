"""
fused conv2d+add+relu modules
"""
import torch
import torch.nn.functional as F
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import intrinsic
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
import changan_plugin_pytorch as hz

__all__ = [
    "ConvTranspose2d",
    "ConvTransposeReLU2d",
    "ConvTransposeAdd2d",
    "ConvTransposeReLUAdd2d",
    "ConvTransposeReLU62d",
    "ConvTransposeAddReLU62d",
]


class ConvTranspose2d(nn.ConvTranspose2d):
    r"""
    A ConvTrnaspose2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.ConvTranspose2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#
    torch.nn.ConvTranspose2d
    for documentation.

    Similar to `torch.nn.ConvTranspose2d`, with FakeQuantize modules
    initialized to default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for \
                ConvTranspose2d"
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
        return self._convert_weight()

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

    def _convert_weight(self):
        wsize = self.weight.size()
        qat_weight = self.weight.reshape(
            (
                self.groups,
                wsize[0] // self.groups,
                wsize[1],
                wsize[2],
                wsize[3],
            )
        )
        qat_weight = qat_weight.transpose(dim0=1, dim1=2)
        qat_weight = qat_weight.reshape(
            (
                wsize[1] * self.groups,
                wsize[0] // self.groups,
                wsize[2],
                wsize[3],
            )
        )
        return qat_weight

    def _restore_weight(self, qat_weight):
        wsize = self.weight.size()
        qat_weight = qat_weight.reshape(
            (
                self.groups,
                wsize[1],
                wsize[0] // self.groups,
                wsize[2],
                wsize[3],
            )
        )
        qat_weight = qat_weight.transpose(dim0=1, dim1=2)
        qat_weight = qat_weight.reshape(wsize)
        return qat_weight

    def _fake_quant_weight(self):
        qat_weight = self.weight_fake_quant(self._convert_weight())
        return self._restore_weight(qat_weight.as_subclass(torch.Tensor))

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0],
                qat_weight,
                self._fake_quant_bias(),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return QTensor(out, scale=None, dtype="float32")

        return _func

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(
            input.as_subclass(torch.Tensor),
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        return self._forward_func(output_padding)(
            input.as_subclass(torch.Tensor)
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
        if (
            type(mod) == intrinsic.ConvTransposeReLU2d
            or type(mod) == intrinsic.ConvTransposeAdd2d
            or type(mod) == intrinsic.ConvTransposeAddReLU2d
            or type(mod) == intrinsic.ConvTransposeReLU62d
            or type(mod) == intrinsic.ConvTransposeAddReLU62d
        ):
            mod = mod.conv_transpose2d
        else:
            mod = mod
        qconfig = mod.qconfig
        qat_deconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            output_padding=mod.output_padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_deconv.weight = mod.weight
        qat_deconv.bias = mod.bias
        return qat_deconv


class ConvTransposeReLU2d(ConvTranspose2d):
    r"""
    A ConvTransposeReLU2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.


    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvTransposeReLU2d
    _version: int = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0],
                qat_weight,
                self._fake_quant_bias(),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

            if self.activation_post_process is not None:
                out = hz.nn.qat.compatible_ops.relu(out)
                return self.activation_post_process(out)
            else:
                out = F.relu(out, inplace=True)
                return QTensor(out, scale=None, dtype="float32")

        return _func

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
        return super(ConvTransposeReLU2d, cls).from_float(mod)


class ConvTransposeReLU62d(ConvTranspose2d):
    r"""
    A ConvTransposeReLU62d module attached with FakeQuantize modules for
    weight, used for quantization aware training.


    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvTransposeReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0],
                qat_weight,
                self._fake_quant_bias(),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = F.relu6(out, inplace=True)
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return QTensor(out, scale=None, dtype="float32")

        return _func

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
        return super(ConvTransposeReLU62d, cls).from_float(mod)


class ConvTransposeAdd2d(ConvTranspose2d):
    r"""
    A ConvTransposeAdd2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.


    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvTransposeAdd2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0],
                qat_weight,
                self._fake_quant_bias(),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = out + input[1]
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return QTensor(out, scale=None, dtype="float32")

        return _func

    def forward(self, input1, input2, output_size=None):
        output_padding = self._output_padding(
            input1.as_subclass(torch.Tensor),
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        return self._forward_func(output_padding)(
            input1.as_subclass(torch.Tensor), input2.as_subclass(torch.Tensor)
        )

    @classmethod
    def from_float(cls, mod):
        return super(ConvTransposeAdd2d, cls).from_float(mod)


class ConvTransposeAddReLU2d(ConvTransposeAdd2d):
    r"""
    A ConvTransposeAddReLU2d module attached with FakeQuantize modules for
    weight, used for quantization aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvTransposeAddReLU2d
    _version: int = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0],
                qat_weight,
                self._fake_quant_bias(),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = out + input[1]

            if self.activation_post_process is not None:

                out = hz.nn.qat.compatible_ops.relu(out)
                return self.activation_post_process(out)
            else:
                out = F.relu(out, inplace=True)
                return QTensor(out, scale=None, dtype="float32")

        return _func

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
        return super(ConvTransposeAddReLU2d, cls).from_float(mod)


class ConvTransposeAddReLU62d(ConvTransposeAdd2d):
    r"""
    A ConvTransposeAddReLU62d module attached with FakeQuantize modules for
    weight, used for quantization aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvTransposeAddReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeAddReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0],
                qat_weight,
                self._fake_quant_bias(),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = out + input[1]
            out = F.relu6(out, inplace=True)
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return QTensor(out, scale=None, dtype="float32")

        return _func

    @classmethod
    def from_float(cls, mod):
        return super(ConvTransposeAddReLU62d, cls).from_float(mod)
