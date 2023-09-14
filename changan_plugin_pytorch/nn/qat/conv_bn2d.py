"""
fused conv2d+add+relu modules
"""
import copy

import torch
import torch.nn.functional as F
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import intrinsic
from changan_plugin_pytorch.qat_mode import QATMode, get_qat_mode
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
import changan_plugin_pytorch as hz
from .. import intrinsic
from .conv2d import Conv2d

__all__ = [
    "ConvBN2d",
    "ConvBNReLU2d",
    "ConvBNAdd2d",
    "ConvBNAddReLU2d",
    "ConvBNReLU62d",
    "ConvBNAddReLU62d",
]


class ConvBN2d(Conv2d):
    r"""
    A ConvBN2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.
    """
    _FLOAT_MODULE = intrinsic.ConvBN2d

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
        super(ConvBN2d, self).__init__(
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
        if get_qat_mode() == QATMode.WithBNReverseFold:
            march = get_march()

            assert march != March.BERNOULLI, (
                f"{QATMode.WithBNReverseFold} mode is "
                f"not supported for {march}"
            )

    def _fuse_bn_weight(self):
        assert self.bn.running_var is not None
        running_std = torch.rsqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight * running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        fused_weight = self.weight * scale_factor.reshape(weight_shape)
        return fused_weight, scale_factor

    def _forward_with_fold_and_reverse_bn(self, input):
        fused_weight, scale_factor = self._fuse_bn_weight()
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(fused_weight)
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(
                self.out_channels,
                device=scaled_weight.as_subclass(torch.Tensor).device,
            )
        conv = self._conv_forward(
            input, scaled_weight.as_subclass(torch.Tensor), zero_bias
        )
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self._fake_quant_bias().reshape(bias_shape)
        # adapt for torch.cuda.amp.autocast
        conv_bn_out = self.bn(conv_orig.to(conv.dtype))
        return conv_bn_out

    def _get_weight_for_fake_quant(self):
        # for calibration
        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            return self._fuse_bn_weight()[0]
        else:
            return super(ConvBN2d, self)._get_weight_for_fake_quant()

    def forward(self, input):
        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            out = self._forward_with_fold_and_reverse_bn(input)
        else:
            out = self._conv_forward(
                input,
                self.weight_fake_quant(self.weight),
                self._fake_quant_bias(),
            )
            out = self.bn(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

    def fuse_norm(self, inplace=False):
        if inplace:
            (self.weight, self.bias) = nn.utils.fuse_conv_bn_weights(
                self.weight,
                self.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )
            self.bn = nn.Identity()
            return self
        else:
            fused_conv = copy.deepcopy(self)
            (
                fused_conv.weight,
                fused_conv.bias,
            ) = nn.utils.fuse_conv_bn_weights(
                self.weight,
                self.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )
            fused_conv.bn = nn.Identity()
            return fused_conv

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
        conv = mod.conv
        qconfig = mod.qconfig
        qat_conv_bn = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            qconfig=qconfig,
        )
        qat_conv_bn.weight = conv.weight
        qat_conv_bn.bias = conv.bias
        qat_conv_bn.bn = mod.bn
        return qat_conv_bn


class ConvBNReLU2d(ConvBN2d):
    r"""
    A ConvBNReLU2d module is a fused module of Conv2d and BatchNorm2d and
    ReLU, attached with FakeQuantize modules for weight for quantization
    aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvBNReLU2d
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
        super(ConvBNReLU2d, self).__init__(
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
        from ...quantization import march

        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            out = self._forward_with_fold_and_reverse_bn(input)
        else:
            out = self._conv_forward(
                input,
                self.weight_fake_quant(self.weight),
                self._fake_quant_bias(),
            )
            out = self.bn(out)

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
        return super(ConvBNReLU2d, cls).from_float(mod)


class ConvBNReLU62d(ConvBN2d):
    r"""
    A ConvReLU62d module is a fused module of Conv2d and BatchNorm2d and ReLU,
    attached with FakeQuantize modules for weight for quantization aware
    training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = intrinsic.ConvBNReLU62d

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
        super(ConvBNReLU62d, self).__init__(
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
        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            out = self._forward_with_fold_and_reverse_bn(input)
        else:
            out = self._conv_forward(
                input,
                self.weight_fake_quant(self.weight),
                self._fake_quant_bias(),
            )
            out = self.bn(out)
        out = F.relu6(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

    @classmethod
    def from_float(cls, mod):
        return super(ConvBNReLU62d, cls).from_float(mod)


class ConvBNAdd2d(ConvBN2d):
    _FLOAT_MODULE = intrinsic.ConvBNAdd2d

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
        super(ConvBNAdd2d, self).__init__(
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
        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            out = self._forward_with_fold_and_reverse_bn(input1)
        else:
            out = self._conv_forward(
                input1,
                self.weight_fake_quant(self.weight),
                self._fake_quant_bias(),
            )
            out = self.bn(out)
        out = out + input2.as_subclass(torch.Tensor)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

    @classmethod
    def from_float(cls, mod):
        return super(ConvBNAdd2d, cls).from_float(mod)


class ConvBNAddReLU2d(ConvBNAdd2d):
    _FLOAT_MODULE = intrinsic.ConvBNAddReLU2d
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
        super(ConvBNAddReLU2d, self).__init__(
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
        from ...quantization import march

        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            out = self._forward_with_fold_and_reverse_bn(input1)
        else:
            out = self._conv_forward(
                input1,
                self.weight_fake_quant(self.weight),
                self._fake_quant_bias(),
            )
            out = self.bn(out)
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
        return super(ConvBNAddReLU2d, cls).from_float(mod)


class ConvBNAddReLU62d(ConvBNAdd2d):
    _FLOAT_MODULE = intrinsic.ConvBNAddReLU62d

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
        super(ConvBNAddReLU62d, self).__init__(
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
        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            out = self._forward_with_fold_and_reverse_bn(input1)
        else:
            out = self._conv_forward(
                input1,
                self.weight_fake_quant(self.weight),
                self._fake_quant_bias(),
            )
            out = self.bn(out)
        out = out + input2.as_subclass(torch.Tensor)
        out = F.relu6(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")

    @classmethod
    def from_float(cls, mod):
        return super(ConvBNAddReLU62d, cls).from_float(mod)
