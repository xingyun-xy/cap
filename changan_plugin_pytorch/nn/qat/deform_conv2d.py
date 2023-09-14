from typing import List

import torch
from changan_plugin_pytorch.nn import intrinsic
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import functional as F
from torchvision import ops

__all__ = [
    "DeformConv2d",
    "DeformConvReLU2d",
    "DeformConvReLU62d",
    "DeformConvAdd2d",
    "DeformConvAddReLU2d",
    "DeformConvAddReLU62d",
]


class DeformConv2d(ops.DeformConv2d):

    _FLOAT_MODULE = ops.DeformConv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        assert qconfig is not None, "qconfig must be provided for QAT module"
        self.qconfig = qconfig

        self.weight_fake_quant = self.qconfig.weight(channel_len=out_channels)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation(
                channel_len=out_channels
            )

    def _conv_forward(
        self, input: QTensor, offset: QTensor, mask: QTensor = None
    ):
        return ops.deform_conv2d(
            input.as_subclass(torch.Tensor),
            offset.as_subclass(torch.Tensor),
            self.weight_fake_quant(self.weight).as_subclass(torch.Tensor),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def forward(self, input: QTensor, offset: QTensor, mask: QTensor = None):
        out = self._conv_forward(input, offset, mask)
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
        assert (
            mod.qconfig is not None
        ), "Input float module must have a valid qconfig"

        if type(mod) in (
            intrinsic.DeformConvAdd2d,
            intrinsic.DeformConvAddReLU2d,
            intrinsic.DeformConvAddReLU62d,
            intrinsic.DeformConvReLU2d,
            intrinsic.DeformConvReLU62d,
        ):
            mod = mod.deform_conv2d

        qat_mod = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            qconfig=mod.qconfig,
        )
        qat_mod.weight.copy_(mod.weight)
        if mod.bias is not None:
            qat_mod.bias.copy_(mod.bias)
        return qat_mod


class DeformConvReLU2d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvReLU2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def forward(self, input: QTensor, offset: QTensor, mask: QTensor = None):
        out = self._conv_forward(input, offset, mask)
        out = F.relu(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")


class DeformConvReLU62d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvReLU62d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def forward(self, input: QTensor, offset: QTensor, mask: QTensor = None):
        out = self._conv_forward(input, offset, mask)
        out = F.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")


class DeformConvAdd2d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvAdd2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1: List[QTensor], x2: QTensor):
        out = self._conv_forward(*x1)
        out = torch.add(out, x2.as_subclass(torch.Tensor))
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")


class DeformConvAddReLU2d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvAddReLU2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1: List[QTensor], x2: QTensor):
        out = self._conv_forward(*x1)
        out = torch.add(out, x2.as_subclass(torch.Tensor))
        out = F.relu(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")


class DeformConvAddReLU62d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvAddReLU62d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvAddReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1: List[QTensor], x2: QTensor):
        out = self._conv_forward(*x1)
        out = torch.add(out, x2.as_subclass(torch.Tensor))
        out = F.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, None, "float32")
