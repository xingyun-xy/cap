# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Tuple, Union

import torch.nn as nn

from .conv_module import ConvModule2d

__all__ = ["SeparableConvModule2d", "SeparableGroupConvModule2d"]


class SeparableConvModule2d(nn.Sequential):
    """
    Depthwise sparable convolution module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        dw_norm_layer (nn.Module): Normalization layer in dw conv.
        dw_act_layer (nn.Module): Activation layer in dw conv.
        pw_norm_layer (nn.Module): Normalization layer in pw conv.
        pw_act_layer (nn.Module): Activation layer in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dw_norm_layer: Union[None, nn.Module] = None,
        dw_act_layer: Union[None, nn.Module] = None,
        pw_norm_layer: Union[None, nn.Module] = None,
        pw_act_layer: Union[None, nn.Module] = None,
    ):
        super(SeparableConvModule2d, self).__init__(
            ConvModule2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                in_channels,
                bias,
                padding_mode,
                dw_norm_layer,
                dw_act_layer,
            ),
            ConvModule2d(
                in_channels,
                out_channels,
                1,
                bias=bias,
                norm_layer=pw_norm_layer,
                act_layer=pw_act_layer,
            ),
        )

    def fuse_model(self):
        dw = getattr(self, "0")
        pw = getattr(self, "1")
        if hasattr(dw, "fuse_model"):
            dw.fuse_model()
        if hasattr(pw, "fuse_model"):
            pw.fuse_model()


class SeparableGroupConvModule2d(nn.Sequential):
    """
    Separable group convolution module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        dw_channels (:obj:'int', optional): Number of dw conv output channels.
            Default to None when dw_channels == in_channels.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        dw_norm_layer (nn.Module): Normalization layer in group conv.
        dw_act_layer (nn.Module): Activation layer in group conv.
        pw_norm_layer (nn.Module): Normalization layer in pw conv.
        pw_act_layer (nn.Module): Activation layer in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        dw_channels: Optional[int] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        factor: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dw_norm_layer: Union[None, nn.Module] = None,
        dw_act_layer: Union[None, nn.Module] = None,
        pw_norm_layer: Union[None, nn.Module] = None,
        pw_act_layer: Union[None, nn.Module] = None,
    ):
        if dw_channels is None:
            dw_channels = in_channels

        super(SeparableGroupConvModule2d, self).__init__(
            ConvModule2d(
                in_channels,
                int(dw_channels * factor),
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
                dw_norm_layer,
                dw_act_layer,
            ),
            ConvModule2d(
                int(dw_channels * factor),
                out_channels,
                1,
                bias=bias,
                norm_layer=pw_norm_layer,
                act_layer=pw_act_layer,
            ),
        )

    def fuse_model(self):
        dw = getattr(self, "0")
        pw = getattr(self, "1")
        if hasattr(dw, "fuse_model"):
            dw.fuse_model()
        if hasattr(pw, "fuse_model"):
            pw.fuse_model()
