# Copyright (c) Changan Auto. All rights reserved.

import logging
import os
from typing import Optional, Tuple, Union

import changan_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import cap.core.saved_tensor as savedtensor

__all__ = ["ConvModule2d", "ConvTransposeModule2d", "ConvUpsample2d"]


class ConvModule2d(nn.Sequential):
    """
    A conv block that bundles conv/norm/activation layers.

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
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        conv_list = [conv, norm_layer, act_layer]
        self.conv_list = [layer for layer in conv_list if layer is not None]
        super(ConvModule2d, self).__init__(*self.conv_list)
        st = bool(int(os.environ.get("CAP_USE_SAVEDTENSOR", "0")))
        if st and norm_layer is not None and savedtensor.check_support():
            self.st_fn = savedtensor.get_fn_conv_bn(
                conv, norm_layer, act_layer
            )
        else:
            self.st_fn = None

    def forward(self, x):
        cp = bool(int(os.environ.get("CAP_USE_CHECKPOINT", "0")))
        if cp and torch.is_tensor(x) and x.requires_grad:
            out = checkpoint.checkpoint(super().forward, x)
        elif self.st_fn is not None and self.training and torch.is_tensor(x):
            out = self.st_fn(x)
        else:
            out = super().forward(x)
        return out

    def fuse_model(self):
        if self.st_fn is not None:
            # remove saved tensor fn after fused module,
            # because there is only one conv remained after fused.
            self.st_fn = None

        if len(self.conv_list) > 1 and not isinstance(
            self.conv_list[-1], (nn.BatchNorm2d, nn.ReLU, nn.ReLU6)
        ):
            # not: conv2d+bn, conv2d+relu(6), conv2d+bn+relu(6)
            self.conv_list.pop()

        if len(self.conv_list) <= 1:
            # nn.Conv2d
            return

        try:
            from changan_plugin_pytorch import quantization

            fuser_func = quantization.fuse_known_modules
        except Warning:
            logging.warning(
                "Please install changan_plugin_pytorch first, otherwise use "
                "pytorch official quantification"
            )
            from torch.quantization.fuse_modules import fuse_known_modules

            fuser_func = fuse_known_modules

        fuse_list = ["0", "1", "2"]
        self.fuse_list = fuse_list[: len(self.conv_list)]
        torch.quantization.fuse_modules(
            self,
            self.fuse_list,
            inplace=True,
            fuser_func=fuser_func,
        )


class ConvTransposeModule2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: bool = 1,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        """Transposed convolution, followed by normalization and activation.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): kernel size.
            stride (Union[int, Tuple[int, int]], optional): conv stride.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): conv padding.
                dilation * (kernel_size - 1) - padding zero-padding will be
                added to the input. Defaults to 0.
            output_padding (Union[int, Tuple[int, int]], optional):
                additional size added to the output. Defaults to 0.
            groups (int, optional): number of blocked connections from input
                to output. Defaults to 1.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            dilation (bool, optional): kernel dilation. Defaults to 1.
            padding_mode (str, optional): same as conv2d. Defaults to 'zeros'.
            norm_layer (Optional[nn.Module], optional): normalization layer.
                Defaults to None.
            act_layer (Optional[nn.Module], optional): activation layer.
                Defaults to None.
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.norm = norm_layer
        self.act = act_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def fuse_model(self):
        try:
            from changan_plugin_pytorch import quantization

            fuser_func = quantization.fuse_known_modules
        except Warning:
            logging.warning(
                "Please install changan_plugin_pytorch first, otherwise use "
                "pytorch official quantification."
            )
            from torch.quantization.fuse_modules import fuse_known_modules

            fuser_func = fuse_known_modules
        fuse_list = ["conv"]
        if self.norm is not None:
            fuse_list.append("norm")
        if self.act is not None:
            fuse_list.append("act")
        if len(fuse_list) <= 1:  # nn.Conv2d only
            return
        torch.quantization.fuse_modules(
            self, fuse_list, inplace=True, fuser_func=fuser_func
        )


class ConvUpsample2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        """Conv upsample module.

        Different from ConvTransposeModule2d, this module does conv2d,
        followed by an upsample layer. The final effect is almost the same,
        but one should pay attention to the output size.

        Args:
            in_channels (int): same as nn.Conv2d.
            out_channels (int): same as nn.Conv2d.
            kernel_size (Union[int, Tuple[int, int]]): same as nn.Conv2d.
            stride (Union[int, Tuple[int, int]], optional): Upsample stride.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): same as nn.Conv2d.
                Defaults to 0.
            dilation (Union[int, Tuple[int, int]], optional): same as
                nn.Conv2d. Defaults to 1.
            groups (int, optional): same as nn.Conv2d. Defaults to 1.
            bias (bool, optional): same as nn.Conv2d. Defaults to True.
            padding_mode (str, optional): same as nn.Conv2d.
                Defaults to "zeros".
            norm_layer (Optional[nn.Module], optional): normalization layer.
                Defaults to None.
            act_layer (Optional[nn.Module], optional): activation layer.
                Defaults to None.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.norm = norm_layer
        self.act = act_layer
        self.up = hnn.Interpolate(
            scale_factor=stride, recompute_scale_factor=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        x = self.up(x)
        return x

    def fuse_model(self):
        try:
            from changan_plugin_pytorch import quantization

            fuser_func = quantization.fuse_known_modules
        except Warning:
            logging.warning(
                "Please install changan_plugin_pytorch first, otherwise use "
                "pytorch official quantification."
            )
            from torch.quantization.fuse_modules import fuse_known_modules

            fuser_func = fuse_known_modules
        fuse_list = ["conv"]
        if self.norm is not None:
            fuse_list.append("norm")
        if self.act is not None:
            fuse_list.append("act")
        if len(fuse_list) <= 1:  # nn.Conv2d only
            return
        torch.quantization.fuse_modules(
            self, fuse_list, inplace=True, fuser_func=fuser_func
        )
