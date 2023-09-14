# Copyright (c) Changan Auto. All rights reserved.
from typing import Iterable

import torch
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY
from .conv_module import ConvModule2d
from .separable_conv_module import SeparableGroupConvModule2d

__all__ = [
    "BasicVarGBlock",
    "BasicVarGBlockV2",
    "OnePathResUnit",
    "ExtendVarGNetFeatures",
    "TwoPathResUnit",
]


class BasicVarGBlock(nn.Module):
    """
    A basic block for vargnetv2.

    Args:
        in_channels (int): Input channels.
        mid_channels (int): Mid channels.
        out_channels (int): Output channels.
        stride (int): Stride of basic block.
        bn_kwargs (dict): Dict for BN layer.
        kernel_size (int): Kernel size of basic block.
        padding (int): Padding of basic block.
        bias (bool): Whether to use bias in basic block.
        factor (int): Factor for channels expansion.
        group_base (int): Group base for group conv.
        merge_branch (bool): Whether to merge branch.
        dw_with_relu (bool): Whether to use relu in dw conv.
        pw_with_relu (bool): Whether to use relu in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int,
        bn_kwargs: dict,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
        factor: int = 2,
        group_base: int = 8,
        merge_branch: bool = False,
        dw_with_relu: bool = True,
        pw_with_relu: bool = True,
    ):
        super(BasicVarGBlock, self).__init__()
        self.merge_branch = merge_branch
        self.downsample = None
        if not (stride == 1 and in_channels == out_channels):
            self.downsample = SeparableGroupConvModule2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=padding,
                factor=factor,
                groups=int(in_channels / group_base),
                stride=stride,
                dw_norm_layer=nn.BatchNorm2d(
                    int(in_channels * factor), **bn_kwargs
                ),
                dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
                pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )

        if self.merge_branch:
            self.body_conv = SeparableGroupConvModule2d(
                in_channels,
                mid_channels,
                kernel_size,
                bias=bias,
                padding=padding,
                factor=factor,
                groups=int(in_channels / group_base),
                stride=stride,
                dw_norm_layer=nn.BatchNorm2d(
                    int(in_channels * factor), **bn_kwargs
                ),
                dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
                pw_norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
            )
        self.conv = SeparableGroupConvModule2d(
            in_channels,
            mid_channels,
            kernel_size,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(in_channels / group_base),
            stride=stride,
            dw_norm_layer=nn.BatchNorm2d(
                int(in_channels * factor), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
        )
        self.out_conv = SeparableGroupConvModule2d(
            mid_channels,
            out_channels,
            kernel_size,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(mid_channels / group_base),
            stride=1,
            dw_norm_layer=nn.BatchNorm2d(
                int(mid_channels * factor), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
        )

        self.merge_add = nn.quantized.FloatFunctional()
        self.merge_act = nn.ReLU(inplace=True)
        self.shortcut_add = nn.quantized.FloatFunctional()
        self.out_act = nn.ReLU(inplace=True) if pw_with_relu else None

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.merge_branch:
            body = self.body_conv(x)
            x = self.conv(x)
            x = self.merge_add.add(x, body)
            x = self.merge_act(x)
        else:
            x = self.conv(x)
            x = self.merge_act(x)

        out = self.out_conv(x)
        out = self.shortcut_add.add(out, identity)
        if self.out_act is not None:
            out = self.out_act(out)
        return out

    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        if self.downsample is not None:
            self.downsample.fuse_model()
        if self.merge_branch:
            self.body_conv.fuse_model()
            getattr(self.conv, "0").fuse_model()
            torch.quantization.fuse_modules(
                self,
                ["conv.1.0", "conv.1.1", "merge_add", "merge_act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
        else:
            getattr(self.conv, "0").fuse_model()
            torch.quantization.fuse_modules(
                self,
                ["conv.1.0", "conv.1.1", "merge_act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

        getattr(self.out_conv, "0").fuse_model()
        fuse_list = ["out_conv.1.0", "out_conv.1.1", "shortcut_add"]
        if self.out_act is not None:
            fuse_list += ["out_act"]
        torch.quantization.fuse_modules(
            self,
            fuse_list,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class BasicVarGBlockV2(BasicVarGBlock):
    """
    A basic block for VargNASNet which inherits from BasicVarGBlock.

    The difference between BasicVarGBlockV2 and BasicVarGBlock is that
    `downsample` can be changed from `SeparableGroupConvModule2d` to
    `ConvModule2d`.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride of basic block.
        bn_kwargs (dict): Dict for BN layer.
        merge_branch (bool): Whether to merge branch.
        pw_with_relu (bool): Whether to use relu in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        bn_kwargs: dict,
        merge_branch: bool = False,
        pw_with_relu: bool = False,
        **kwargs
    ):
        self.ch_change = False
        if in_channels != out_channels and stride == 1:
            self.ch_change = True
            merge_branch = False
        super(BasicVarGBlockV2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bn_kwargs=bn_kwargs,
            merge_branch=merge_branch,
            pw_with_relu=pw_with_relu,
            **kwargs,
        )
        if self.ch_change:
            self.downsample = ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )


class OnePathResUnit(nn.Module):
    """
    ResUnit of One path.

    Args:
        dw_num_filter (int): Num filters of dw conv.
        group_base (int): Group base of group conv.
        pw_num_filter (int): Num filters of pw conv.
        pw_num_filter2 (int): Num filters of second pw conv.
        bn_kwargs (dict): Kwargs of BN layer.
        stride (int): Stride of group conv.
        is_dim_match (bool): Whether to use dim match.
        in_filter (:obj:'int', optional): Num filters of input. Default to
            None when in_filter == dw_num_filter
        use_bias (bool): Whether to use bias.
        pw_with_act (bool): Whether to use act of pw.
        factor (float): Factor of group conv.
    """

    def __init__(
        self,
        dw_num_filter,
        group_base,
        pw_num_filter,
        pw_num_filter2,
        bn_kwargs,
        stride,
        is_dim_match,
        in_filter=None,
        use_bias=False,
        pw_with_act=False,
        factor=2.0,
    ):
        super().__init__()
        assert dw_num_filter % group_base == 0
        assert pw_num_filter % group_base == 0

        if in_filter is None:
            in_filter = dw_num_filter
        if not is_dim_match:
            self.short_cut = SeparableGroupConvModule2d(
                in_channels=in_filter,
                dw_channels=dw_num_filter,
                out_channels=pw_num_filter2,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=use_bias,
                factor=factor,
                groups=dw_num_filter // group_base,
                dw_norm_layer=nn.BatchNorm2d(
                    int(dw_num_filter * factor), **bn_kwargs
                ),
                dw_act_layer=nn.ReLU(inplace=True),
                pw_norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            )

        self.sep1 = SeparableGroupConvModule2d(
            in_channels=in_filter,
            dw_channels=dw_num_filter,
            out_channels=pw_num_filter,
            kernel_size=3,
            stride=stride,
            padding=1,
            factor=factor,
            groups=dw_num_filter // group_base,
            bias=use_bias,
            dw_norm_layer=nn.BatchNorm2d(
                int(dw_num_filter * factor), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=True),
            pw_norm_layer=nn.BatchNorm2d(pw_num_filter, **bn_kwargs),
            pw_act_layer=nn.ReLU(inplace=True),
        )

        self.sep2 = SeparableGroupConvModule2d(
            in_channels=pw_num_filter,
            out_channels=pw_num_filter2,
            kernel_size=3,
            stride=1,
            padding=1,
            factor=factor,
            groups=pw_num_filter // group_base,
            bias=use_bias,
            dw_norm_layer=nn.BatchNorm2d(
                int(pw_num_filter * factor), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=True),
            pw_act_layer=None,
            pw_norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
        )
        self.sep2_add = nn.quantized.FloatFunctional()
        self.sep2_relu = nn.ReLU(inplace=True) if pw_with_act else None

    def forward(self, x):
        short_cut = self.short_cut(x) if hasattr(self, "short_cut") else x

        out = self.sep1(x)
        out = self.sep2(out)

        out = self.sep2_add.add(out, short_cut)
        if self.sep2_relu:
            out = self.sep2_relu(out)
        return out

    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        if hasattr(self, "short_cut"):
            self.short_cut.fuse_model()
        self.sep1.fuse_model()
        getattr(self.sep2, "0").fuse_model()

        relu = ["sep2_relu"] if self.sep2_relu else []
        torch.quantization.fuse_modules(
            self,
            ["sep2.1.0", "sep2.1.1", "sep2_add"] + relu,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class TwoPathResUnit(nn.Module):
    """
    ResUnit of Two path.

    Args:
        dw_num_filter: Num filters of dw conv.
        group_base: Group base of group conv.
        pw_num_filter: Num filters of pw conv.
        pw_num_filter2: Num filters of second pw conv.
        bn_kwargs: Kwargs of BN layer.
        stride: Stride of group conv.
        is_dim_match: Whether to use dim match.
        use_bias: Whether to use bias.
        pw_with_act: Whether to use act of pw.
        factor: Factor of group conv.
        fix_group_base: Whether to fix group_base.
    """

    def __init__(
        self,
        dw_num_filter: int,
        group_base: int,
        pw_num_filter: int,
        pw_num_filter2: int,
        bn_kwargs: dict,
        stride: int,
        is_dim_match: bool,
        use_bias: bool = False,
        pw_with_act: bool = False,
        factor: int = 2,
        fix_group_base: bool = False,
    ):
        super().__init__()

        self.pw_with_act = pw_with_act

        assert dw_num_filter % group_base == 0
        assert pw_num_filter % group_base == 0

        if fix_group_base:
            assert group_base % int(factor) == 0
            group_base = group_base // int(factor)
        else:
            group_base = group_base

        if is_dim_match:
            self.short_cut = None
            self.p1_conv1 = SeparableGroupConvModule2d(
                kernel_size=3,
                stride=stride,
                padding=1,
                in_channels=int(dw_num_filter * factor),
                dw_channels=int(dw_num_filter * factor),
                out_channels=pw_num_filter,
                factor=factor,
                groups=dw_num_filter // group_base,
                pw_act_layer=nn.ReLU(inplace=True),
                bias=use_bias,
                dw_norm_layer=nn.BatchNorm2d(
                    int(dw_num_filter * factor), **bn_kwargs
                ),
                pw_norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            )
            self.p1_conv2 = None
        else:
            self.short_cut = SeparableGroupConvModule2d(
                kernel_size=3,
                stride=stride,
                padding=1,
                in_channels=int(dw_num_filter * factor),
                dw_channels=int(dw_num_filter * factor),
                out_channels=pw_num_filter2,
                factor=factor,
                groups=dw_num_filter // group_base,
                dw_act_layer=nn.ReLU(inplace=True),
                pw_act_layer=None,
                bias=use_bias,
                dw_norm_layer=nn.BatchNorm2d(
                    int(dw_num_filter * factor), **bn_kwargs
                ),
                pw_norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            )
            self.p1_conv1 = SeparableGroupConvModule2d(
                kernel_size=3,
                stride=stride,
                padding=1,
                in_channels=int(dw_num_filter * factor),
                dw_channels=int(dw_num_filter * factor),
                out_channels=pw_num_filter,
                factor=factor,
                groups=dw_num_filter // group_base,
                dw_act_layer=nn.ReLU(inplace=True),
                pw_act_layer=None,
                bias=use_bias,
                dw_norm_layer=nn.BatchNorm2d(
                    int(dw_num_filter * factor), **bn_kwargs
                ),
                pw_norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            )
            self.p1_conv2 = SeparableGroupConvModule2d(
                kernel_size=3,
                stride=stride,
                padding=1,
                in_channels=int(dw_num_filter * factor),
                dw_channels=int(dw_num_filter * factor),
                out_channels=pw_num_filter,
                factor=factor,
                groups=dw_num_filter // group_base,
                dw_act_layer=nn.ReLU(inplace=True),
                pw_act_layer=None,
                bias=use_bias,
                dw_norm_layer=nn.BatchNorm2d(
                    int(dw_num_filter * factor), **bn_kwargs
                ),
                pw_norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            )
            self.qat_add = nn.quantized.FloatFunctional()
            self.relu = nn.ReLU(inplace=True)

        self.p2_conv = SeparableGroupConvModule2d(
            kernel_size=3,
            stride=1,
            padding=1,
            in_channels=int(pw_num_filter * factor),
            dw_channels=int(pw_num_filter * factor),
            out_channels=pw_num_filter2,
            factor=factor,
            groups=pw_num_filter // group_base,
            dw_act_layer=nn.ReLU(inplace=True),
            pw_act_layer=None,
            dw_norm_layer=nn.BatchNorm2d(
                int(pw_num_filter * factor), **bn_kwargs
            ),
            pw_norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            bias=use_bias,
        )

        self.qat_add2 = nn.quantized.FloatFunctional()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        if self.short_cut is None:
            short_cut = x
            p1_out = self.p1_conv1(x)
        else:
            short_cut = self.short_cut(x)
            p1_out = self.qat_add.add(self.p1_conv2(x), self.p1_conv1(x))
            p1_out = self.relu(p1_out)

        p2_out = self.qat_add2.add(self.p2_conv(p1_out), short_cut)

        if self.pw_with_act:
            p2_out = self.relu2(p2_out)

        return p2_out

    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        self.p1_conv1.fuse_model()

        if self.short_cut is not None:
            self.short_cut.fuse_model()
            getattr(self.p1_conv2, "0").fuse_model()
            torch.quantization.fuse_modules(
                self,
                ["p1_conv2.1.0", "p1_conv2.1.1", "qat_add", "relu"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

        act = ["relu2"] if self.pw_with_act else []

        getattr(self.p2_conv, "0").fuse_model()
        torch.quantization.fuse_modules(
            self,
            ["p2_conv.1.0", "p2_conv.1.1", "qat_add2"] + act,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


@OBJECT_REGISTRY.register
class ExtendVarGNetFeatures(nn.Module):
    """
    Extend features.

    Args:
        prev_channel (int): Input channels.
        channels (list): Channels of output featuers.
        num_units (list): The number of units of each extend stride.
        group_base (int): The number of channels per group.
        bn_kwargs (dict): BatchNormEx kwargs.
        factor (float, optional): Channel factor, by default 2.0
        dropout_kwargs (dict, optional): QuantiDropout kwargs,
            None means do not use drop, by default None
    """

    def __init__(
        self,
        prev_channel,
        channels,
        num_units,
        group_base,
        bn_kwargs,
        factor=2.0,
        dropout_kwargs=None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        if dropout_kwargs is not None:
            self.dropout = nn.Dropout(**dropout_kwargs)

        channels = channels if isinstance(channels, Iterable) else [channels]
        num_units = (
            num_units if isinstance(num_units, Iterable) else [num_units]
        )

        for channel_i, num_unit_i in zip(channels, num_units):
            block = []
            assert num_unit_i >= 1
            for n_idx in range(num_unit_i):
                block.append(
                    OnePathResUnit(
                        dw_num_filter=prev_channel
                        if n_idx == 0
                        else channel_i,
                        group_base=group_base,
                        pw_num_filter=channel_i,
                        pw_num_filter2=channel_i,
                        is_dim_match=n_idx != 0,
                        stride=2 if n_idx == 0 else 1,
                        pw_with_act=False,
                        bn_kwargs=bn_kwargs,
                        use_bias=True,
                        factor=factor,
                    )
                )

            if dropout_kwargs is not None:
                block.append(nn.Dropout(**dropout_kwargs))
            self.blocks.append(nn.Sequential(*block))
            prev_channel = channel_i

    def forward(self, features):
        extend_features = list(features)
        if hasattr(self, "dropout"):
            extend_features[-1] = self.dropout(extend_features[-1])
        for block in self.blocks:
            extend_features.append(block(extend_features[-1]))

        return extend_features

    def fuse_model(self):
        for block in self.blocks:
            for m in block:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
