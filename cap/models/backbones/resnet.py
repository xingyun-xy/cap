# Copyright (c) Changan Auto. All rights reserved.
import numpy as np
import torch.nn as nn
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import (
    BasicResBlock,
    BottleNeck,
    ConvModule2d,
    ExtendVarGNetFeatures,
)
from cap.registry import OBJECT_REGISTRY

__all__ = ["ResNet18", "ResNet50", "ResNet", "ResNet18V2", "ResNet50V2", "ResNetV2"]


class ResNet(nn.Module):
    """
    A module of resnet.

    Args:
        num_classes (int): Num classes of output layer.
        basic_block (nn.Module): Basic block for resnet.
        expansion (int): expansion of channels in basic_block.
        unit (list): Unit num for each block.
        channels_list (list): Channels for each block.
        bn_kwargs (dict): Dict for BN layer.
        bias (bool): Whether to use bias in module.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
    """

    def __init__(
        self,
        num_classes: int,
        basic_block: nn.Module,
        expansion: int,
        unit: list,
        channels_list: list,
        bn_kwargs: dict,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        super(ResNet, self).__init__()
        self.basic_block = basic_block
        self.expansion = expansion
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.include_top = include_top
        self.flat_output = flat_output
        self.num_classes = num_classes
        self.in_channels = channels_list[0]

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.mod1 = nn.Sequential(
            ConvModule2d(
                3,
                channels_list[0],
                7,
                stride=2,
                padding=3,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels_list[0], **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.mod2 = self._make_stage(channels_list[1], unit[0], 1)
        self.mod3 = self._make_stage(channels_list[2], unit[1], 2)
        self.mod4 = self._make_stage(channels_list[3], unit[2], 2)
        self.mod5 = self._make_stage(channels_list[4], unit[3], 2)

        if self.include_top:
            self.output = nn.Sequential(
                nn.AvgPool2d(7),
                ConvModule2d(
                    channels_list[4] * self.expansion,
                    num_classes,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(self, channels, repeats, stride):
        layers = []
        layers.append(
            self.basic_block(
                self.in_channels,
                channels,
                self.bn_kwargs,
                stride,
                bias=self.bias,
                expansion=self.expansion,
            )
        )
        self.in_channels = channels * self.expansion
        for _ in range(1, repeats):
            layers.append(
                self.basic_block(
                    self.in_channels,
                    channels,
                    self.bn_kwargs,
                    bias=self.bias,
                    expansion=self.expansion,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        x = self.quant(x)
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]
        if self.include_top:
            modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class ResNet18(ResNet):
    """
    A module of resnet18.

    Args:
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        bias (bool): Whether to use bias in module.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
    """

    def __init__(
        self,
        num_classes: int,
        bn_kwargs: dict,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        unit = [2, 2, 2, 2]
        block = BasicResBlock
        expansion = 1
        channels_list = [64, 64, 128, 256, 512]
        super(ResNet18, self).__init__(
            num_classes,
            block,
            expansion,
            unit,
            channels_list,
            bn_kwargs,
            bias,
            include_top,
            flat_output,
        )


@OBJECT_REGISTRY.register
class ResNet50(ResNet):
    """
    A module of resnet50.

    Args:
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        bias (bool): Whether to use bias in module.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
    """

    def __init__(
        self,
        num_classes: int,
        bn_kwargs: dict,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        unit = [3, 4, 6, 3]
        block = BottleNeck
        expansion = 4
        channels_list = [64, 64, 128, 256, 512]
        super(ResNet50, self).__init__(
            num_classes,
            block,
            expansion,
            unit,
            channels_list,
            bn_kwargs,
            bias,
            include_top,
            flat_output,
        )


@OBJECT_REGISTRY.register
class ResNetV2(nn.Module):
    """
    A module of resnetv2.

    Args:
        num_classes : Num classes of output layer.
        basic_block : Basic block for resnet.
        expansion : expansion of channels in basic_block.
        unit : Unit num for each block.
        channels_list : Channels for each block.
        group_base : Group base for ExtendVarGNetFeatures.
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        extend_features : Whether to extend features.
        include_top : Whether to include output layer.
        flat_output : Whether to view the output tensor.
    """

    def __init__(
        self,
        num_classes: int,
        basic_block: nn.Module,
        expansion: int,
        unit: list,
        channels_list: list,
        group_base: int,
        bn_kwargs: dict,
        bias: bool = True,
        extend_features: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        super(ResNetV2, self).__init__()
        self.basic_block = basic_block
        self.expansion = expansion
        self.bias = bias
        self.group_base = group_base
        self.bn_kwargs = bn_kwargs
        self.extend_features = extend_features
        self.include_top = include_top
        self.flat_output = flat_output
        self.num_classes = num_classes
        self.in_channels = channels_list[0]

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.mod1 = nn.Sequential(
            ConvModule2d(
                3,
                channels_list[0],
                7,
                stride=2,
                padding=3,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels_list[0], **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.mod2 = self._make_stage(channels_list[1], unit[0], 1)
        self.mod3 = self._make_stage(channels_list[2], unit[1], 2)
        self.mod4 = self._make_stage(channels_list[3], unit[2], 2)
        self.mod5 = self._make_stage(channels_list[4], unit[3], 2)

        if extend_features:
            self.ext = ExtendVarGNetFeatures(
                prev_channel=channels_list[-1] * expansion,
                channels=channels_list[-1] * expansion,
                num_units=2,
                group_base=group_base,
                bn_kwargs=bn_kwargs,
            )

        if self.include_top:
            self.output = nn.Sequential(
                nn.AvgPool2d(7),
                ConvModule2d(
                    channels_list[-1] * self.expansion,
                    num_classes,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(self, channels, repeats, stride):
        layers = []
        layers.append(
            self.basic_block(
                self.in_channels,
                channels,
                self.bn_kwargs,
                stride,
                bias=self.bias,
                expansion=self.expansion,
            )
        )
        self.in_channels = channels * self.expansion
        for _ in range(1, repeats):
            layers.append(
                self.basic_block(
                    self.in_channels,
                    channels,
                    self.bn_kwargs,
                    bias=self.bias,
                    expansion=self.expansion,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        
        # input_save = np.array(x.cpu()).flatten()
        # np.savetxt("bev_input/input.txt",input_save,fmt="%.8f")

        output = []
        x = self.quant(x)
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
        if self.extend_features:
            output = self.ext(output)
        # Fit strides of FPN inputs, feature maps strides in [2, 4, 8, 16, 32].
        output[0] = nn.functional.interpolate(
            output[0], scale_factor=2, mode="bilinear"
        )
        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]
        if self.include_top:
            modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class ResNet18V2(ResNetV2):
    """
    A module of resnet50V2.

    Args:
        num_classes : Num classes of output layer.
        group_base : Group base for ExtendVarGNetFeatures.
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        extend_features : Whether to extend features.
        include_top : Whether to include output layer.
        flat_output : Whether to view the output tensor.
    """

    def __init__(
        self,
        num_classes: int,
        group_base: int,
        bn_kwargs: dict,
        bias: bool = True,
        extend_features: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        unit = [2, 2, 2, 2]
        block = BottleNeck
        expansion = 1
        channels_list = [64, 64, 128, 256, 512]
        super(ResNet18V2, self).__init__(
            num_classes,
            block,
            expansion,
            unit,
            channels_list,
            group_base,
            bn_kwargs,
            bias,
            extend_features,
            include_top,
            flat_output,
        )

@OBJECT_REGISTRY.register
class ResNet50V2(ResNetV2):
    """
    A module of resnet50V2.

    Args:
        num_classes : Num classes of output layer.
        group_base : Group base for ExtendVarGNetFeatures.
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        extend_features : Whether to extend features.
        include_top : Whether to include output layer.
        flat_output : Whether to view the output tensor.
    """

    def __init__(
        self,
        num_classes: int,
        group_base: int,
        bn_kwargs: dict,
        bias: bool = True,
        extend_features: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        unit = [3, 4, 6, 3]
        block = BottleNeck
        expansion = 4
        channels_list = [64, 64, 128, 256, 512]
        super(ResNet50V2, self).__init__(
            num_classes,
            block,
            expansion,
            unit,
            channels_list,
            group_base,
            bn_kwargs,
            bias,
            extend_features,
            include_top,
            flat_output,
        )
