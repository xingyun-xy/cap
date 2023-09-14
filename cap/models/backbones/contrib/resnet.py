# Copyright (c) Changan Auto. All rights reserved.
# Adjusted from torchvision models, for QAT examples.

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub

from cap.registry import OBJECT_REGISTRY

__all__ = ["ResNet", "resnet18", "resnet50"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # Replace torch.add with floatFunctional()
        self.short_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.short_add.add(out, identity)
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for
    # downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride
    # at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image
    # recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and
    # improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample
        # the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        # Replace torch.add with floatFunctional()
        self.short_add = nn.quantized.FloatFunctional()
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.short_add.add(out, identity)
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        flat_output=True,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block
        self.num_classes = num_classes
        self.flat_output = flat_output

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        # AdaptiveAvgPool is not supported now.
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # replace Linear with Conv2d
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and
        # each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        if self.flat_output:
            x = torch.flatten(x, 1)
        return x

    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        # fused modules conv1+bn1+relu1, layer1, layer2, layer3, layer4
        torch.quantization.fuse_modules(
            self,
            ["conv1", "bn1", "relu1"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in layer:
                if type(m) == BasicBlock:
                    if m.downsample is not None:
                        torch.quantization.fuse_modules(
                            m.downsample,
                            ["0", "1"],
                            inplace=True,
                            fuser_func=quantization.fuse_known_modules,
                        )
                    torch.quantization.fuse_modules(
                        m,
                        [
                            ["conv1", "bn1", "relu1"],
                            ["conv2", "bn2", "short_add", "relu2"],
                        ],
                        inplace=True,
                        fuser_func=quantization.fuse_known_modules,
                    )
                elif type(m) == Bottleneck:
                    if m.downsample is not None:
                        torch.quantization.fuse_modules(
                            m.downsample,
                            ["0", "1"],
                            inplace=True,
                            fuser_func=quantization.fuse_known_modules,
                        )
                    torch.quantization.fuse_modules(
                        m,
                        [
                            ["conv1", "bn1", "relu1"],
                            ["conv2", "bn2", "relu2"],
                            ["conv3", "bn3", "short_add", "relu3"],
                        ],
                        inplace=True,
                        fuser_func=quantization.fuse_known_modules,
                    )
                else:
                    pass

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        self.fc.qconfig = qconfig_manager.get_default_qat_out_qconfig()


def _resnet(arch, block, layers, pretrained_path, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        linear_weight = state_dict["fc.weight"]
        state_dict["fc.weight"] = linear_weight.view(
            linear_weight.shape + (1, 1)
        )
        model.load_state_dict(state_dict)
    return model


@OBJECT_REGISTRY.register
def resnet18(pretrained_path=None, **kwargs):  # noqa: D205,D400
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True,
            returns a model pre-trained on ImageNet.
        path (str): The path of pretrained model.
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained_path, **kwargs
    )


@OBJECT_REGISTRY.register
def resnet50(pretrained_path=None, **kwargs):  # noqa: D205,D400
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True,
            returns a model pre-trained on ImageNet
        path (str): The path of pretrained model.
    """
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained_path, **kwargs
    )
