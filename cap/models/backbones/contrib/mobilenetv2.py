# Copyright (c) Changan Auto. All rights reserved.
# Adjusted from torchvision models, for QAT examples.

import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub

from cap.registry import OBJECT_REGISTRY

__all__ = ["MobileNetV2", "mobilenet_v2"]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        norm_layer=None,
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            # Repalce with ReLU
            nn.ReLU(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(
                    inp, hidden_dim, kernel_size=1, norm_layer=norm_layer
                )
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            # return x + self.conv(x)
            return self.skip_add.add(self.conv(x), x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        block=None,
        norm_layer=None,
        flat_output=True,
    ):  # noqa: D403
        """
        MobileNet V2 main class.

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts
                number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in
                each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building
                block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        self.flat_output = flat_output

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element,
        # assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        # building first layer
        input_channel = _make_divisible(
            input_channel * width_mult, round_nearest
        )
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features = [
            ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            # replace Linear with Conv2d
            # nn.Dropout(0.2),
            # nn.Linear(self.last_channel, num_classes),
            nn.Identity(),
            nn.Conv2d(self.last_channel, num_classes, kernel_size=1),
        )
        self.pool = nn.AvgPool2d(7, stride=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        x = self.dequant(x)
        if self.flat_output:
            x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    # This operation does not change the numerics
    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(
                    m,
                    ["0", "1", "2"],
                    inplace=True,
                    fuser_func=quantization.fuse_known_modules,
                )
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        if not m.use_res_connect:
                            torch.quantization.fuse_modules(
                                m.conv,
                                [str(idx), str(idx + 1)],
                                inplace=True,
                                fuser_func=quantization.fuse_known_modules,
                            )
                        else:
                            torch.quantization.fuse_modules(
                                m,
                                [
                                    "conv." + str(idx),
                                    "conv." + str(idx + 1),
                                    "skip_add",
                                ],
                                inplace=True,
                                fuser_func=quantization.fuse_known_modules,
                            )

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        getattr(
            self.classifier, "1"
        ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
def mobilenet_v2(pretrained_path=None, **kwargs):  # noqa: D205,D400
    """
    Construct a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
      <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        path (str): The path of pretrained model.
    """

    model = MobileNetV2(**kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        linear_weight = state_dict["classifier.1.weight"]
        state_dict["classifier.1.weight"] = linear_weight.view(
            linear_weight.shape + (1, 1)
        )
        model.load_state_dict(state_dict)
    return model
