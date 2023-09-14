# Copyright (c) Changan Auto. All rights reserved.

import torch
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["Scale", "DynamicWeight"]


for module in [nn.ReLU, nn.ReLU6, nn.SiLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh]:
    OBJECT_REGISTRY.register(module)
OBJECT_REGISTRY.register(nn.ReLU, name="relu")
OBJECT_REGISTRY.register(nn.ReLU6, name="relu6")
OBJECT_REGISTRY.register(nn.SiLU, name="swish")


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class DynamicWeight(Scale):
    """A learnable scale parameter for loss.

    This class refactor the forword function of `Scale`.
    """

    def forward(self, x):
        x = torch.exp(-self.scale) * x
        return x
