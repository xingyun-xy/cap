# Copyright (c) Changan Auto. All rights reserved.

from . import contrib
from .efficientnet import EfficientNet, efficientnet, efficientnet_lite
from .iresnet import IResNet100, IResNet180
from .mixvargenet import MixVarGENet, get_mixvargenet_stride2channels
from .mmdetbackbone_adaptor import MMDetBackboneAdaptor
from .mobilenetv1 import MobileNetV1
from .mobilenetv2 import MobileNetV2
from .patcher import ZeroPad2DPatcher
from .resnet import ResNet18, ResNet50, ResNet50V2, ResNet18V2
from .vargdarknet import VarGDarkNet53
from .vargnasnet import VargNASNet
from .vargnetv2 import (
    CocktailVargNetV2,
    TinyVargNetV2,
    VargNetV2,
    get_vargnetv2_stride2channels,
)
from .vargnetv2_2631 import VargNetV2Stage2631
from .resnet_bevdepth import ResNetBevDepth
from .base_lss_fpn import BaseLSSFPN

__all__ = [
    "contrib",
    "EfficientNet",
    "efficientnet",
    "efficientnet_lite",
    "MMDetBackboneAdaptor",
    "MobileNetV1",
    "MobileNetV2",
    "ResNet18",
    "ResNet50",
    "VarGDarkNet53",
    "ResNet50V2",
    "ResNet18V2"
    "VargNASNet",
    "VargNetV2",
    "VargNetV2Stage2631",
    "TinyVargNetV2",
    "get_vargnetv2_stride2channels",
    "CocktailVargNetV2",
    "MixVarGENet",
    "get_mixvargenet_stride2channels",
    "ZeroPad2DPatcher",
    "IResNet100",
    "IResNet180",
    "ResNetBevDepth",
    "BaseLSSFPN",
]
