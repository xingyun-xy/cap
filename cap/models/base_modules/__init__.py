# Copyright (c) Changan Auto. All rights reserved.

from . import postprocess, target
from .activation import DynamicWeight, Scale
from .anchor_generator import AnchorGenerator
from .basic_efficientnet_module import MBConvBlock, SEBlock
from .basic_mixvargenet_module import BasicMixVarGEBlock, MixVarGEBlock
from .basic_resnet_module import BasicResBlock, BottleNeck
from .basic_vargdarknet_module import VargDarkNetBlock
from .basic_vargnasnet_module import VargNASNetBlock
from .basic_vargnet_module import (
    BasicVarGBlock,
    BasicVarGBlockV2,
    ExtendVarGNetFeatures,
    OnePathResUnit,
    TwoPathResUnit,
)
from .bbox_decoder import XYWHBBoxDecoder
from .conv_compactor import ConvCompactor2d
from .conv_module import ConvModule2d, ConvTransposeModule2d, ConvUpsample2d
from .extend_container import ExtSequential, MultiInputSequential
from .gn_module import GroupNorm2d
from .inverted_residual import InvertedResidual
from .label_encoder import (
    MatchLabelFlankEncoder,
    MatchLabelGroundLineEncoder,
    MatchLabelSepEncoder,
    OneHotClassEncoder,
    RCNN3DLabelFromMatch,
    RCNNBinDetLabelFromMatch,
    RCNNKPSLabelFromMatch,
    RCNNMultiBinDetLabelFromMatch,
    XYWHBBoxEncoder,
)
from .loss_hard_neg_mining import LossHardNegativeMining
from .matcher import IgRegionMatcher, MaxIoUMatcher
from .resize_parser import ResizeParser
from .roi_feat_extractors import Cropper, CropperQAT, MultiScaleRoIAlign
from .separable_conv_module import (
    SeparableConvModule2d,
    SeparableGroupConvModule2d,
)

__all__ = [
    "AnchorGenerator",
    "postprocess",
    "target",
    "Scale",
    "Swish",
    "MBConvBlock",
    "SEBlock",
    "BasicResBlock",
    "BottleNeck",
    "VargDarkNetBlock",
    "VargNASNetBlock",
    "BasicVarGBlock",
    "BasicVarGBlockV2",
    "ExtendVarGNetFeatures",
    "OnePathResUnit",
    "TwoPathResUnit",
    "BasicMixVarGEBlock",
    "MixVarGEBlock",
    "XYWHBBoxDecoder",
    "MatchLabelGroundLineEncoder",
    "MatchLabelFlankEncoder",
    "ConvModule2d",
    "ConvTransposeModule2d",
    "ConvUpsample2d",
    "ExtSequential",
    "MultiInputSequential",
    "InvertedResidual",
    "MatchLabelSepEncoder",
    "OneHotClassEncoder",
    "XYWHBBoxEncoder",
    "RCNNKPSLabelFromMatch",
    "RCNN3DLabelFromMatch",
    "RCNNBinDetLabelFromMatch",
    "RCNNMultiBinDetLabelFromMatch",
    "LossHardNegativeMining",
    "IgRegionMatcher",
    "MaxIoUMatcher",
    "ResizeParser",
    "SeparableConvModule2d",
    "SeparableGroupConvModule2d",
    "GroupNorm2d",
    "CropperQAT",
    "Cropper",
    "MultiScaleRoIAlign",
    "ConvCompactor2d",
]
