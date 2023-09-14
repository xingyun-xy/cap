# Copyright (c) Changan Auto. All rights reserved.

from .auto_assign_loss import AutoAssignLoss, CenterLoss, NegLoss, PosLoss
from .bev_discrete_object_loss import BEVDiscreteObjectLoss
from .camera3d_losses import HMFocalLoss, HML1Loss
from .ciou_loss import CIoULoss
from .cross_entropy_loss import (
    CEWithLabelSmooth,
    CEWithWeightMap,
    CrossEntropyLoss,
    CrossEntropyLossV2,
    SoftTargetCrossEntropy,
)
from .dddv_losses import (
    BEV3DLoss,
    DepthConfidenceLoss,
    DepthLoss,
    DepthPoseResflowLoss,
    LossCalculationWrapper,
)
from .elevation_loss import ElevationLoss, GammaLoss, GroundLoss
from .fcos_loss import FCOSLoss
from .focal_loss import (
    FocalLoss,
    FocalLossV2,
    GaussianFocalLoss,
    SoftmaxFocalLoss,
)
from .giou_loss import GIoULoss
from .hinge_loss import (
    ElementwiseL1HingeLoss,
    ElementwiseL2HingeLoss,
    WeightedSquaredHingeLoss,
)
from .l1_loss import L1Loss
from .lnnorm_loss import LnNormLoss
from .lovasz_loss import LovaszSoftmaxLoss
from .mse_loss import MSELoss
from .real3d_losses import Real3DLoss
from .seg_loss import (
    MixSegLoss,
    MixSegLossMultipreds,
    MultiStrideLosses,
    SegLoss,
)
from .smooth_l1_loss import SmoothL1Loss
from .softmax_ce_loss import SoftmaxCELoss
from .yolo_losses import YOLOV3Loss
from .gaussian_focal_loss import GaussianFocalLoss_bev

# TODO(min.du, 0.5): format loss file names #

__all__ = [
    "CEWithLabelSmooth",
    "CrossEntropyLoss",
    "CrossEntropyLossV2",
    "SoftTargetCrossEntropy",
    "CEWithWeightMap",
    "LovaszSoftmaxLoss",
    "DepthConfidenceLoss",
    "DepthLoss",
    "DepthPoseResflowLoss",
    "ElementwiseL1HingeLoss",
    "ElementwiseL2HingeLoss",
    "LossCalculationWrapper",
    "FCOSLoss",
    "AutoAssignLoss",
    "PosLoss",
    "NegLoss",
    "CenterLoss",
    "FocalLoss",
    "FocalLossV2",
    "SoftmaxFocalLoss",
    "GaussianFocalLoss",
    "GIoULoss",
    "CIoULoss",
    "MSELoss",
    "Real3DLoss",
    "SegLoss",
    "SmoothL1Loss",
    "SoftmaxCELoss",
    "WeightedSquaredHingeLoss",
    "YOLOV3Loss",
    "ElevationLoss",
    "GammaLoss",
    "GroundLoss",
    "HMFocalLoss",
    "HML1Loss",
    "BEV3DLoss",
    "BEVDiscreteObjectLoss",
    "MixSegLossMultipreds",
    "MixSegLoss",
    "LnNormLoss",
    "L1Loss",
    "MultiStrideLosses",
    "GaussianFocalLoss_bev",
]
