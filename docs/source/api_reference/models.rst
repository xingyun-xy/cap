cap.models
==========

Models widely used in upper module in CAP.

Models
------

.. py:currentmodule:: cap.models

.. autosummary::
    :nosignatures:

    transforms

backbones
^^^^^^^^^

.. py:currentmodule:: cap.models.backbones

.. autosummary::
    :nosignatures:

    EfficientNet

    efficientnet

    efficientnet_lite

    MMDetBackboneAdaptor

    MobileNetV1

    MobileNetV2

    ResNet18

    ResNet50

    VarGDarkNet53

    ResNet50V2

    ResNet18V2

    VargNASNet

    VargNetV2

    VargNetV2Stage2631

    TinyVargNetV2

    get_vargnetv2_stride2channels

    CocktailVargNetV2

    MixVarGENet

    get_mixvargenet_stride2channels

    ZeroPad2DPatcher

    IResNet100

    IResNet180

    ResNetBevDepth

    BaseLSSFPN

losses
^^^^^^

.. py:currentmodule:: cap.models.losses

.. autosummary::
    :nosignatures:

    CEWithLabelSmooth

    CrossEntropyLoss

    CrossEntropyLossV2

    SoftTargetCrossEntropy

    CEWithWeightMap

    LovaszSoftmaxLoss

    DepthConfidenceLoss

    DepthLoss

    DepthPoseResflowLoss

    ElementwiseL1HingeLoss

    ElementwiseL2HingeLoss

    LossCalculationWrapper

    FCOSLoss

    AutoAssignLoss

    PosLoss

    NegLoss

    CenterLoss

    FocalLoss

    FocalLossV2

    SoftmaxFocalLoss

    GaussianFocalLoss

    GIoULoss

    CIoULoss

    MSELoss

    Real3DLoss

    SegLoss

    SmoothL1Loss

    SoftmaxCELoss

    WeightedSquaredHingeLoss

    YOLOV3Loss

    ElevationLoss

    GammaLoss

    GroundLoss

    HMFocalLoss

    HML1Loss

    BEV3DLoss

    BEVDiscreteObjectLoss

    MixSegLossMultipreds

    MixSegLoss

    LnNormLoss

    L1Loss

    MultiStrideLosses

    GaussianFocalLoss_bev

necks
^^^^^

.. py:currentmodule:: cap.models.necks

.. autosummary::
    :nosignatures:

    BiFPN

    DwUnet

    FPN

    RetinaNetFPN

    SequentialBottleNeck

    Unet

    YOLOV3Neck

    PAFPN

    RPAFPN

    FixChannelNeck

    UFPN

    FastSCNNNeck

    SECONDFPN

structures
^^^^^^^^^^

.. py:currentmodule:: cap.models.structures

.. autosummary::
    :nosignatures:

    Classifier

    GraphModel

    MultitaskGraphModel

    Segmentor

    LipmoveModel

    ReIDModule

    Camera3D

    bev

    bev_matrixvt

    bev_matrixvt_depth_loss

    bev_matrixvt_train

    bev_matrixvt_val

    bev_matrixvt_ONNX

detectors
*********

.. py:currentmodule:: cap.models.structures.detectors

.. autosummary::
    :nosignatures:

    RetinaNet

    TwoStageDetector

    YOLOV3

    FCOS

opticalflow
***********

.. py:currentmodule:: cap.models.structures.opticalflow

.. autosummary::
    :nosignatures:

    PwcNet

task_modules
^^^^^^^^^^^^

.. py:currentmodule:: cap.models.task_modules

.. autosummary::
    :nosignatures:

    AnchorModule

    RoIModule

bev
***

.. py:currentmodule:: cap.models.task_modules.bev

.. autosummary::
    :nosignatures:

    BEVFusionModule

    RandomRotation

    BEVPostprocess

    BEVTarget

    SpatialTransfomer

    BEV3DHead

    BEV3Decoder

    BEVDiscreteObjectDecoder

    BEVDepthHead

    BEVDepthHead_loss

    BEVDepthHead_loss_v2

    CenterPointBBoxCoder

    BaseLSSFPN_matrixvt

    MatrixVT

dddv
****

.. py:currentmodule:: cap.models.task_modules.dddv

.. autosummary::
    :nosignatures:

    DepthPoseResflowHead

    PixelHead

    ResidualFlowPoseHead

    DepthTarget

    OutputBlock

depth
*****

.. py:currentmodule:: cap.models.task_modules.depth

.. autosummary::
    :nosignatures:

    MultiStrideDepthLoss

fcos
****

.. py:currentmodule:: cap.models.task_modules.fcos

.. autosummary::
    :nosignatures:

    FCOSDecoder

    FCOSMultiStrideFilter

    FCOSMultiStrideCatFilter

    FCOSHead

    FCOSTarget

    DynamicFcosTarget

    multiclass_nms

    get_points

    distance2bbox

real3d
******

.. py:currentmodule:: cap.models.task_modules.real3d

.. autosummary::
    :nosignatures:

    Real3DDecoder

    Real3DHead

    Camera3DHead

    Camera3DLoss

retinanet
*********

.. py:currentmodule:: cap.models.task_modules.retinanet

.. autosummary::
    :nosignatures:

    RetinaNetHead

    RetinaNetPostProcess

reid
****

.. py:currentmodule:: cap.models.task_modules.reid

.. autosummary::
    :nosignatures:

    ReIDClsOutputBlock

rpn
***

.. py:currentmodule:: cap.models.task_modules.rpn

.. autosummary::
    :nosignatures:

    RPNVarGNetHead

    RPNSepLoss

seg
***

.. py:currentmodule:: cap.models.task_modules.seg

.. autosummary::
    :nosignatures:

    SegDecoder

    VargNetSegDecoder

    SegHead

    SegTarget

    FRCNNSegHead

yolo
****

.. py:currentmodule:: cap.models.task_modules.yolo

.. autosummary::
    :nosignatures:

    YOLOV3AnchorGenerator

    YOLOV3Head

    YOLOV3LabelEncoder

    YOLOV3Matcher

    YOLOV3PostProcess

deeplab
*******

.. py:currentmodule:: cap.models.task_modules.deeplab

.. autosummary::
    :nosignatures:

    Deeplabv3plusHead

fcn
***

.. py:currentmodule:: cap.models.task_modules.fcn

.. autosummary::
    :nosignatures:

    FCNHead

    DepthwiseSeparableFCNHead

    FCNTarget

    FCNDecoder

elevation
*********

.. py:currentmodule:: cap.models.task_modules.elevation

.. autosummary::
    :nosignatures:

    ElevationHead

    GroundHead

    ElevationPostprocess

roi_modules
***********

.. py:currentmodule:: cap.models.task_modules.roi_modules

.. autosummary::
    :nosignatures:

    RCNNMixVarGEShareHead

    RCNNVarGNetShareHead

    RCNNVarGNetSplitHead

    RCNNVarGNetHead

    RCNNLoss

    RCNNKPSLoss

    RCNNCLSLoss

    RCNNDecoder

    RCNNKPSSplitHead

    RCNNHM3DMixVarGEHead

    RCNNHM3DVarGNetHead

    RCNNBinDetLoss

    RCNNMultiBinDetLoss

    RCNNSparse3DLoss

    HeatmapBox2dDecoder

    GroundLinePointDecoder

    ROI3DDecoder

    KpsDecoder

    RoIDecoder

    SoftmaxRoIClsDecoder

    FlankPointDecoder

    RoIRandomSampler

    RoIHardProposalSampler

pwcnet
******

.. py:currentmodule:: cap.models.task_modules.pwcnet

.. autosummary::
    :nosignatures:

    PwcNetHead

    PwcNetNeck

ipm_seg
*******

.. py:currentmodule:: cap.models.task_modules.ipm_seg

.. autosummary::
    :nosignatures:

    MaskcatFeatHead

    SemSegDecoder

    IPMHeadParser

    IPMSegTarget

API Reference
--------------

.. automodule:: cap.models
    :members:
    :exclude-members: base_modules

.. automodule:: cap.models.backbones
    :members:
    :exclude-members: contrib

.. automodule:: cap.models.losses
    :members:

.. automodule:: cap.models.necks
    :members:

.. automodule:: cap.models.structures
    :members:

.. automodule:: cap.models.structures.detectors
    :members:

.. automodule:: cap.models.structures.opticalflow
    :members:

.. automodule:: cap.models.task_modules
    :members:
    :exclude-members: OutputModule

.. automodule:: cap.models.task_modules.bev
    :members:

.. automodule:: cap.models.task_modules.dddv
    :members:

.. automodule:: cap.models.task_modules.depth
    :members:

.. automodule:: cap.models.task_modules.fcos
    :members:

.. automodule:: cap.models.task_modules.real3d
    :members:

.. automodule:: cap.models.task_modules.retinanet
    :members:

.. automodule:: cap.models.task_modules.reid
    :members:

.. automodule:: cap.models.task_modules.rpn
    :members:

.. automodule:: cap.models.task_modules.seg
    :members:

.. automodule:: cap.models.task_modules.yolo
    :members:

.. automodule:: cap.models.task_modules.deeplab
    :members:

.. automodule:: cap.models.task_modules.fcn
    :members:

.. automodule:: cap.models.task_modules.elevation
    :members:

.. automodule:: cap.models.task_modules.roi_modules
    :members:

.. automodule:: cap.models.task_modules.pwcnet
    :members:

.. automodule:: cap.models.task_modules.ipm_seg
    :members:

