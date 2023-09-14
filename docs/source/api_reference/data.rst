cap.data
========

Main data module for training in CAP, which contains datasets, transforms, samplers.

Data
----

collates
^^^^^^^^

.. py:currentmodule:: cap.data.collates

.. autosummary::
    :nosignatures:

    collate_2d

    collate_3d

    collate_psd

    CocktailCollate

    collate_lidar

    collate_fn_bevdepth

    collate_fn_bevdepth_cooperate_pilot

    collate_fn_changanbev

    collate_fn_bevdepth_onnx

dataloaders
^^^^^^^^^^^

.. py:currentmodule:: cap.data.dataloaders

.. autosummary::
    :nosignatures:

    MultitaskLoader

    PassThroughDataLoader

datasets
^^^^^^^^

.. py:currentmodule:: cap.data.datasets

.. autosummary::
    :nosignatures:

    Auto3DV

    Cityscapes

    CityscapesPacker

    RepeatDataset

    ComposeDataset

    ResampleDataset

    ConcatDataset

    DenseboxDataset

    DetSeg2DAnnoDataset

    DetSeg2DAnnoDatasetToDetFormat

    DetSeg2DAnnoDatasetToROIFormat

    FrameDataset

    RoidbDetectionDataset

    Auto2dFromImage

    ImageNet

    ImageNetPacker

    ImageNetFromImage

    Kitti2DDetection

    Kitti2DDetectionPacker

    Kitti2D

    AutoDetPacker

    AutoSegPacker

    Auto2dFromLMDB

    Coco

    CocoDetection

    CocoDetectionPacker

    CocoFromImage

    RandDataset

    Real3DDataset

    Auto3dFromImage

    Real3DDatasetRec

    PascalVOC

    VOCDetectionPacker

    VOCFromImage

    Elevation

    ElevationRec

    WaymoDataset

    Bev3DDatasetRec

    ElevationFromImage

    HeartRateDataset

    PSDSlotDataset

    PSDTestSlotDataset

    HomoGenerator

    FlyingChairsFromImage

    FlyingChairs

    FlyingChairsPacker

    YUVFrames

    CocktailDatasetV0

    CocktailCmdDataset

    TendisJ2MMVideoImageReader

    HDF5WaveformReader

    MMASRInfoList

    MMCMDInfoList

    TendisBasicInfoReader

    Face3dDataset

    DeepInsightRecordDataset

    WaicBoxyDataset

    WaicLaneDataset

    PilotTestDataset

    NuscDetDataset

samplers
^^^^^^^^

.. py:currentmodule:: cap.data.samplers

.. autosummary::
    :nosignatures:

    DistributedCycleMultiDatasetSampler

    DistSamplerHook

    SelectedSampler

    DistributedGroupSampler

transforms
^^^^^^^^^^

.. py:currentmodule:: cap.data.transforms

.. autosummary::
    :nosignatures:

    Resize3DV

    Crop3DV

    ToTensor3DV

    Normalize3DV

    SelectDataByIdx

    StackData

    PrepareDataBEV

    ConvertLayout

    BgrToYuv444

    OneHot

    LabelSmooth

    TimmTransforms

    TimmMixup

    Resize

    RandomFlip

    Pad

    Normalize

    RandomCrop

    ToTensor

    Batchify

    FixedCrop

    ColorJitter

    DetInputPadding

    AugmentHSV

    RandomExpand

    MinIoURandomCrop

    ImageTransform

    ImageToTensor

    ImageBgrToYuv444

    ImageConvertLayout

    ImageNormalize

    Real3dTargetGenerator

    RepeatImage

    ROIHeatmap3DDetectionLableGenerate

    RoI3DDetInputPadding

    Heatmap3DDetectionLableGenerate

    SegRandomCrop

    SegReWeightByArea

    LabelRemap

    SegOneHot

    SegResize

    SegRandomAffine

    Scale

    FlowRandomAffineScale

    ListToDict

    DeleteKeys

    RenameKeys

    Undistortion

    PILToTensor

    TensorToNumpy

    ResizeElevation

    CropElevation

    ToTensorElevation

    NormalizeElevation

    PrepareDataElevation

    YUVTurboJPEGDecoder

    BPUPyramidResizer

    ImgBufToYUV444

    KPSIterableDetRoITransform

    KPSDetInputPadding

    VehicleFlankRoiTransform

    FlankDetInputPadding

    SemanticSegAffineAugTransformerEx

    ROIDetectionIterableDetRoITransform

    RoIDetInputPadding

    ReIDTransform

packer
^^^^^^

.. py:currentmodule:: cap.data.packer

.. autosummary::
    :nosignatures:

    DenseBoxDetAnnoTs

    DetSeg2DPacker

    VizDenseBoxDetAnno

    VizRoiDenseBoxDetAnno

    RecToLmdbPacker

API Reference
--------------

.. automodule:: cap.data.collates
    :members:

.. automodule:: cap.data.dataloaders
    :members:

.. automodule:: cap.data.datasets
    :members:

.. automodule:: cap.data.samplers
    :members:

.. automodule:: cap.data.transforms
    :members:

.. automodule:: cap.data.packer
    :members:

