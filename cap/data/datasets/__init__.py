# Copyright (c) Changan Auto. All rights reserved.

from cap.data.datasets.carp.audio import HDF5WaveformReader
from cap.data.datasets.carp.basic_data_info import TendisBasicInfoReader
from cap.data.datasets.carp.image import TendisJ2MMVideoImageReader
from cap.data.datasets.carp.info import MMASRInfoList, MMCMDInfoList
from . import pack_type
from .auto_3dv import Auto3DV, Bev3DDatasetRec
from .bev import HomoGenerator
from .carp_dataset import CocktailCmdDataset, CocktailDatasetV0
from .cityscapes import Cityscapes, CityscapesPacker
from .dataset_wrappers import (
    ComposeDataset,
    ConcatDataset,
    RepeatDataset,
    ResampleDataset,
)
from .densebox_dataset import DenseboxDataset
from .det_seg_2d_anno_dataset import (
    DetSeg2DAnnoDataset,
    DetSeg2DAnnoDatasetToDetFormat,
    DetSeg2DAnnoDatasetToROIFormat,
)
from .elevation_dataset import Elevation, ElevationFromImage, ElevationRec
from .face3d_dataset import Face3dDataset
from .faceid_dataset import DeepInsightRecordDataset
from .flyingchairs_dataset import (
    FlyingChairs,
    FlyingChairsFromImage,
    FlyingChairsPacker,
)
from .frame import FrameDataset
from .heartrate_dataset import HeartRateDataset
from .image_auto2d import Auto2dFromImage
from .imagenet import ImageNet, ImageNetFromImage, ImageNetPacker
from .kitti2d import Kitti2D, Kitti2DDetection, Kitti2DDetectionPacker
from .lmdb_auto2d import Auto2dFromLMDB, AutoDetPacker, AutoSegPacker
from .mscoco import Coco, CocoDetection, CocoDetectionPacker, CocoFromImage
from .psd_dataset import PSDSlotDataset, PSDTestSlotDataset
from .rand_dataset import RandDataset
from .real3d_dataset import Auto3dFromImage, Real3DDataset, Real3DDatasetRec
from .roidb_detection_dataset import RoidbDetectionDataset
from .voc import PascalVOC, VOCDetectionPacker, VOCFromImage
from .waymo import WaymoDataset
from .yuv_data import YUVFrames
from .waic_dataset import WaicBoxyDataset, WaicLaneDataset
from .pilot_dataset import PilotTestDataset
from .bevdepth import CaBev3dDataset
from .testvis import testvisdataset
from .changanbevdataset import changanbevdataset
from . import carp

__all__ = [
    "Auto3DV",
    "carp",
    "Cityscapes",
    "CityscapesPacker",
    "RepeatDataset",
    "ComposeDataset",
    "ResampleDataset",
    "ConcatDataset",
    "DenseboxDataset",
    "DetSeg2DAnnoDataset",
    "DetSeg2DAnnoDatasetToDetFormat",
    "DetSeg2DAnnoDatasetToROIFormat",
    "FrameDataset",
    "RoidbDetectionDataset",
    "Auto2dFromImage",
    "ImageNet",
    "ImageNetPacker",
    "ImageNetFromImage",
    "Kitti2DDetection",
    "Kitti2DDetectionPacker",
    "Kitti2D",
    "AutoDetPacker",
    "AutoSegPacker",
    "Auto2dFromLMDB",
    "Coco",
    "CocoDetection",
    "CocoDetectionPacker",
    "CocoFromImage",
    "RandDataset",
    "Real3DDataset",
    "Auto3dFromImage",
    "Real3DDatasetRec",
    "PascalVOC",
    "VOCDetectionPacker",
    "VOCFromImage",
    "Elevation",
    "ElevationRec",
    "WaymoDataset",
    "Bev3DDatasetRec",
    "ElevationFromImage",
    "HeartRateDataset",
    "PSDSlotDataset",
    "PSDTestSlotDataset",
    "HomoGenerator",
    "FlyingChairsFromImage",
    "FlyingChairs",
    "FlyingChairsPacker",
    "YUVFrames",
    # 鸡尾酒(多模)算法
    "CocktailDatasetV0",
    "CocktailCmdDataset",
    "TendisJ2MMVideoImageReader",
    "HDF5WaveformReader",
    "MMASRInfoList",
    "MMCMDInfoList",
    "TendisBasicInfoReader",
    # halo
    "Face3dDataset",
    "DeepInsightRecordDataset",
    # waic
    "WaicBoxyDataset",
    "WaicLaneDataset",
    # Pilot
    "PilotTestDataset",
    "NuscDetDataset",
    "CaBev3dDataset",
    #TestVisDataset
    'testvisdataset',
    'changanbevdataset'
]
