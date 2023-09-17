# Copyright (c) Changan Auto. All rights reserved.

from . import detectors, opticalflow
from .camera_3d import Camera3D
from .classifier import Classifier
from .encoder_decoder import EncoderDecoder
from .graph_model import GraphModel
from .lipmove_model import LipmoveModel
from .multitask_graph_model import MultitaskGraphModel
from .reid import ReIDModule
from .segmentor import Segmentor
from .bev import bev
from .bev_matrixvt import bev_matrixvt, bev_matrixvt_depth_loss
#from cap.models.task_modules.bev.bev_depth_head import BEVDepthHead, BEVDepthHead_loss, BEVDepthHead_loss_v2

__all__ = [
    "detectors", "opticalflow", "Classifier", "GraphModel",
    "MultitaskGraphModel", "Segmentor", "Face3dModel", "LipmoveModel",
    "ReIDModule", "Camera3D", "bev", "bev_matrixvt", "bev_matrixvt_depth_loss",
    "bev_matrixvt"
]
