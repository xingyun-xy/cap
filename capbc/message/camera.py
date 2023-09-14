from typing import List, Union
from enum import Enum

from capbc.message import Message

__all__ = [
    "VCSPram",
    "CameraValidHeight",
    "CameraMatrix",
    "CameraParam",
]


class VCSPram(Message):
    def __init__(self, rotation: float = None, translation: float = None):
        super().__init__(topic="vcs_param")
        self.rotation = rotation
        self.translation = translation


class CameraValidHeight(Message):
    def __init__(self, left_y: int = None, right_y: int = None):
        super().__init__(topic="camera_valid_height")
        self.left_y = left_y
        self.right_y = right_y


class CameraMatrix(Message):
    def __init__(
        self,
        gnd2img: List[List[float]] = None,  # 3x3 matrix
        img2gnd: List[List[float]] = None,  # 3x3 matrix
        vcsgnd2img: List[List[float]] = None,  # 3x3 matrix
        img2vcsgnd: List[List[float]] = None,  # 3x3 matrix
        local2img: List[List[float]] = None,  # 3x3 matrix
        img2local: List[List[float]] = None,  # 3x3 matrix
    ):
        super().__init__(topic="camera_matrix")
        self.gnd2img = gnd2img
        self.img2gnd = img2gnd
        self.vcsgnd2img = vcsgnd2img
        self.img2vcsgnd = img2vcsgnd
        self.local2img = local2img
        self.img2local = img2local


class CameraParam(Message):
    def __init__(
        self,
        focal_u: float = None,
        focal_v: float = None,
        center_u: float = None,
        center_v: float = None,
        camera_x: float = None,
        camera_y: float = None,
        camera_z: float = None,
        pitch: float = None,
        yaw: float = None,
        roll: float = None,
        camera_type: Union[int, Enum] = None,
        fov: float = None,
        vender: str = None,
        timestamp: int = None,
        calibration_status: Union[int, Enum] = None,
        vcs: VCSPram = None,
        valid_height: CameraValidHeight = None,
        distort: List[float] = None,
        mat: CameraMatrix = None,
    ):
        super().__init__(topic="camera_param")
        self.focal_u = focal_u
        self.focal_v = focal_v
        self.center_u = center_u
        self.center_v = center_v
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.camera_z = camera_z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.camera_type = camera_type
        self.fov = fov
        self.vender = vender
        self.timestamp = timestamp
        self.calibration_status = calibration_status
        self.vcs = vcs
        self.valid_height = valid_height
        self.distort = distort
        self.mat = mat
