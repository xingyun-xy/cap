from typing import List

from capbc.message import Message

__all__ = [
    "LaserParam",
    "LidarParam",
]


class LaserParam(Message):
    def __init__(
        self,
        laser_id: int = None,
        azuimuth: float = None,
        vertical: float = None,
        distance: float = None,
        horizon_off: float = None,
        vertical_off: float = None,
        two_pt_correction_available: bool = None,
        distance_correction_x: float = None,
        distance_correction_y: float = None,
        focal_distance: float = None,
        focal_slope: float = None,
    ):
        super().__init__(topic="laser_param")
        self.laser_id = laser_id
        self.azuimuth = azuimuth
        self.vertical = vertical
        self.distance = distance
        self.horizon_off = horizon_off
        self.vertical_off = vertical_off
        self.two_pt_correction_available = two_pt_correction_available
        self.distance_correction_x = distance_correction_x
        self.distance_correction_y = distance_correction_y
        self.focal_distance = focal_distance
        self.focal_slope = focal_slope


class LidarParam(Message):
    def __init__(
        self,
        samplestamp: int = None,
        pitch: float = None,
        yaw: float = None,
        roll: float = None,
        x_offset: float = None,
        y_offset: float = None,
        z_offset: float = None,
        lidar_type: str = None,
        laser_param: List[LaserParam] = None,
    ):
        super().__init__(topic="lidar_param")
        self.samplestamp = samplestamp
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.lidar_type = lidar_type
        self.laser_param = laser_param
