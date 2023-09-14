from typing import List, Union, Dict
from enum import Enum

import numpy as np

from capbc.message import (
    Message,
    MessageMeta,
    filter_topics,
    CameraParam,
    LidarParam,
    Attribute,
    Instance,
    LaneLine,
)

__all__ = [
    "Frame",
    "CameraFrame",
    "LidarFrame",
]


class Frame(Message):
    """ Base class of frame. """

    def __init__(
        self,
        topic: str = None,
        perceptions: List[Message] = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            topic (str, optional): message topic. Defaults to None.
            perceptions (List[Message], optional): perception results, like \
                Instance, LaneLine. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)
        self.perceptions = perceptions

    def get_perceptions(
        self, topics: Union[str, List[str]] = None
    ) -> List[Message]:
        """ Get the peception results by topic filter. """
        return filter_topics(self.perceptions, topics)

    def get_instances(
        self, topics: Union[str, List[str]] = None
    ) -> List[Instance]:
        """ Get the instances by topic filter. """
        perceptions = self.get_perceptions(topics)
        return list(filter(lambda x: isinstance(x, Instance), perceptions))

    def get_lanelines(
        self, topics: Union[str, List[str]] = None
    ) -> List[LaneLine]:
        """ Get the lane lines by topic filter. """
        perceptions = self.get_perceptions(topics)
        return list(filter(lambda x: isinstance(x, LaneLine), perceptions))


class CameraFrame(Frame):
    """ A camera frame contains camara parameters. """

    def __init__(
        self,
        topic: str = None,
        image: np.ndarray = None,
        camera_param: CameraParam = None,
        camera_type: Union[str, Enum] = None,
        attributes: List[Attribute] = None,
        perceptions: List[Message] = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            topic (str, optional): message topic. Defaults to None.
            image (np.ndarray, optional): image content. Defaults to None.
            camera_param (Dict, optional): camera parameters. Defaults to None.
            camera_type (Union[str, Enum], optional): type of the camera. \
                Defaults to None.
            attributes (List[Attribute], optional): attributes. Defaults to \
                None.
            perceptions (List[Message], optional): perception results, like \
                Instance, LaneLine. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta, perceptions=perceptions)
        self.image = image
        self.camera_param = camera_param
        self.camera_type = camera_type
        self.attributes = attributes

    def get_attributes(
        self, topics: Union[str, List[str]] = None
    ) -> List[Attribute]:
        """ Get the attributes by topic filter. """
        return filter_topics(self.attributes, topics)


class LidarFrame(Frame):
    """A camera frame contains camara parameters."""

    def __init__(
        self,
        topic: str = None,
        pcl: np.ndarray = None,
        lidar_param: LidarParam = None,
        attributes: List[Attribute] = None,
        perceptions: List[Message] = None,
        meta: MessageMeta = None,
    ):
        """Constructor.

        Args:
            topic (str, optional): message topic. Defaults to None.
            pcl (np.ndarray, optional): pcl content. Defaults to None.
            lidar_param (Dict, optional): lidar parameters. Defaults to None.
            attributes (List[Attribute], optional): attributes. Defaults to \
                None.
            perceptions (List[Message], optional): perception results, like \
                Instance, LaneLine. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta, perceptions=perceptions)
        self.pcl = pcl
        self.lidar_param = lidar_param
        self.attributes = attributes

    def get_attributes(
        self, topics: Union[str, List[str]] = None
    ) -> List[Attribute]:
        """Get the attributes by topic filter."""
        return filter_topics(self.attributes, topics)
