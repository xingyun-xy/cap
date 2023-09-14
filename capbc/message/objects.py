from typing import List, Union

from capbc.message import (
    Message,
    MessageMeta,
    filter_topics,
    Attribute,
    BBox2D,
    BBox3D,
    Mask2D,
    KeyPoint2D,
)

__all__ = [
    "Instance",
    "LaneLine",
]


class Instance(Message):
    """ Instance object. """
    __slots__ = ["attributes", "bbox2ds", "mask2ds", "keypoint2d"]

    def __init__(
        self,
        topic: str,
        track_id: int = None,
        attributes: List[Attribute] = None,
        bbox2ds: List[BBox2D] = None,
        bbox3ds: List[BBox3D] = None,
        mask2ds: List[Mask2D] = None,
        keypoint2ds: List[KeyPoint2D] = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        The strucure contained in this object represents different parts of \
            instance.

        For example:
            >>> instance = Instance(
            ...     topic="vehicle",
            ...     bbox2ds=[
            ...         BBox2D(topic="vehicle"),
            ...         BBox2D(topic="vehicle_rear"),
            ...     ]
            ... )

        Args:
            topic (str): message topic to decribe the detected instance.
            track_id (int): track_id of instance. Defaults to None.
            attributes (List[Attribute], optional): attributes. Defaults to \
                None.
            bbox2ds (List[BBox2D], optional): 2d bounding boxes. Defaults to \
                None.
            bbox3ds (List[BBox3D], optional): 3d bounding boxes. Defaults to \
                None.
            mask2ds (List[Mask2D], optional): 2d masks. Defaults to None.
            keypoint2ds (List[KeyPoint2D], optional): 2d key points. Defaults \
                to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)
        self.track_id = track_id
        self.attributes = attributes
        self.bbox2ds = bbox2ds
        self.bbox3ds = bbox3ds
        self.mask2ds = mask2ds
        self.keypoint2d = keypoint2ds

    def get_attributes(
        self, topics: Union[str, List[str]] = None
    ) -> List[Attribute]:
        """ Get attributes by topic filter.

        Args:
            topics (Union[str, List[str]], optional): message topics. If \
                None, return all attributes. Defaults to None.

        Returns:
            List[Attribute]: list of attributes.
        """
        return filter_topics(self.attributes, topics)

    def get_bbox2ds(
        self, topics: Union[str, List[str]] = None
    ) -> List[BBox2D]:
        """ Get 2d bounding boxes by topic filter. """
        return filter_topics(self.bbox2ds, topics)

    def get_bbox3ds(
        self, topics: Union[str, List[str]] = None
    ) -> List[BBox3D]:
        """ Get 3d bounding boxes by topic filter. """
        return filter_topics(self.bbox3ds, topics)

    def get_mask2ds(
        self, topics: Union[str, List[str]] = None
    ) -> List[Mask2D]:
        """ Get 2d masks by topic filter. """
        return filter_topics(self.mask2ds, topics)

    def get_keypoint2ds(
        self, topics: Union[str, List[str]] = None
    ) -> List[KeyPoint2D]:
        """ Get keypoints by topic filter. """
        return filter_topics(self.keypoint2d, topics)


class LaneLine(Message):
    """ Lane line. """
    __slots__ = ["attributes", "mask2ds"]

    def __init__(
        self,
        topic: str,
        track_id: int = None,
        attributes: List[Attribute] = None,
        mask2ds: List[Mask2D] = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        see :class:`Instance` for more information.

        Args:
            topic (str): message topic to decribe the lane line.
            track_id (int): track_id of lane line. Defaults to None.
            attributes (List[Attribute], optional): attributes. Defaults to \
                None.
            mask2ds (List[Mask2D], optional): 2d masks. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)
        self.track_id = track_id
        self.attributes = attributes
        self.mask2ds = mask2ds

    def get_attributes(
        self, topics: Union[str, List[str]] = None
    ) -> List[Attribute]:
        """ Get attributes by topic filter.

        Args:
            topics (Union[str, List[str]], optional): message topics. If \
                None, return all attributes. Defaults to None.

        Returns:
            List[Attribute]: list of attributes.
        """
        return filter_topics(self.attributes, topics)

    def get_mask2ds(
        self, topics: Union[str, List[str]] = None
    ) -> List[Mask2D]:
        """ Get 2d masks by topic filter. """
        return filter_topics(self.mask2ds, topics)
