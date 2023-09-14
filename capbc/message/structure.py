from typing import Any, Union, List
from enum import Enum

from capbc.message import Message, MessageMeta, DataStore

__all__ = [
    "Attribute",
    "Point2D",
    "BBox2D",
    "BBox3D",
    "Mask2D",
    "KeyPoint2D",
]


class Attribute(Message):
    """ Attribute to describe other messages type. """

    __slots__ = ["value", "score"]

    def __init__(
        self,
        value: Any,
        score: float = None,
        topic: str = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            value (Any): attribute value
            score (float, optional): confidence score. Defaults to None.
            topic (str, optional): message topic. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)
        self.value = value
        self.score = score


class Point2D(Message):
    """ 2D point."""
    __slots__ = ["data", "score", "occlusion"]

    def __init__(
        self,
        data,
        score: float = None,
        occlusion: Union[int, Enum] = None,
        topic: str = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            data: list / numpy / tensor data.
            score (float, optional): confidence score. Defaults to None.
            occlusion (Union[int, Enum], optional): occlusion level. Defaults \
                to None.
            topic (str, optional): message topic. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)

        if len(data) != 2:
            raise ValueError("data should contain 2 elements")

        self.data = DataStore(data)
        self.score = score
        self.occlusion = occlusion

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]


class BBox2D(Message):
    """ 2D bounding box. """
    __slots__ = ["data", "score", "occlusion"]

    def __init__(
        self,
        data,
        score: float = None,
        occlusion: Union[int, Enum] = None,
        truncation: Union[int, Enum] = None,
        topic: str = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            data: list / numpy / tensor data, arranged in order of 'x1, y1, \
                x2, y2'.
            score (float, optional): confidence score. Defaults to None.
            occlusion (Union[int, Enum], optional): occlusion level. Defaults \
                to None.
            truncation (Union[int, Enum], optional): truncation level. \
                Defaults to None.
            topic (str, optional): message topic. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)

        if len(data) != 4:
            raise ValueError("data should contain 4 elements")

        self.data = DataStore(data)
        self.score = score
        self.occlusion = occlusion
        self.truncation = truncation

    def __repr__(self) -> str:
        repr_str_list = [
            f"BBox2D: {self.topic}",
            f"  x1: {self.x1}",
            f"  y1: {self.y1}",
            f"  x2: {self.x2}",
            f"  y2: {self.y2}",
        ]
        if self.score is not None:
            repr_str_list.append(f"  score: {self.score}")
        if self.occlusion is not None:
            repr_str_list.append(f"  occlusion: {self.occlusion}")
        if self.truncation is not None:
            repr_str_list.append(f"  truncation: {self.truncation}")
        repr_str = "\n".join(repr_str_list)
        return repr_str

    @property
    def x1(self):
        return self.data[0]

    @property
    def y1(self):
        return self.data[1]

    @property
    def x2(self):
        return self.data[2]

    @property
    def y2(self):
        return self.data[3]


class BBox3D(Message):
    """ 3D bounding box. """
    __slots__ = ["dim", "loc", "yaw", "alpha", "score"]

    def __init__(
        self,
        dim,
        loc,
        yaw: float,
        alpha: float = None,
        score: float = None,
        topic: str = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            dim: list / numpy / tensor data of dimension, arranged in order \
                of 'width, height, length'.
            loc: list / numpy / tensor data of location,  arranged in order \
                of 'x, y, z'.
            yaw (float): yaw.
            alpha (float, optional): observation angle of object, ranging \
                [-pi, pi]. Defaults to None.
            score (float, optional): confidence score. Defaults to None.
            topic (str, optional): message topic. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)

        if len(dim) != 3:
            raise ValueError("dim should contain 3 elements")
        if len(loc) != 3:
            raise ValueError("loc should contain 3 elements")

        self.dim = DataStore(dim)
        self.loc = DataStore(loc)
        self.yaw = yaw
        self.alpha = alpha

        self.score = score

    def __repr__(self) -> str:
        repr_str_list = [
            f"BBox3D: {self.topic}",
            f"  width: {self.width}",
            f"  height: {self.height}",
            f"  length: {self.length}",
            f"  x: {self.x}",
            f"  y: {self.y}",
            f"  z: {self.z}",
            f"  yaw: {self.yaw}",
            f"  alpha: {self.alpha}",
            f"  score: {self.score}",
        ]
        if self.alpha is not None:
            repr_str_list.append(f"  alpha: {self.alpha}")
        if self.score is not None:
            repr_str_list.append(f"  score: {self.score}")
        repr_str = "\n".join(repr_str_list)
        return repr_str

    @property
    def width(self):
        return self.dim[0]

    @property
    def height(self):
        return self.dim[1]

    @property
    def length(self):
        return self.dim[2]

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    @property
    def z(self):
        return self.loc[2]


class Mask2D(Message):
    """ 2D mask. """
    __slots__ = ["data", "score", "occlusion"]

    def __init__(
        self,
        data,
        score: float = None,
        occlusion: Union[int, Enum] = None,
        topic: str = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            data: list / numpy / tensor data.
            score (float, optional): confidence score. Defaults to None.
            occlusion (Union[int, Enum], optional): occlusion level. Defaults \
                to None.
            topic (str, optional): message topic. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)
        self.data = DataStore(data)
        self.score = score
        self.occlusion = occlusion

    @property
    def mask(self):
        return self.data.data


class KeyPoint2D(Message):
    """ 2D key points. """
    __slots__ = ["data", "scores", "occlusion"]

    def __init__(
        self,
        data,
        scores: List[float] = None,
        occlusion: Union[int, Enum] = None,
        topic: str = None,
        meta: MessageMeta = None,
    ):
        """ Constructor.

        Args:
            data: list / numpy / tensor data.
            score (List[float], optional): confidence score. Defaults to None.
            occlusion (Union[int, Enum], optional): occlusion level. Defaults \
                to None.
            topic (str, optional): message topic. Defaults to None.
            meta (MessageMeta, optional): meta info. Defaults to None.
        """
        super().__init__(topic=topic, meta=meta)

        if len(data) % 2:
            raise ValueError("Can not form points correctly")

        self.data = DataStore(data)
        self.scores = scores
        self.occlusion = occlusion

    @property
    def points(self):
        return [
            Point2D(self.data[i:i+2], self.topic)
            for i in range(0, len(self.data), 2)
        ]
