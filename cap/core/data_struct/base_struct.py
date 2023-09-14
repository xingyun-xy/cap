from dataclasses import InitVar, dataclass, make_dataclass
from typing import ClassVar, Dict, List, Optional, Sequence, Union

import torch
from torch.nn.functional import interpolate

from .base import BaseData, BaseDataList

__all__ = [
    "ClsLabel",
    "ClsLabels",
    "DetBox2D",
    "DetBoxes2D",
    "Mask",
    "Masks",
    "MultipleBox2D",
    "MultipleBoxes2D",
    "DetBox3D",
    "DetBoxes3D",
    "Line2D",
    "Lines2D",
]


def is_tensor_scalar(x: torch.Tensor):
    return x.numel() == 1 and x.ndim == 0


def not_none(*args):
    for a in args:
        if a is None:
            return False
    return True


def is_tensor(x):
    return isinstance(x, torch.Tensor)


@dataclass
class ClsLabel(BaseData):
    """Class label data class for single sample.

    Args:
        cls_idx: Class category id.
        score: Prediction score.
        cls_name: Class name.
    """

    cls_idx: Union[torch.Tensor, int] = None
    score: Optional[Union[torch.Tensor, float]] = None
    cls_name: InitVar[str] = None

    def __post_init__(self, cls_name):
        assert self.cls_idx is not None
        self._cls_name = cls_name
        self._check_values()

    @property
    def cls_name(self):
        return self._cls_name

    def to(self, device):
        return self._to(device, cls_name=self.cls_name)

    def _check_values(self):
        if not_none(self.score):
            if is_tensor(self.score):
                assert is_tensor_scalar(self.score)
            else:
                assert isinstance(self.score, float)

        if not_none(self.cls_idx):
            if is_tensor(self.cls_idx):
                assert is_tensor_scalar(self.cls_idx)
                self.cls_idx = self.cls_idx.int()
            else:
                assert isinstance(self.cls_idx, int)

        if (not_none(self.score, self.cls_idx) and is_tensor(self.score)
                and is_tensor(self.cls_idx)):
            assert self.score.device == self.cls_idx.device

    def rescale(self, *args, **kwargs):
        return self

    def to_text(self, with_score=False):
        text = self._cls_name
        if with_score and self.score is not None:
            text += " {:.2f}".format(
                self.score.item() if is_tensor(self.score) else self.score)
        return text

    def to_sda_eval(self):
        """Get results in SDA Eval format."""
        return {
            "attrs": {
                "sub_type":
                self.cls_idx.item()
                if is_tensor(self.cls_idx) else self.cls_idx
            }
        }


@dataclass
class ClsLabels(BaseDataList):
    """Class label data class for multiple samples.

    Args:
        cls_idxs: Shape(N), lass category ids
        score: Shape(N), prediction scores.
        cls_name: Shape(N), class names.
    """

    singleton_cls: ClassVar = ClsLabel
    cls_idxs: torch.Tensor = None
    scores: Optional[torch.Tensor] = None
    cls_name_mapping: InitVar[Dict[int, str]] = None

    def __post_init__(self, cls_name_mapping):
        assert self.cls_idxs is not None
        if cls_name_mapping is None:
            cls_name_mapping = {}
        self._cls_name_mapping = cls_name_mapping
        self._check_values()

    @property
    def cls_names(self):
        return self._cls_name_mapping

    def _check_values(self):

        if self.cls_idxs is not None:
            self.cls_idxs = self.cls_idxs.int()

        if not_none(self.scores, self.cls_idxs):
            assert len(self.scores) == len(self.cls_idxs)
            assert self.scores.device == self.cls_idxs.device

    def __getitem__(self, idx: Union[int, slice, torch.Tensor]):
        if isinstance(idx, int):
            kwargs = ({
                "cls_name": self.cls_names[self.cls_idxs[idx].item()]
            } if self.cls_names else {})
        else:
            kwargs = {"cls_name_mapping": self.cls_names}

        return self._getitem(idx, **kwargs)

    def to(self, device):
        return self._to(device, cls_name_mapping=self.cls_names)

    def rescale(self, *args, **kwargs):
        return self

    def inv_pad(self, *args):
        return self

    def with_scores_gt(self, threshold: float):
        """Get results with score greater than threshold."""

        assert not_none(self.scores), "Do not have scores attribute"
        return self.filter_by_lambda(lambda x: x.scores > threshold)

    def with_scores_ge(self, threshold: float):
        """Get the result with score greater than or equal to threshold."""

        assert not_none(self.scores), "Do not have scores attribute"
        return self.filter_by_lambda(lambda x: x.scores >= threshold)

    def with_cls_idxs_in(self, indices: Union[int, Sequence[int]]):
        """Get all results with cls_idx in indices."""
        assert (self.cls_idxs
                is not None), "Boxes do not have cls_idxs attribute"

        if isinstance(indices, int):
            indices = [indices]

        assert indices, "indices should not be empty"

        def cls_in(x):
            mask = None
            for i in indices:
                cur_mask: torch.Tensor = x.cls_idxs == i
                if mask is None:
                    mask = cur_mask
                else:
                    mask.logical_or_(cur_mask)
            return mask

        return self.filter_by_lambda(cls_in)


@dataclass
class DetBox2D(ClsLabel):
    """Detection bbox2d data class for single sample."""

    box: torch.Tensor = None

    def __post_init__(self, *args):
        assert self.box is not None
        assert self.box.ndim == 1 and len(self.box) == 4
        device = self.box.device

        if not_none(self.score) and is_tensor(self.score):
            assert self.score.device == device

        if not_none(self.cls_idx) and is_tensor(self.cls_idx):
            assert self.cls_idx.device == device

        super().__post_init__(*args)

    def rescale(self, *scales):
        if len(scales) == 2:
            scale_w, scale_h = scales
        else:
            assert len(scales) == 1
            scale_w = scale_h = scales[0]
        self.box[::2] *= scale_w
        self.box[1::2] *= scale_h

    def to_text(self, with_score=True):
        return super().to_text(with_score=with_score)

    def to_sda_eval(self):
        """Get results in SDA Eval format."""
        return {
            "bbox": self.box.numpy().tolist(),
            "bbox_score":
            self.score.item() if is_tensor(self.score) else self.score,
            "attrs": {},
        }


@dataclass
class DetBoxes2D(ClsLabels):
    """Detection bbox2d data class for multiple samples.

    Args:
        boxes: Shape(N, 4), prediction box.
    """

    singleton_cls: ClassVar = DetBox2D
    boxes: torch.Tensor = None

    def __post_init__(self, *args):
        assert self.boxes is not None
        assert self.boxes.ndim == 2 and self.boxes.shape[1] == 4

        device = self.boxes.device
        length = len(self.boxes)

        if not_none(self.scores):
            assert len(self.scores) == length and self.scores.device == device

        if not_none(self.cls_idxs):
            assert (len(self.cls_idxs) == length
                    and self.cls_idxs.device == device)

        super().__post_init__(*args)

    def rescale(self, *scales):
        if len(scales) == 2:
            scale_w, scale_h = scales
        else:
            assert len(scales) == 1
            scale_w = scale_h = scales[0]

        self.boxes[:, ::2] *= scale_w
        self.boxes[:, 1::2] *= scale_h

    def inv_pad(self, *padding):
        assert len(padding) == 4
        self.boxes[:, ::2] -= padding[0]
        self.boxes[:, 1::2] -= padding[2]

    @classmethod
    def from_gt_data(cls, gt_data: torch.Tensor):
        assert gt_data.shape[1] == 5
        return cls(boxes=gt_data[:, :4], cls_idxs=gt_data[:, 4])


@dataclass
class MultipleBox2D(DetBoxes2D):

    def to_sda_eval(self):
        return [box.to_sda_eval() for box in iter(self)]


@dataclass
class MultipleBoxes2D(BaseDataList):
    singleton_cls: ClassVar = MultipleBox2D
    cls_idxs_list: List[torch.Tensor] = None
    scores_list: List[torch.Tensor] = None
    boxes_list: List[torch.Tensor] = None

    def __post_init__(self):
        not_none(
            self.boxes_list,
            self.scores_list,
            self.cls_idxs_list,
        )

        for boxes, scores, cls_idxs in zip(
                self.boxes_list,
                self.scores_list,
                self.cls_idxs_list,
        ):
            MultipleBox2D(boxes=boxes, scores=scores, cls_idxs=cls_idxs)

    def rescale(self, *scales):
        for boxes in iter(self):
            boxes.rescale(*scales)

    def inv_pad(self, *padding):
        for boxes in iter(self):
            boxes.inv_pad(*padding)


@dataclass
class DetBox3D(ClsLabel):
    """Detection bbox3d data class for single sample."""

    h: torch.Tensor = None
    w: torch.Tensor = None
    l: torch.Tensor = None
    x: torch.Tensor = None
    y: torch.Tensor = None
    z: torch.Tensor = None
    row: torch.Tensor = None
    yaw: torch.Tensor = None
    pitch: torch.Tensor = None
    bbox: torch.Tensor = None

    @property
    def location(self):
        return torch.tensor([self.x, self.y, self.z], device=self.device)

    @property
    def dimension(self):
        return torch.tensor([self.h, self.w, self.l], device=self.device)

    def __post_init__(self, *args):
        data = [
            self.h,
            self.w,
            self.l,
            self.x,
            self.y,
            self.z,
            self.yaw,
            self.bbox,
        ]
        assert not_none(*data)
        # assert all([is_tensor_scalar(x) for x in data])

        if self.row is None:
            self.row = 0
        if self.pitch is None:
            self.pitch = 0

        super().__post_init__(*args)

    def to_text(self):
        return " {:.2f}".format(self.score.item())

    def to_sda_eval(self):
        """Get results in SDA Eval format."""
        # return {
        #     "depth": self.z.item(),
        #     "dimensions": self.dimension.numpy().tolist(),
        #     "rotation_y": self.yaw.item(),
        #     "location": self.location.numpy().tolist(),
        #     "score": self.score.item(),l
        # }
        return {
            "dim": self.dimension.numpy().tolist(),
            "category_id": [self.cls_idx.item()],
            "score": [self.score.item()],
            "center": (self.x.item(), self.y.item()),
            # "bbox": (self.w.item(), self.h.item()),
            "bbox": self.bbox.numpy().tolist(),
            "dep": [self.z.item()],
            "alpha": [self.pitch.item()],
            "location": self.location.numpy().tolist(),
            "score": [self.score.item()],
            "rotation_y": [self.row.item()],
        }


@dataclass
class DetBoxes3D(ClsLabels):
    """Detection bbox3d data class for multiple samples."""

    singleton_cls: ClassVar = DetBox3D
    h: torch.Tensor = None
    w: torch.Tensor = None
    l: torch.Tensor = None
    x: torch.Tensor = None
    y: torch.Tensor = None
    z: torch.Tensor = None
    row: torch.Tensor = None
    yaw: torch.Tensor = None
    pitch: torch.Tensor = None
    bbox: torch.Tensor = None

    @property
    def locations(self):
        return torch.stack([self.x, self.y, self.z], dim=-1)

    @property
    def dimensions(self):
        return torch.stack([self.h, self.w, self.l], dim=-1)

    def __post_init__(self, *args):
        data = [
            self.h,
            self.w,
            self.l,
            self.x,
            self.y,
            self.z,
            self.yaw,
            self.bbox,
        ]
        assert not_none(*data)

        length = len(self.x)
        if self.row is None:
            self.row = self.x.new_zeros(length)
        if self.pitch is None:
            self.pitch = self.x.new_zeros(length)

        super().__post_init__(*args)


@dataclass
class Mask(BaseData):
    """Mask data class for single sample."""

    mask: torch.Tensor

    def __post_init__(self):
        assert self.mask.ndim == 2

    def rescale(self, scale_w, scale_h):
        self.mask = interpolate(
            self.mask[None, None].float(),
            scale_factor=(scale_h, scale_w),
            mode="nearest",
        ).long()[0, 0]

    def inv_pad(self, *padding):
        h, w = self.mask.shape
        self.mask = self.mask[padding[2]:h - padding[3],
                              padding[0]:w - padding[1]]


@dataclass
class Masks(BaseDataList):
    """Mask data class for multiple samples."""

    singleton_cls: ClassVar = Mask
    masks: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.masks, torch.IntTensor)
        assert self.masks.ndim == 3

    def rescale(self, scale_w, scale_h):
        self.masks = interpolate(
            self.masks[None].float(),
            scale_factor=(scale_h, scale_w),
            mode="nearest",
        ).long()[0]


PTS_DICT = {}


def generate_nd_points(ndim: int):
    global PTS_DICT

    if ndim not in PTS_DICT:

        @dataclass
        class PointND(ClsLabel):
            point: torch.Tensor = None

            def __post_init__(self, *args):
                assert self.point is not None
                assert self.point.ndim == 1 and len(self.point) == ndim
                device = self.point.device

                if self.score is not None and isinstance(
                        self.score, torch.Tensor):
                    assert self.score.device == device

                if self.cls_idx is not None and isinstance(
                        self.cls_idx, torch.Tensor):
                    assert self.cls_idx.device == device

                super().__post_init__(*args)

            def rescale(self, *scales):
                if len(scales) != ndim:
                    assert len(scales) == 1
                    scales = [scales] * ndim

                scales = self.point.new_tensor(scales)
                self.point *= scales

        @dataclass
        class PointsND(ClsLabels):
            singleton_cls: ClassVar = PointND
            points: torch.Tensor = None

            def __post_init__(self, *args):
                assert self.points is not None
                assert self.points.ndim == 2 and self.points.shape[1] == ndim

                device = self.points.device
                length = len(self.points)

                if self.scores is not None:
                    assert (len(self.scores) == length
                            and self.scores.device == device)

                if self.cls_idxs is not None:
                    assert (len(self.cls_idxs) == length
                            and self.cls_idxs.device == device)

                super().__post_init__(*args)

            def rescale(self, *scales):
                if len(scales) != ndim:
                    assert len(scales) == 1
                    scales = [scales] * ndim

                scales = self.points.new_tensor(scales)
                self.points *= scales[None]

            def inv_pad(self, *padding):
                if ndim == 2:
                    self.points[:, 0] -= padding[0]
                    self.points[:, 1] -= padding[2]
                else:
                    raise NotImplementedError

        classes = (PointND, PointsND)
        PTS_DICT[ndim] = classes

    return PTS_DICT[ndim]


Point2D, Points2D = generate_nd_points(2)
Point3D, Points3D = generate_nd_points(3)


def generate_m_nd_points(m: int, ndim: int):

    global PTS_DICT

    if (ndim, m) not in PTS_DICT:

        PointND, PointsND = generate_nd_points(ndim)

        @dataclass
        class MPointBase(BaseData):
            num_points: ClassVar = m

            def rescale(self, *scales):
                for i in range(self.num_points):
                    getattr(self, f"point{i}").rescale(*scales)

        @dataclass
        class MPointsBase(BaseDataList):
            num_points: ClassVar = m

            def rescale(self, *scales):
                for i in range(self.num_points):
                    getattr(self, f"points{i}").rescale(*scales)

            def inv_pad(self, *padding):
                for i in range(self.num_points):
                    getattr(self, f"points{i}").inv_pad(*padding)

        MPointND = make_dataclass(
            "MPointND",
            [(f"point{i}", PointND, None) for i in range(m)],
            bases=(MPointBase, ),
            namespace={
                "__post_init__":
                lambda x: all(
                    [getattr(x, f"point{i}") is not None for i in range(m)]),
            },
        )

        def check_lengths(inputs):
            lengths = [len(x) for x in inputs]
            assert min(lengths) == max(lengths)

        MPointsND = make_dataclass(
            "MPointsND",
            [("singleton_cls", ClassVar, MPointND)] +
            [(f"points{i}", PointsND, None) for i in range(m)],
            bases=(MPointsBase, ),
            namespace={
                "__post_init__":
                lambda x: check_lengths(
                    [getattr(x, f"points{i}") for i in range(m)]),
            },
        )

        classes = (MPointND, MPointsND)
        PTS_DICT[(m, ndim)] = classes

    return PTS_DICT[(m, ndim)]


Point2D_2, Points2D_2 = generate_m_nd_points(2, 2)
Point2D_2.to_sda_eval = lambda self: {
    "attrs": {
        "p_WheelKeyPoints_2": [
            self.point0.point.numpy().tolist(),
            self.point1.point.numpy().tolist(),
        ],
        "kps_scores": [
            self.point0.score.item(),
            self.point1.score.item(),
        ],
    }
}


@dataclass
class Line2D(Point2D_2):

    def to_sda_eval(self):
        res = super().to_sda_eval()
        res["attrs"]["vehicle_ground_line"] = res["attrs"].pop(
            "p_WheelKeyPoints_2")
        return res


@dataclass
class Lines2D(Points2D_2):
    singleton_cls: ClassVar = Line2D
