from typing import List, Tuple

import torch
from changan_plugin_pytorch.dtype import qint8
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
from torch.nn.modules.utils import _pair

from .functional import multi_scale_roi_align, requantize


class MultiScaleRoIAlign(nn.Module):
    """
    Args:
        same as float version.
    """

    _QAT_MODULE = qat.MultiScaleRoIAlign

    def __init__(
        self,
        output_size: Tuple[int, int],
        feature_strides: List[int],
        sampling_ratio: int = 1,
        interpolate_mode: str = "bilinear",
        canonical_box_size: int = 224,
        canonical_level: int = 4,
        aligned=False,
        out_dtype=qint8,
    ):
        super(MultiScaleRoIAlign, self).__init__()
        self.output_size = output_size
        self.feature_strides = feature_strides
        self.sampling_ratio = sampling_ratio
        self.mode = interpolate_mode
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level
        self.out_dtype = out_dtype
        self.aligned = aligned
        self._check_init()
        self.register_buffer("out_scale", torch.ones([1], dtype=torch.float32))

    def _check_init(self):
        # check output_size
        if isinstance(self.output_size, int):
            self.output_size = _pair(self.output_size)
        assert (
            len(self.output_size) == 2
        ), "output size must have exactly two elements h*w."
        assert isinstance(self.output_size[0], int) and isinstance(
            self.output_size[1], int
        ), "output size must be 'int'."
        out_height, out_width = _pair(self.output_size)
        assert out_height > 0 and out_width > 0, "output size must be positive"

        # check feature_strides. Must be a power of 2.
        levels = torch.log2(torch.tensor(self.feature_strides))
        assert torch.isclose(
            levels, levels.round()
        ).all(), "feature strides must be a power of 2"

        assert self.sampling_ratio == 1, "only support sampling_ratio = 1"
        assert self.mode == "bilinear", "only support 'bilinear' mode now"

        if (
            get_march() in (March.BERNOULLI2, March.BERNOULLI)
        ) and self.aligned is not None:
            assert False, (
                "Not support 'aligned' parameter on Bernoulli or Bernoulli2! "
                + "Bernoulli and Bernoulli2 set roi_w = roi * spatical_scale "
                + "+ 1 and use origin interpolate mode."
            )

    def forward(self, x, box_lists):
        """
        Args:
            x(List[QTensor]): a list of feature maps of NCHW shape.
                Note they may have different scales.
            box_lists(List[QTensor/Tensor[L, 4]]): a list of N boxes.
        """
        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "arguments to forward must be lists"
        assert len(x) == len(self.feature_strides), (
            "the number of feature maps must equal to"
            + "the number of feature strides"
        )
        assert len(box_lists) == x[0].size(
            0
        ), "the length of box_lists must equal to the batch size N"

        if get_march() == March.BERNOULLI:
            if len(x) > 1:
                for per_feature in x:
                    assert (
                        per_feature.q_scale().item() == x[0].q_scale().item()
                    ), "BERNOULLI only supports features with same scale input"
            features = [per_feature.int_repr() for per_feature in x]
        else:
            features = [
                requantize(
                    per_feature.int_repr(),
                    per_feature.q_scale(),
                    torch.zeros(1).to(torch.long),
                    per_feature.dtype,
                    self.out_scale,
                    torch.zeros(1).to(torch.long),
                    self.out_dtype,
                )
                for per_feature in x
            ]

        if len(box_lists) and isinstance(box_lists[0], QTensor):
            for box in box_lists:
                assert box.q_scale().item() == 0.25, (
                    "invalid input box scale, "
                    + "we expect 0.25, but receive {}".format(
                        box.q_scale().item()
                    )
                )
            box_lists = [box.int_repr() for box in box_lists]

        result = multi_scale_roi_align(
            features,
            box_lists,
            self.output_size,
            [1 / stride for stride in self.feature_strides],
            self.sampling_ratio,
            self.aligned,
            self.mode,
            self.canonical_box_size,
            self.canonical_level,
        )

        return QTensor(result, self.out_scale, self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module."""

        assert type(mod) == cls._QAT_MODULE, (
            "qat." + cls.__name__,
            +".from_float only works for " + cls._QAT_MODULE.__name__,
        )

        quantized_mod = cls(
            output_size=mod.output_size,
            feature_strides=mod.feature_strides,
            sampling_ratio=mod.sampling_ratio,
            interpolate_mode=mod.mode,
            canonical_box_size=mod.canonical_box_size,
            canonical_level=mod.canonical_level,
            aligned=mod.aligned,
            out_dtype=mod.activation_post_process.dtype,
        )
        quantized_mod.out_scale.copy_(mod.activation_post_process.scale)

        return quantized_mod
