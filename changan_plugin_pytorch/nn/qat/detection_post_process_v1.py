from typing import Optional

import torch
from changan_plugin_pytorch.dtype import qinfo, qint8, qint16
from changan_plugin_pytorch.march import get_march
from changan_plugin_pytorch.nn.detection_post_process_v1 import (
    DetectionPostProcessV1 as FloatDetectionPostProcessV1,
)
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import Tensor
from torch.jit.annotations import List, Tuple


class DetectionPostProcessV1(FloatDetectionPostProcessV1):
    _FLOAT_MODULE = FloatDetectionPostProcessV1

    def __init__(
        self,
        num_classes: int,
        box_filter_threshold: float,
        class_offsets: List[int],
        use_clippings: List[bool],
        image_size: Tuple[int, int],
        nms_threshold: float,
        pre_nms_top_k: int,
        post_nms_top_k: int,
        qconfig,
        nms_padding_mode: Optional[str] = None,
        nms_margin: float = 0.0,
        bbox_min_hw: Tuple[float, float] = (0, 0),
    ):
        super(DetectionPostProcessV1, self).__init__(
            num_classes,
            box_filter_threshold,
            class_offsets,
            use_clippings,
            image_size,
            nms_threshold,
            pre_nms_top_k,
            post_nms_top_k,
            nms_padding_mode=nms_padding_mode,
            nms_margin=nms_margin,
            bbox_min_hw=bbox_min_hw,
        )
        self.kInputShift = 4
        self.kOutputBBoxShift = 2

        self.register_buffer(
            "in_scale",
            torch.ones(1, dtype=torch.float32) / (1 << self.kInputShift),
        )
        self.register_buffer(
            "out_scale",
            torch.ones(1, dtype=torch.float32) / (1 << self.kOutputBBoxShift),
        )

        self.input_shifts = [self.kInputShift] * len(class_offsets)

        self.qconfig = qconfig

    def forward(
        self,
        data: List[QTensor],
        anchors: List[Tensor],
        image_sizes=None,
    ):
        assert_qtensor_input(data)

        shifted_data = []
        shifted_anchor = []
        for per_branch_data in data:
            if torch.equal(
                per_branch_data.q_scale(),
                self.in_scale.expand_as(per_branch_data.q_scale()),
            ):
                shifted_data.append(per_branch_data.as_subclass(torch.Tensor))
            else:
                raise ValueError(
                    "DetectionPostProcessV1 requires"
                    + " all inputs to be scale = 1 / 2**4, "
                    + "but receive scale = %f"
                    % per_branch_data.q_scale()[0].item()
                )
        march = get_march()

        approximate_mode = "floor" if march == "bernoulli" else "bpu_round"
        for per_branch_anchor in anchors:
            shifted_anchor.append(
                torch.ops.changan.scale_quanti(
                    per_branch_anchor.to(dtype=torch.float),
                    self.out_scale,
                    torch.zeros_like(self.out_scale).to(dtype=torch.long),
                    -1,
                    -1 << 31,
                    (1 << 31) - 1,
                    True,
                    False,
                    approximate_mode,
                    march,
                )
            )

        ret = super(DetectionPostProcessV1, self).forward(
            shifted_data, shifted_anchor, image_sizes
        )

        return [
            (
                QTensor(
                    torch.ops.changan.scale_quanti(
                        r[0].to(dtype=torch.float),
                        self.out_scale,
                        torch.zeros_like(self.out_scale).to(dtype=torch.long),
                        -1,
                        qinfo(qint16).min,
                        qinfo(qint16).max,
                        True,
                        False,
                        approximate_mode,
                        march,
                    ),
                    self.out_scale,
                    qint16,
                ),
                QTensor(
                    torch.ops.changan.scale_quanti(
                        r[1].to(dtype=torch.float),
                        self.in_scale,
                        torch.zeros_like(self.in_scale).to(dtype=torch.long),
                        -1,
                        qinfo(qint8).min,
                        qinfo(qint8).max,
                        True,
                        False,
                        approximate_mode,
                        march,
                    ),
                    self.in_scale,
                    qint8,
                ),
                r[2],
            )
            for r in ret
        ]

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qat_mod = cls(
            num_classes=mod.num_classes,
            box_filter_threshold=mod.box_filter_threshold,
            class_offsets=mod.class_offsets,
            use_clippings=mod.use_clippings,
            image_size=mod.image_size,
            nms_threshold=mod.nms_threshold,
            pre_nms_top_k=mod.pre_nms_top_k,
            post_nms_top_k=mod.post_nms_top_k,
            nms_padding_mode=mod.nms_padding_mode,
            nms_margin=mod.nms_margin,
            bbox_min_hw=mod.bbox_min_hw,
            qconfig=mod.qconfig,
        )
        return qat_mod
