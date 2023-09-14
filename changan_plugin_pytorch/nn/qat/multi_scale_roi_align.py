from typing import List, Tuple

import torch
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import (
    multi_scale_roi_align as float_multi_scale_roi_align,
)
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import nn

from .. import multi_scale_roi_align as float_multi_scale_roi_align
from .functional import requantize
from .roi_align import RoIAlign


class MultiScaleRoIAlign(float_multi_scale_roi_align.MultiScaleRoIAlign):
    """
    Args:
        same as float version.
    """

    _FLOAT_MODULE = float_multi_scale_roi_align.MultiScaleRoIAlign

    def __init__(
        self,
        output_size: Tuple[int, int],
        feature_strides: List[int],
        sampling_ratio: int = 1,
        interpolate_mode: str = "bilinear",
        canonical_box_size: int = 224,
        canonical_level: int = 4,
        aligned=False,
        qconfig=None,
    ):
        self.qconfig = qconfig
        super(MultiScaleRoIAlign, self).__init__(
            output_size,
            feature_strides,
            sampling_ratio,
            interpolate_mode,
            canonical_box_size,
            canonical_level,
            aligned,
        )
        self.activation_post_process = self.qconfig.activation()
        assert self.qconfig, "qconfig must be provided for QAT module"
        assert (
            self.qconfig.activation
        ), "qconfig must have member activation for qat.MultiscaleRoIAlign"
        self.activation_post_process.disable_observer()

        march = get_march()

        if (
            march == March.BERNOULLI2 or march == March.BERNOULLI
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
            box_lists(list[QTensor/Tensor[L, 4]]): a list of N boxes.
                QTensor or Tensor
        """
        assert_qtensor_input(x)

        march = get_march()

        # requantize input to the max input scale
        max_scale = max([per_feature.q_scale() for per_feature in x])
        self.activation_post_process.set_qparams(max_scale)
        info = qinfo(x[0].dtype)
        if get_march() == March.BERNOULLI:
            if len(x) > 1:
                for per_feature in x:
                    assert (
                        per_feature.q_scale().item() == x[0].q_scale().item()
                    ), "BERNOULLI only supports features with same scale input"
        else:
            x = [
                QTensor(
                    torch.ops.changan.scale_quanti(
                        per_feature.as_subclass(torch.Tensor),
                        max_scale,
                        per_feature.q_zero_point(),
                        -1,
                        info.min,
                        info.max,
                        True,
                        False,
                        "floor" if march == March.BERNOULLI else "bpu_round",
                        march,
                    ),
                    # use qat requantize to align with quantized results
                    # requantize(
                    #     per_feature.as_subclass(torch.Tensor),
                    #     per_feature.q_scale(),
                    #     max_scale,
                    #     per_feature.q_zero_point(),
                    #     per_feature.q_zero_point(),
                    #     -1,
                    #     per_feature.dtype,
                    #     per_feature.dtype,
                    #     march,
                    # ),
                    max_scale,
                    per_feature.dtype,
                )
                for per_feature in x
            ]

        if isinstance(box_lists[0], QTensor):
            for box in box_lists:
                assert box.q_scale().item() == 0.25, (
                    "invalid input box scale, "
                    + "we expect 0.25, but receive {}".format(
                        box.q_scale().item()
                    )
                )

        result = super(MultiScaleRoIAlign, self).forward(x, box_lists)

        # if only one feature map, result is QTensor.
        # if no boxes, result is empty, directly return QTensor.
        # Normally result is float tensor, need to fake quantize
        # to uniform scale.
        if not isinstance(result, QTensor):
            result = (
                self.activation_post_process(result)
                if result.numel()
                else QTensor(result, x[0].q_scale(), x[0].dtype)
            )
        return result

    def _get_aligners(self):
        aligners = nn.ModuleList(
            RoIAlign(
                self.output_size,
                1 / stride,
                self.sampling_ratio,
                self.aligned,
                self.mode,
                self.qconfig,
            )
            for stride in self.feature_strides
        )

        for mod in aligners:
            mod._allow_tensor_roi = True

        return aligners

    def _per_level_align_result(self, x):
        return x.as_subclass(torch.Tensor)

    def _data(self, x):
        return x[0].as_subclass(torch.Tensor)

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
            output_size=mod.output_size,
            feature_strides=mod.feature_strides,
            sampling_ratio=mod.sampling_ratio,
            interpolate_mode=mod.mode,
            canonical_box_size=mod.canonical_box_size,
            canonical_level=mod.canonical_level,
            aligned=mod.aligned,
            qconfig=mod.qconfig,
        )

        return qat_mod
