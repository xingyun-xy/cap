# Copyright (c) Changan Auto. All rights reserved.
import logging
from collections.abc import Mapping
from typing import List, Union

from torch import Tensor, nn

from cap.registry import OBJECT_REGISTRY

try:
    from mmcv.utils import ConfigDict
    from cap.models import build_backbone

    _MMDET_IMPORTED = True
except ImportError:
    _MMDET_IMPORTED = False

logger = logging.getLogger(__name__)

__all__ = ["MMDetBackboneAdaptor"]


@OBJECT_REGISTRY.register
class MMDetBackboneAdaptor(nn.Module):
    """
    Adaptor of mmdetection backbones to use in CAP.

    Args:
        backbone: either a backbone module or a cap config dict
        that defines a backbone.
        The backbone takes a 4D image tensor and returns a list of tensors.
    """

    def __init__(
        self,
        backbone: Union[nn.Module, Mapping],
    ):

        if _MMDET_IMPORTED is False:
            raise ModuleNotFoundError(
                "cap and mmcv is required for MMDetBackboneAdaptor."
            )

        super().__init__()
        if isinstance(backbone, Mapping):
            backbone = build_backbone(ConfigDict(backbone))
        self.backbone = backbone

    def forward(self, x) -> List[Tensor]:
        outs = self.backbone(x)
        assert isinstance(
            outs, (list, tuple)
        ), "cap backbone should return a list of tensors!"
        return outs
