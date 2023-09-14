# Copyright (c) Changan Auto. All rights reserved.

from typing import Mapping, Optional, Sequence, Union

from torch import nn

from cap.models.base_modules.target import ReshapeTarget
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .head import SpatialTransfomer

__all__ = ["BEVTarget"]


@OBJECT_REGISTRY.register
class BEVTarget(nn.Module):
    """Generate bev targets for bev task.

        e.g. Rotate bev gt to keep consistency between bev input and bev gt,
            reshape gt_bev_seg to specific shape.

    Args:
        gt_name (str or list(str) ): name of bev gt,
            or in list with more than one name.
        bev_height (int): height of bev gt.
        bev_width (int): height of bev gt.
        gt_shape (Sequence): the shape of bev gt.

    """

    def __init__(
        self,
        gt_name: Union[Sequence[str], str],
        bev_height: int,
        bev_width: int,
        gt_shape: Optional[Sequence] = None,
    ):
        super(BEVTarget, self).__init__()
        self.gt_name = _as_list(gt_name)

        self.st = SpatialTransfomer(
            bev_height,
            bev_width,
            mode="nearest",
            padding_mode="border",
            use_horizon_grid_sample=False,
        )
        self.reshape_target_dict = {}
        for name in self.gt_name:
            self.reshape_target_dict[name] = ReshapeTarget(name, gt_shape)

    def forward(self, label_dict: Mapping, pred_dict: Mapping) -> Mapping:
        bev_rot_mat = label_dict.get("bev_rot_mat", None)
        for name in self.gt_name:
            if bev_rot_mat is not None:
                data_type = label_dict[name].dtype
                label_dict[name] = self.st(
                    label_dict[name].float(), bev_rot_mat
                )[0].type(data_type)
            if name in label_dict:
                label_dict = self.reshape_target_dict[name](
                    label_dict, pred_dict
                )
        return label_dict
