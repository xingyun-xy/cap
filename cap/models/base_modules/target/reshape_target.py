# Copyright (c) Changan Auto. All rights reserved.

from typing import Mapping, Optional, Sequence

import torch

from cap.registry import OBJECT_REGISTRY

__all__ = ["ReshapeTarget"]


@OBJECT_REGISTRY.register
class ReshapeTarget(object):
    """Reshape target data in label_dict to specific shape.

    Args:
        data_name (str): name of original data to reshape.
        shape (Sequence): the new shape.
    """

    def __init__(self, data_name: str, shape: Optional[Sequence] = None):
        self.data_name = data_name
        self.shape = shape

    def __call__(self, label_dict: Mapping, pred_dict: Mapping) -> Mapping:
        shape = self.shape if self.shape else label_dict[self.data_name].shape
        if isinstance(label_dict[self.data_name], Mapping):
            for k, v in label_dict[self.data_name].items():
                label_dict[self.data_name][k] = torch.reshape(v, shape)
        elif isinstance(label_dict[self.data_name], Sequence):
            for k, v in enumerate(label_dict[self.data_name]):
                label_dict[self.data_name][k] = torch.reshape(v, shape)
        else:
            label_dict[self.data_name] = torch.reshape(
                label_dict[self.data_name], shape
            )
        return label_dict
