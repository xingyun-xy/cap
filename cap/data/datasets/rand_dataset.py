# Copyright (c) Changan Auto. All rights reserved.

import copy
from typing import Any

import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY

__all__ = ["RandDataset"]


@OBJECT_REGISTRY.register
class RandDataset(data.Dataset):
    def __init__(self, length: int, example: Any, clone: bool = True):
        self.length = length
        self.example = example
        self.clone = clone

    def __getitem__(self, index):
        if self.clone:
            return copy.deepcopy(self.example)
        else:
            return self.example

    def __len__(self):
        return self.length
