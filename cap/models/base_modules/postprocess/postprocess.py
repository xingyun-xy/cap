# Copyright (c) Changan Auto. All rights reserved.

from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = ["PostProcessorBase"]


class PostProcessorBase(nn.Module):
    """Interface class of post processor."""

    @abstractmethod
    def forward(
        self,
        pred: Union[
            torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]
        ],
        meta_data: Optional[Dict[str, Any]] = None,
    ):
        """Do post process for model predictions.

        Args:
            pred: Prediction tensors.
            meta_data: Meta data used in post processor, e.g. image width,
                height.
        """
        raise NotImplementedError
