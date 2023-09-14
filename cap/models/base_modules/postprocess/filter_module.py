# Copyright (c) Changan Auto. All rights reserved.
from typing import Optional, Tuple

import torch.nn as nn
from changan_plugin_pytorch.nn.functional import filter as plugin_filter

from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FilterModule(nn.Module):
    """Based Filter Module, filter by threshold.

    Args:
        threshold (float): Threshold, the lower bound of output.
        idx_range (Optional[Tuple[int, int]]): The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
    """

    def __init__(
        self,
        threshold: float,
        idx_range: Optional[Tuple[int, int]] = None,
    ):
        super(FilterModule, self).__init__()
        self.threshold = threshold
        self.idx_range = idx_range

    def forward(self, *inputs):
        """
        Forward method.

        Args:
            inputs (Union[Tuple[Tensor], Tuple[QTensor]]): Data in NCHW format.
                Each input shold have the same size in N, H, W.
                The output will be selected according to the first input.

        Returns:
            Union[List[List[Tensor]], List[List[QTensor]]]:
            A list with same length of batch size, and each element contains:
            max_value: Flattened max value within idx_range in channel dim.
            max_idx: Flattened max value index in channel dim.
            coord: The original coordinates of the output data in the
                input data in the shape of [M, (h, w)].
            (multi) data: Filtered data in the shape of [M, C].
        """
        return plugin_filter(
            *inputs, threshold=self.threshold, idx_range=self.idx_range
        )
