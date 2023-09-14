from numbers import Integral

import torch
from torch import nn
from torch.nn.modules.utils import _pair


class BaseGridGenerator(nn.Module):
    """
    Generate base grid for affine transform or perspective transform.

    Args:
        size (int or tuple(int, int)): Output size.
        with_third_channel (bool): Whether append the all ones third channel.
    """

    def __init__(self, size, with_third_channel):
        super(BaseGridGenerator, self).__init__()
        size = _pair(size)
        assert isinstance(size[0], Integral) and isinstance(
            size[1], Integral
        ), "param 'size' must be int or Tuple[int, int]"
        assert isinstance(
            with_third_channel, bool
        ), "param 'with_third_channel' must be a boolean"

        self.size = size
        self.with_third_channel = with_third_channel
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))

    def forward(self):
        from .quantized.functional import base_grid_generator

        ret = base_grid_generator(
            self.size, self.with_third_channel, self.scale.device
        )

        return ret.to(dtype=torch.float)
