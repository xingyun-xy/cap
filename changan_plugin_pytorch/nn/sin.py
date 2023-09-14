import torch

from .segment_lut import SegmentLUT


class Sin(torch.nn.Module):
    "Module implementation of torch.sin"

    def __init__(self):
        super(Sin, self).__init__()

        self.sin = SegmentLUT(torch.sin, True, None, None, "curvature")

    def forward(self, input):
        return self.sin(input)
