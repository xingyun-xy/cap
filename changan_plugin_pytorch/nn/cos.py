import torch

from .segment_lut import SegmentLUT


class Cos(torch.nn.Module):
    "Module implementation of torch.cos"

    def __init__(self):
        super(Cos, self).__init__()

        self.cos = SegmentLUT(torch.cos, False, None, None, "curvature")

    def forward(self, input):
        return self.cos(input)
