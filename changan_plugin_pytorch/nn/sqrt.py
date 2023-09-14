import torch

from .segment_lut import SegmentLUT


class Sqrt(torch.nn.Module):
    "Module implementation of torch.sqrt"

    def __init__(self):
        super(Sqrt, self).__init__()

        self.sqrt = SegmentLUT(torch.sqrt, True, None, None, "curvature")

    def forward(self, input):
        return self.sqrt(input)
