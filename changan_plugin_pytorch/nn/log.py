import torch

from .segment_lut import SegmentLUT


class HardLog(torch.nn.Module):
    "Module implementation of torch.log"

    def __init__(self):
        super(HardLog, self).__init__()

        self.log = SegmentLUT(
            lambda x: torch.clamp(torch.log(x), min=-10),
            True,
            None,
            None,
            "curvature",
        )

    def forward(self, input):
        return self.log(input)
