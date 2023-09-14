from typing import Dict

import torch.nn.functional as F
from torch import nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["SemSegDecoder"]


@OBJECT_REGISTRY.register
class SemSegDecoder(nn.Module):
    """Generate prediction class map for IPM-Seg task.

    Args:
        output_name: The key corresponding to the prediction class map.
            Default: semseg_pred
    """

    def __init__(
        self,
        output_name: str = "semseg_pred",
    ) -> Dict:
        super().__init__()
        self.output_name = output_name

    def forward(self, pred, label):
        if pred is not None:
            if isinstance(pred, tuple):
                pred = pred[0]
            if isinstance(label, dict):
                label = label["gt_seg"]
            h, w = label.shape[-2:]
            pred = F.interpolate(pred, size=(h, w), mode="bilinear")
            output_label = F.softmax(pred, dim=1).max(dim=1)[1]
            output = {self.output_name: output_label}
        else:
            output = {}
        return output
