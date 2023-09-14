# Copyright (c) Changan Auto. All rights reserved.
import torch
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY

__all__ = ["FCNDecoder"]


@OBJECT_REGISTRY.register
class FCNDecoder(nn.Module):
    """FCN Decoder.

    Args:
        upsample_output_scale: Output upsample scale. Default: 8.
    """

    def __init__(
        self,
        upsample_output_scale=8,
    ):
        super(FCNDecoder, self).__init__()
        self.upsample_output_scale = upsample_output_scale
        self.resize = torch.nn.Upsample(
            scale_factor=upsample_output_scale,
            align_corners=False,
            mode="bilinear",
        )

    def forward(self, pred):
        pred = self.resize(pred)
        pred = pred.argmax(dim=1)
        return pred
