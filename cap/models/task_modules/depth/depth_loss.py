from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

__all__ = ["MultiStrideDepthLoss"]


@OBJECT_REGISTRY.register
class MultiStrideDepthLoss(nn.Module):
    """Multi-Stride Depth Loss.

    Args:
        multi_scale_weight: Weight of each scale prediction.
        depth_scale: Scale of depth.
        max_depth: Maximum limit of depth.
        min_depth: Minimum limit of depth.
        avoid_zero: Epsilon.
        out_strides: Stride of output feature map.
        output_confidence: Whether the outputs contains confidence predictions.
        confidence_scale_weight: Confidence weight of each scale prediction.
    """

    def __init__(
        self,
        multi_scale_weight: List[float],
        depth_scale: int,
        max_depth: float,
        min_depth: float,
        avoid_zero: float,
        out_strides: List[int],
        output_confidence: bool = True,
        confidence_scale_weight: List[float] = None,
    ):

        super().__init__()
        if output_confidence:
            if confidence_scale_weight is None:
                self.confidence_scale_weight = multi_scale_weight
            else:
                assert len(confidence_scale_weight) == len(
                    multi_scale_weight
                ), "%d vs. %d" % (
                    len(confidence_scale_weight),
                    len(multi_scale_weight),
                )
                self.confidence_scale_weight = confidence_scale_weight
        else:
            self.confidence_scale_weight = [None] * len(multi_scale_weight)

        self.multi_scale_weight = multi_scale_weight
        self.max_depth = max_depth / depth_scale
        self.min_depth = min_depth / depth_scale
        self.avoid_zero = avoid_zero / depth_scale
        self.out_strides = out_strides
        self.output_confidence = output_confidence

    @autocast(enabled=False)
    def forward(self, preds: List[torch.Tensor], label: torch.Tensor):

        assert len(preds) == len(self.multi_scale_weight), "%d vs. %d" % (
            len(preds),
            len(self.multi_scale_weight),
        )

        # convert to float32 while using amp
        preds = [pred.float() for pred in preds]

        losses = OrderedDict()
        for i, (pred, scale, conf_scale) in enumerate(
            zip(preds, self.multi_scale_weight, self.confidence_scale_weight)
        ):  # noqa
            pred = F.interpolate(
                pred,
                scale_factor=label.shape[2] / pred.shape[2],
                mode="bilinear",
                align_corners=False,
            )
            if self.output_confidence:
                depth, conf = torch.split(pred, 1, dim=1)
            else:
                depth = pred
                conf = None

            # 1. depth loss
            depth = depth.clamp(self.min_depth, self.max_depth)

            label_mask_bigger_min = label >= self.min_depth
            label_mask_smaller_max = label <= self.max_depth
            label_valid_mask = label_mask_bigger_min * label_mask_smaller_max
            pred_valid = depth * label_valid_mask
            label_valid = label * label_valid_mask

            valid_element = label_valid_mask.sum((1, 2, 3))
            l1_depth_error = (
                pred_valid - label_valid
            ).abs_() * label_valid_mask
            rel_norm = (
                1
                + 1.2 * ((self.max_depth - label_valid) / self.max_depth) ** 2
            )

            quad_rel_depth_error = rel_norm * l1_depth_error
            loss_quadl1 = (
                quad_rel_depth_error.sum((1, 2, 3))
                / (valid_element + self.avoid_zero)
                * (valid_element > 0)
            )

            losses[f"loss_stride_{self.out_strides[i]}"] = loss_quadl1 * scale

            # 2. confidence loss
            if self.output_confidence:
                assert conf_scale is not None
                conf_valid = conf * label_valid_mask
                l1_depth_error_block = l1_depth_error.detach() / (
                    label_valid + self.avoid_zero
                )
                l1_conf_error = (
                    conf_valid - torch.exp(-l1_depth_error_block)
                ).abs_() * label_valid_mask
                loss_conf = (
                    l1_conf_error.sum((1, 2, 3))
                    / (valid_element + self.avoid_zero)
                    * (valid_element > 0)
                )

                losses[f"conf_loss_stride_{self.out_strides[i]}"] = (
                    loss_conf * conf_scale
                )

        return losses
