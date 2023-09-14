# Copyright (c) Changan Auto. All rights reserved.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.models.losses.real3d_losses import (
    hm_focal_loss,
    hm_l1_loss,
    sigmoid_and_clip,
)
from cap.registry import OBJECT_REGISTRY

__all__ = ["BEVDiscreteObjectLoss"]


@OBJECT_REGISTRY.register
class BEVDiscreteObjectLoss(nn.Module):
    """Calculate BEV discrete obj heatmap loss and dimension loss.

    Classification uses focal loss by default.
    Regression (dimension, rotation, center_offset
    etc.) use L1 loss.

    Args:
        loss_weights(dict): default: None
            Global loss weight for each sub-loss. Default
            weight is 1.0. (e.g. bev_discobj_hm:1, bev_discobj_dim:1,
            bev_discobj_rot:1 etc.).
        use_focal_hm_loss (bool): Use focal loss for heatmap otherwise l1 loss.
    Returns:
        a dict contains loss.
    """

    def __init__(
        self,
        loss_weights,
        use_focal_hm_loss,
    ):
        super(BEVDiscreteObjectLoss, self).__init__()
        self.loss_weights = defaultdict(lambda: 1.0)
        self.use_focal_hm_loss = use_focal_hm_loss

        if loss_weights is not None:
            self.loss_weights.update(**loss_weights)
        self.reg_keys = list(self.loss_weights.keys())
        if "bev_discobj_hm" in self.reg_keys and use_focal_hm_loss:
            self.reg_keys.remove("bev_discobj_hm")

    @autocast(enabled=False)
    def forward(self, pred_dict, target_dict):
        """Calculate loss between pred and target items.

        Args:
            pred_dict (dict): Predict bev discrete obj output (e.g.,
                pred_bev_discobj_hm, pred_bev_discobj_wh, etc).
            target_dict (dict): Target contains bev discrete obj ground truth
        """
        assert "gt_bev_discrete_obj" in target_dict
        target = target_dict["gt_bev_discrete_obj"]

        pred_hm = pred_dict["pred_bev_discobj_hm"]
        target_hm = target["bev_discobj_hm"]

        assert "bev_discobj_weight_hm" in target
        weight_mask = target["bev_discobj_weight_hm"]
        ignore_mask = (weight_mask == 0).float()

        pred_hm = sigmoid_and_clip(pred_hm)
        result_dict = {}

        if self.use_focal_hm_loss:
            loss_weight = self.loss_weights["bev_discobj_hm"]
            hm_loss = (
                hm_focal_loss(pred_hm, target_hm, torch.zeros_like(pred_hm))
                * loss_weight
            )
        else:
            loss_weight = self.loss_weights["bev_discobj_hm"]
            hm_loss = (
                hm_l1_loss(
                    pred_hm,
                    target_hm,
                    weight_mask,
                    ignore_mask,
                    heatmap_type=None,
                )
                * loss_weight
            )
        result_dict["loss_bev_discobj_hm"] = hm_loss

        # regression map use L1 loss
        for key in self.reg_keys:
            pred_key = "pred_%s" % key
            if pred_key not in pred_dict:
                continue
            loss_weight = self.loss_weights[key]
            loss = (
                hm_l1_loss(
                    pred_dict[pred_key],
                    target[key],
                    weight_mask,
                    ignore_mask,
                    heatmap_type=None,
                )
                * loss_weight
            )
            result_dict["loss_%s" % key] = loss

        return result_dict
