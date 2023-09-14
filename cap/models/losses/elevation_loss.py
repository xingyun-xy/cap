# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list

__all__ = [
    "ElevationLoss",
    "GammaLoss",
    "GroundLoss",
]


@OBJECT_REGISTRY.register
class ElevationLoss(nn.Module):
    """Calculate elevation loss.

    Args:
        pred_gammas_name: gamma predction`s name.
        gt_gamma_name: gamma gt`s name.
        pred_gammas_name: ground predction`s name.
        gt_ground_name: ground gt`s name.
        gamma_loss: gamam loss.
        ground_loss: ground loss.
    """

    def __init__(
        self,
        pred_gammas_name: Optional[str] = None,
        gt_gamma_name: Optional[str] = None,
        pred_ground_name: Optional[str] = None,
        gt_ground_name: Optional[str] = None,
        gamma_loss: Optional[torch.nn.Module] = None,
        ground_loss: Optional[torch.nn.Module] = None,
    ):
        super(ElevationLoss, self).__init__()
        assert isinstance(
            gamma_loss, nn.Module
        ), "gamma loss must be subclass of torch.nn.Module"
        self.pred_gammas_name = pred_gammas_name
        self.gt_gamma_name = gt_gamma_name
        self.pred_ground_name = pred_ground_name
        self.gt_ground_name = gt_ground_name

        self.gamma_loss = gamma_loss
        self.ground_loss = None
        if ground_loss:
            assert isinstance(
                ground_loss, nn.Module
            ), "ground loss must be subclass of torch.nn.Module"
            self.ground_loss = ground_loss

    @autocast(enabled=False)
    def forward(self, pred_dict, target_dict):
        result_dict = {}
        # cast to fp32
        pred_gammas = [
            data.float() for data in pred_dict[self.pred_gammas_name]
        ]  # maybe contain two gamma imgs
        gt_gamma = target_dict[self.gt_gamma_name]
        result_dict["gamma_loss"] = self.gamma_loss(pred_gammas, gt_gamma)

        if self.ground_loss:
            pred_ground = [
                data.float() for data in pred_dict[self.pred_ground_name]
            ]
            gt_ground = target_dict[self.gt_ground_name]
            result_dict["ground_loss"] = self.ground_loss(
                pred_ground, gt_ground
            )
        return result_dict


@OBJECT_REGISTRY.register
class GammaLoss(nn.Module):
    """Calculate gamma l1 loss.

    Args:
        low: low threshold value durning calculating.
        high: high threshold value durning calculating.
        loss_weight: loss weight.
        gamma_scale: gamma scale .
    """

    def __init__(
        self,
        low: float,
        high: float,
        loss_weight: float = 1.0,
        gamma_scale: float = 1.0,
    ):
        super(GammaLoss, self).__init__()
        self.loss_weight = loss_weight
        self.low = low * gamma_scale
        self.high = high * gamma_scale
        self.gamma_scale = gamma_scale

    def forward(
        self,
        pred_gammas: Union[torch.Tensor, Sequence[torch.Tensor]],
        gt_gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of gamma loss.

        Args:
            pred_gammas: multi-stride predict gamma maps.
            gt_gamma: gt gamma map.
        """
        if isinstance(gt_gamma, (list, tuple)):
            gt_gamma = gt_gamma[0]
        pred_gammas = _as_list(pred_gammas)
        gt_gamma_mask = (gt_gamma > self.low) * (gt_gamma < self.high)
        valid_element = gt_gamma_mask.sum() + 0.0001

        loss = 0
        for gamma in pred_gammas:
            diff = (gamma - gt_gamma).abs() * gt_gamma_mask
            loss += diff.sum() / valid_element
        return loss * self.loss_weight


@OBJECT_REGISTRY.register
class GroundLoss(nn.Module):
    """Calculate ground loss.

    Args:
        loss_weight: loss weight.
        loss_type: 'l1' or 'cosin ssim' loss.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        loss_type: Optional[str] = None,
    ):
        super(GroundLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type

    def forward(
        self,
        pred_ground: Union[torch.Tensor, Sequence[torch.Tensor]],
        gt_ground: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of ground loss.

        Args:
            pred_ground: pred ground norm, shape (b,1,3,1).
            gt_ground: gt ground norm.
        """
        if isinstance(gt_ground, (list, tuple)):
            gt_ground = gt_ground[0]
        if isinstance(pred_ground, (list, tuple)):
            pred_ground = pred_ground[0]
        if self.loss_type == "l1":
            loss = (gt_ground.squeeze() - pred_ground.squeeze()).abs().mean()
        elif self.loss_type == "cosin":
            loss = (
                F.cosine_similarity(gt_ground, pred_ground.squeeze(-1), dim=1)
                .abs()
                .mean()
            )
        return loss * self.loss_weight
