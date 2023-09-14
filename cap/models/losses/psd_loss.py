# Copyright (c) Changan Auto. All rights reserved.
from typing import List

import torch
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY

__all__ = [
    "SuperPSDLoss",
    "SuperPSDLocalLoss",
    "SuperPSDGlobalLoss",
]


@OBJECT_REGISTRY.register
class SuperPSDLoss(nn.Module):
    """Super psd loss.

    Args:
        global_loss: Loss function for global head.
        local_loss: Loss function for local head.
    """

    def __init__(
        self, global_loss: nn.Module = None, local_loss: nn.Module = None
    ):
        super().__init__()
        self.global_loss = None
        self.local_loss = None
        if global_loss is not None:
            self.global_loss = global_loss
        if local_loss is not None:
            self.local_loss = local_loss

    def forward(self, preds, targets, grads):

        total_loss = dict()  # noqa
        total_target_dict = targets[0]
        total_grad_dict = grads[0]
        global_preds, local_preds = preds[0:5], preds[5:]
        if self.global_loss is not None:
            global_loss = self.global_loss(
                global_preds, total_target_dict, total_grad_dict
            )
            total_loss.update(global_loss)
        if self.local_loss is not None:
            local_loss = self.local_loss(
                local_preds, total_target_dict, total_grad_dict
            )
            total_loss.update(local_loss)
        return total_loss


@OBJECT_REGISTRY.register
class SuperPSDGlobalLoss(nn.Module):
    """Super psd global loss function.

    Args:
        classification_loss: Slot center point classification loss
            function for global head.
        offset_loss: Coordinate offset loss function for global head.
        occupancy_loss: Loss function used to optimize whether slot
            is occupancy.
        slot_type_loss: Loss function for slot type classification.
        direction_loss: Loss function for slot direction regression.
        loss_weights: Weight list for all five loss function.
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        offset_loss: nn.Module,
        occupancy_loss: nn.Module,
        slot_type_loss: nn.Module,
        direction_loss: nn.Module,
        loss_weights: List[float] = [1, 1, 1, 1, 1],  # noqa
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.classification_loss = classification_loss
        self.offset_loss = offset_loss
        self.occupancy_loss = occupancy_loss
        self.slot_type_loss = slot_type_loss
        self.direction_loss = direction_loss

    def forward(self, global_preds, global_target_dict, global_grad_dict):
        classification, offset, occupancy, slot_type, direction = global_preds
        (
            classification_obj,
            offset_obj,
            occupancy_obj,
            slot_type_obj,
            direction_obj,
        ) = (
            global_target_dict["global_classification_obj"],
            global_target_dict["global_offset_obj"],
            global_target_dict["global_occupancy_obj"],
            global_target_dict["global_slot_type_obj"],
            global_target_dict["global_direction_obj"],
        )
        (
            classification_grad,
            offset_grad,
            occupancy_grad,
            slot_type_grad,
            direction_grad,
        ) = (
            global_grad_dict["global_classification_grad"],
            global_grad_dict["global_offset_grad"],
            global_grad_dict["global_occupancy_grad"],
            global_grad_dict["global_slot_type_grad"],
            global_grad_dict["global_direction_grad"],
        )
        classification = torch.sigmoid(classification)
        occupancy = torch.sigmoid(occupancy)
        slot_type = torch.sigmoid(slot_type)
        direction = torch.tanh(direction)
        classification_loss = self.classification_loss(
            logits=classification, labels=classification_obj
        )
        offset_loss = self.offset_loss(logits=offset, labels=offset_obj)
        occupancy_loss = self.occupancy_loss(
            logits=occupancy, labels=occupancy_obj
        )
        slot_type_loss = self.slot_type_loss(
            logits=slot_type, labels=slot_type_obj
        )
        direction_loss = self.direction_loss(
            logits=direction, labels=direction_obj
        )

        classification_loss = (
            classification_grad * classification_loss * self.loss_weights[0]
        )
        classification_loss = torch.sum(classification_loss) / (
            torch.sum(classification_obj) + 1
        )
        offset_loss = offset_grad * offset_loss * self.loss_weights[1]
        offset_loss = torch.sum(offset_loss) / (
            torch.sum(classification_obj) + 1
        )
        occupancy_loss = occupancy_grad * occupancy_loss * self.loss_weights[2]
        occupancy_loss = torch.sum(occupancy_loss) / (
            torch.sum(classification_obj) + 1
        )
        slot_type_loss = slot_type_grad * slot_type_loss * self.loss_weights[3]
        slot_type_loss = torch.sum(slot_type_loss) / (
            torch.sum(classification_obj) + 1
        )
        direction_loss = direction_grad * direction_loss * self.loss_weights[4]
        direction_loss = torch.sum(direction_loss) / (
            torch.sum(classification_obj) + 1
        )
        total_loss = dict(  # noqa
            global_classification_loss=classification_loss,
            global_offset_loss=offset_loss,
            global_occupancy_loss=occupancy_loss,
            global_slot_type_loss=slot_type_loss,
            global_direction_loss=direction_loss,
        )
        return total_loss


@OBJECT_REGISTRY.register
class SuperPSDLocalLoss(nn.Module):
    """Super psd local loss function.

    Args:
        classification_loss: Slot corner point classification loss
            function for local head.
        offset_loss: Coordinate offset loss function for local head.
        sline_angle_loss: Sline angle loss function for local head.
        point_type_loss: Loss function for point type classification.
        loss_weights: Weight list for all four loss functions.
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        offset_loss: nn.Module,
        sline_angle_loss: nn.Module,
        point_type_loss: nn.Module,
        loss_weights: List[int] = [1, 1, 1, 1],  # noqa
    ):
        super().__init__()
        self.classification_loss = classification_loss
        self.offset_loss = offset_loss
        self.sline_angle_loss = sline_angle_loss
        self.point_type_loss = point_type_loss
        self.loss_weights = loss_weights

    def forward(self, local_preds, local_target_dict, local_grad_dict):
        # local_preds: [batch_size, 24, 112, 112]
        # local_targets: [[[x1, y1, x2, y2, x3, y3, x4, y4,
        #                p1_vec_x, p1_vec_y, p2_vec_x, p2_vec_y,
        #                p3_vec_x, p3_vec_y, p4_vec_x, p4_vec_y,
        #                p1_type, p2_type, p3_type, p4_type], []],[[]]]
        classification, offset, sline_angle, point_type = local_preds
        classification_obj, offset_obj, sline_angle_obj, point_type_obj = (
            local_target_dict["local_classification_obj"],
            local_target_dict["local_offset_obj"],
            local_target_dict["local_sline_angle_obj"],
            local_target_dict["local_point_type_obj"],
        )
        classification_grad, offset_grad, sline_angle_grad, point_type_grad = (
            local_grad_dict["local_classification_grad"],
            local_grad_dict["local_offset_grad"],
            local_grad_dict["local_sline_angle_grad"],
            local_grad_dict["local_point_type_grad"],
        )

        classification = torch.sigmoid(classification)
        sline_angle = torch.tanh(sline_angle)
        point_type = torch.sigmoid(point_type)
        classification_loss = (
            self.classification_loss(
                logits=classification,
                labels=classification_obj,
                grad_tensor=classification_grad,
            )
            * self.loss_weights[0]
        )
        offset_loss = (
            self.offset_loss(logits=offset, labels=offset_obj)
            * self.loss_weights[1]
        )
        sline_angle_loss = (
            self.sline_angle_loss(logits=sline_angle, labels=sline_angle_obj)
            * self.loss_weights[2]
        )
        point_type_loss = (
            self.point_type_loss(logits=point_type, labels=point_type_obj)
            * self.loss_weights[3]
        )
        offset_loss *= offset_grad
        offset_loss = torch.sum(offset_loss) / (torch.sum(offset_grad) + 1)
        sline_angle_loss *= sline_angle_grad
        sline_angle_loss = torch.sum(sline_angle_loss) / (
            torch.sum(sline_angle_grad) + 1
        )
        point_type_loss *= point_type_grad
        point_type_loss = torch.sum(point_type_loss) / (
            torch.sum(point_type_grad) + 1
        )
        total_loss = dict(  # noqa
            local_classification_loss=classification_loss,
            local_offset_loss=offset_loss,
            local_sline_angle_loss=sline_angle_loss,
            local_point_type_loss=point_type_loss,
        )
        return total_loss
