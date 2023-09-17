# Copyright (c) Changan Auto. All rights reserved.
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from cap.registry import OBJECT_REGISTRY
from projects.panorama.configs.resize.common import bev_depth_loss_coeff

__all__ = ["bev_matrixvt_depth_loss", "bev_matrixvt"]


@OBJECT_REGISTRY.register
class bev_matrixvt(nn.Module):
    """
    The basic structure of bev task.
    """

    SUPPORTED_MODES = ("train", "val", "onnx", "test")

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        mode="train",
        depth_channels: Optional[int] = None,
        downsample_factor: Optional[int] = None,
        dbound: Optional[list] = None,
        neck: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        loss_v2: Optional[nn.Module] = None,
        loss_v3: Optional[nn.Module] = None,
        depthnet: Optional[nn.Module] = None,
    ):
        super(bev_matrixvt, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.depthnet = depthnet
        self.head = head
        self.desc = desc
        self.loss = loss
        self.loss_v2 = loss_v2
        self.loss_v3 = loss_v3
        self.mode = mode
        self.depth_channels = depth_channels
        self.downsample_factor = downsample_factor
        self.dbound = dbound

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Invalid Mode: {mode}")

    def forward(self, data: Dict, outname=None):

        N, C, H, W = data["img"].shape
        img = data["img"]
        depth_img = data["img"].view(-1, 1, N, C, H, W)
        sensor2ego_mats = data["sensor2ego_mats"]  # [:, 0:1, :, :3, :]
        intrin_mats = data["intrin_mats"]  # [:, 0:1, :,:3, :3]
        ida_mats = data["ida_mats"]  # [:, 0:1, :,:]
        sensor2sensor_mats = data["sensor2sensor_mats"]
        bda_mat = data["bda_mat"]

        mlp_input = data["mlp_input"] if self.mode == "onnx" else None
        circle_map = data["circle_map"] if self.mode == "onnx" else None
        ray_map = data["ray_map"] if self.mode == "onnx" else None
        timestamps = None

        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            imH,
            imW,
        ) = depth_img.shape
        features_resnet = self.backbone(data["img"])
        features_seconfpn = self.neck(features_resnet)[0]

        features_seconfpn = features_seconfpn.reshape(
            batch_size,
            num_sweeps,
            num_cams,
            features_seconfpn.shape[1],
            features_seconfpn.shape[2],
            features_seconfpn.shape[3],
        )
        if self.mode == "train":
            features, depth_pred = self.depthnet(
                depth_img,
                features_seconfpn,
                sensor2ego_mats,
                intrin_mats,
                ida_mats,
                sensor2sensor_mats,
                bda_mat,
                mlp_input,
                circle_map,
                ray_map,
                timestamps,
                is_return_depth=True,
            )

            gt_boxes = data["gt_boxes_batch"]
            gt_labels = data["gt_labels_batch"]

            preds, targets = self.head(features, gt_boxes, gt_labels)

            depth_loss = self.loss_v3(data, depth_pred, preds, targets)
            out_dict = OrderedDict()

            out_dict.update(depth_loss)
            return out_dict

        else:
            features = self.depthnet(
                depth_img,
                features_seconfpn,
                sensor2ego_mats,
                intrin_mats,
                ida_mats,
                sensor2sensor_mats,
                bda_mat,
                mlp_input,
                circle_map,
                ray_map,
                timestamps,
                is_return_depth=False,
            )
            preds = self.head(features)
            out = []
            for i in range(6):
                out.append(preds[0][i]["reg"])
                out.append(preds[0][i]["height"])
                out.append(preds[0][i]["dim"])
                out.append(preds[0][i]["rot"])
                out.append(preds[0][i]["vel"])
                out.append(preds[0][i]["heatmap"])
            if "img_metas_batch" in data.keys():
                return preds, data["img_metas_batch"]
            else:
                return out


from cap.models.task_modules.bev.bev_depth_head import CenterHead_loss


@OBJECT_REGISTRY.register
class bev_matrixvt_depth_loss(nn.Module):
    """
    The basic structure of Camera3D task.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        desc: Description module.
        losses: Losses module.
    """

    def __init__(
        self,
        BEVDepthHead_loss: CenterHead_loss,
        BEVDepthHead_lossv2: CenterHead_loss,
        depth_channels: Optional[int] = None,
        downsample_factor: Optional[int] = None,
        dbound: Optional[list] = None,
    ):
        super(bev_matrixvt_depth_loss, self).__init__()
        self.depth_channels = depth_channels
        self.downsample_factor = downsample_factor
        self.dbound = dbound
        self.BEVDepthHead_loss = BEVDepthHead_loss
        self.BEVDepthHead_lossv2 = BEVDepthHead_lossv2

    def fuse_model(self):
        for module in self.children():
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        if self.loss is not None:
            self.loss.qconfig = None

        for module in self.children():
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

    def forward(self, data, depth_preds, preds, targets):

        # import numpy as np
        # np.save("preds.npy", preds[0])

        # gt_boxes = data["gt_boxes_batch"]
        # gt_labels = data["gt_labels_batch"]
        depth_labels = data["depth_labels_batch"]
        # targets = self.BEVDepthHead_loss(gt_boxes, gt_labels)
        detection_loss = self.BEVDepthHead_lossv2(targets, preds[0])

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_labels = depth_labels.cuda()
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = (
            depth_preds.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.depth_channels)
        )
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        
        new_pres = torch.clamp(depth_preds[fg_mask],1e-5,1.0-(1e-5))

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                new_pres,
                # depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
                ).sum()
                / max(1.0, fg_mask.sum())
            )

        output = OrderedDict()
        output["singletask_bev_loss"] = [
            bev_depth_loss_coeff * depth_loss + detection_loss
        ]
        return output

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor
        )
        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(
            B * N, H // self.downsample_factor, W // self.downsample_factor
        )

        gt_depths = (
            gt_depths - (self.dbound[0] - self.dbound[2])
        ) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1
        ).view(-1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()
