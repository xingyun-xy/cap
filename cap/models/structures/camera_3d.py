# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, Optional

import torch.nn as nn
import torch

from cap.registry import OBJECT_REGISTRY

__all__ = ["Camera3D"]


@OBJECT_REGISTRY.register
class Camera3D(nn.Module):
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
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        desc: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        convert_task_sturct = False
    ):
        super(Camera3D, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.desc = desc
        self.loss = loss
        self.postprocess = postprocess
        self.convert_task_sturct = convert_task_sturct

    def forward(self, data: Dict):

        img = data["img"]
        # N,C,H,W = data["img"].shape
        # data["img"] = data["img"].view(-1,1,N,C,H,W)
        # img =  data["img"]

        # batch_size, num_sweeps, num_cams, num_channels, imH, imW = img.shape
        # if torch.onnx.is_in_onnx_export(): ###################
        #     imgs = img.flatten().view(batch_size * num_cams, num_channels, imH, imW)    
        # else:
        #     imgs = img.flatten().view(batch_size * num_sweeps * num_cams, num_channels, imH, imW)

        features = self.backbone(img)
        # features = self.backbone(imgs)
        feat_maps = self.neck(features)
        preds = self.head(feat_maps)

        out_dict = OrderedDict()
        out_dict.update(preds)

        if self.loss is not None:
            out_dict.update(
                self.loss(
                    preds,
                    data["dimensions"],
                    data["location_offset"],
                    data["heatmap"],
                    data["heatmap_weight"],
                    data["depth"],
                    box2d_wh=data["box2d_wh"],
                    ignore_mask=data["ignore_mask"],
                    index_mask=data["index_mask"],
                    index=data["index"],
                    location=data["location"],
                    dimensions_=data["dimensions_"],
                    rotation_y=data["rotation_y"],
                )
            )
        if self.postprocess:
            # import torch
            # label_tmp = dict(
            #     calibration=torch.tensor(
            #                 # front
            #                 [
            #                     [
            #                         [1252.8131021185304, 0.0, 826.588114781398, 0.0],
            #                         [0.0, 1252.8131021185304, 469.9846626224581, 0.0],
            #                         [0.0, 0.0, 1.0, 0.0]
            #                     ]
            #                 ]

            #                 # back
            #                 # [
            #                 #     [
            #                 #         [809.2209905677063, 0.0, 829.2196003259838, 0.0],
            #                 #         [0.0, 809.2209905677063, 481.77842384512485, 0.0],
            #                 #         [0.0, 0.0, 1.0, 0.0]
            #                 #     ]
            #                 # ]
            #                 ),
            #     image_transform={"original_size": [
            #         torch.tensor([1600], dtype=torch.int64),
            #         torch.tensor([900], dtype=torch.int64)
            #     ]},
            #     dist_coffes=torch.tensor([[0,0,0,0]])
            #     )
            label_tmp = None     # postprocess里会重新构建label

            # clear out_dict, don't need 'preds'     # zmj
            out_dict.clear()
            task_name = self.desc.get('task_name','vehicle_heatmap_3d_detection')\
                if self.desc else 'vehicle_heatmap_3d_detection'
            pred = self.postprocess(
                    preds, label_tmp, data, self.convert_task_sturct, task_name
                )
            if self.convert_task_sturct:
                return pred
            else:
                out_dict.update(pred)

        return out_dict

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
