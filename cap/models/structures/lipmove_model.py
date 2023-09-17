# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.

import torch
from torch import nn

from cap.models.frame_utils import framewise_operation, pad_and_split
from cap.registry import OBJECT_REGISTRY

__all__ = ["LipmoveModel"]


@OBJECT_REGISTRY.register
class LipmoveModel(nn.Module):
    """
    The basic structure of Lipmove Model.

    Args:
        vfea_extractor: vfea_extractor module.
        lipmove_net: lipmove module.
        loss: Loss module.
        num_cached_frames: Time dimension window length.
        ftr_in_framewise_operation: Difference in time dimension.
        replicate_type_in_tile_and_pad_net: The splicing method
           in the time dimension.
        no_cut_seqlen_in_tile_and_padding: The cut method in the
           time dimension.
    """

    def __init__(
        self,
        vfea_extractor: nn.Module,
        lipmove_net: nn.Module,
        loss: nn.Module,
        num_cached_frames: int,
        ftr_in_framewise_operation: str,
        replicate_type_in_tile_and_pad_net: str,
        no_cut_seqlen_in_tile_and_padding: bool,
    ):
        super(LipmoveModel, self).__init__()
        self.vfea_extractor = vfea_extractor
        self.lipmove_net = lipmove_net
        self.loss = loss
        self.num_cached_frames = num_cached_frames
        self.ftr_in_framewise_operation = ftr_in_framewise_operation
        self.replicate_type_in_tile_and_pad_net = (
            replicate_type_in_tile_and_pad_net
        )
        self.no_cut_seqlen_in_tile_and_padding = (
            no_cut_seqlen_in_tile_and_padding
        )

    def pre_vision_preprocess(self, x):
        shape = x.shape
        x_batch_flatten = torch.reshape(
            x, [shape[0] * shape[1], shape[2], shape[3], shape[4]]
        )
        x = x_batch_flatten
        return x

    def pre_lipmove_preprocess(self, x, batch_size):
        num_channel = x.shape[1]
        x = torch.reshape(x, [batch_size, -1, num_channel, 1])
        x = framewise_operation(x, self.ftr_in_framewise_operation)
        x = x.permute((0, 2, 3, 1))
        x = pad_and_split(
            x,
            self.num_cached_frames,
            replicate_type=self.replicate_type_in_tile_and_pad_net,
            no_cut_seqlen=self.no_cut_seqlen_in_tile_and_padding,
        )
        return x

    def post_multimod_preprocess(self, x, batch_size):
        x = torch.reshape(x, [batch_size, -1, 2])
        return x

    def forward(self, data: dict):
        video = data["images"]
        label = data["label"]
        # mask = data['mask']
        # weight = data['weight']
        batchsize = video.shape[0]
        vx = self.pre_vision_preprocess(video)
        vx = self.vfea_extractor(vx)
        vx = self.pre_lipmove_preprocess(vx, batchsize)
        x = (
            self.lipmove_net(*vx)
            if isinstance(vx, (tuple, list))
            else self.lipmove_net(vx)
        )
        out = self.post_multimod_preprocess(x, batchsize)
        out = out.reshape((-1, 2))
        label = label.reshape((-1))
        # mask = weight.reshape((-1))
        losses = self.loss(out, label)
        # loss = loss * mask
        return losses

    def fuse_model(self):
        for module in [self.vfea_extractor, self.lipmove_net]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.vfea_extractor, self.lipmove_net]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
