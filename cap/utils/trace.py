# Copyright (c) Changan Auto. All rights reserved.
import numpy as np
import torch
from collections import ChainMap, OrderedDict
from collections.abc import Mapping
from typing import List
from projects.panorama.configs.resize.common import input_hw
from capbc.workflow.trace import make_traceable


__all__ = [
    "combine_dict",
    "get_item_from_dict",
    "get_part_dict",
    "get_item_from_list",
    "reshape_image_with_index",
    "reshape_image_from_other",
    "get_geometry",
    "get_depth",
    "voxel_pooling_inference_ppj",
    "cat_feature",
]


@make_traceable
def combine_dict(*dict_list):
    """
    Traceable function for combining multi dict to a single dict.

    usually used for building a graph model.
    """
    return OrderedDict(ChainMap(*dict_list))


@make_traceable
def get_item_from_list(data_list, index):
    """
    Traceable function return an item from a list.

    usually used for building a graph model.
    """
    return data_list[index]


@make_traceable
def get_item_from_dict(data_dict, name):
    """
    Traceable function to return an item from a dict.

    usually used for building a graph model.
    """
    return data_dict.get(name, None)


@make_traceable
def get_part_dict(data_dict: Mapping, names: List):
    """
    Traceable function to return a subsetion of of original dict.

    usually used for building a graph model.
    """
    result = OrderedDict()
    for name in names:
        if name in data_dict:
            result[name] = data_dict[name]
    return result
@make_traceable
def reshape_image_with_index(imgs, index):
    imgs = imgs[:, index:index+1, ...]
    batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape
    imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams, num_channels, imH, imW)
    return imgs

@make_traceable
def reshape_image_from_other(imgsa, imgsb):
    imgsa = imgsa[:, 0:1, ...]
    batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgsa.shape
    imgsb = imgsb[0].reshape(batch_size, num_sweeps, num_cams, imgsb[0].shape[1], imgsb[0].shape[2], imgsb[0].shape[3])[:, 0, ...]
    return imgsb.reshape(batch_size * num_cams, imgsb.shape[2], imgsb.shape[3], imgsb.shape[4])

@make_traceable
def get_geometry(mats_dict, sweep_index):
    def create_frustum():
        """Generate frustum"""
        # make grid in image plane
        # ogfH, ogfW = self.final_dim
        # ogfH, ogfW = (256, 640)
        ogfH, ogfW = (input_hw[0], input_hw[1])
        # fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        fH, fW = ogfH // 16, ogfW // 16
        # d_coords = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        d_coords = torch.arange(*[2.0, 58.0, 0.5], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                    dtype=torch.float).view(1, fH,
                                                            1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum
    """Transfer points from camera coord to ego coord.

    Args:
        rots(Tensor): Rotation matrix from camera to ego.
        trans(Tensor): Translation matrix from camera to ego.
        intrins(Tensor): Intrinsic matrix.
        post_rots_ida(Tensor): Rotation matrix for ida.
        post_trans_ida(Tensor): Translation matrix for ida
        post_rot_bda(Tensor): Rotation matrix for bda.

    Returns:
        Tensors: points ego coord.
    """
    sensor2ego_mat = mats_dict['sensor2ego_mats'][:, sweep_index, ...]
    intrin_mat = mats_dict['intrin_mats'][:, sweep_index, ...]
    ida_mat = mats_dict['ida_mats'][:, sweep_index, ...]
    bda_mat = mats_dict.get('bda_mat', None)

    batch_size, num_cams, _, _ = sensor2ego_mat.shape

    # undo post-transformation
    # B x N x D x H x W x 3
    points = create_frustum().cuda()
    ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
    points = ida_mat.inverse().matmul(points.unsqueeze(-1))
    # cam_to_ego
    points = torch.cat(
        (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:]), 5)

    combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
    points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                            4).matmul(points)
    if bda_mat is not None:
        bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
            batch_size, num_cams, 1, 1, 1, 4, 4)
        points = (bda_mat @ points).squeeze(-1)
    else:
        points = points.squeeze(-1)
    geom_xyz = points[..., :3]

    x_bound = [-51.2, 51.2, 0.8]
    y_bound = [-51.2, 51.2, 0.8]
    z_bound = [-5, 3, 8]
    voxel_size = torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]).cuda()
    voxel_coord = torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]]).cuda()
    voxel_num = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]]).cuda()
    geom_xyz = ((geom_xyz - (voxel_coord - voxel_size / 2.0)) / voxel_size).int()

    return geom_xyz

@make_traceable
def get_depth(depth_feature):
    depth = depth_feature[:, :112].softmax(dim=1, dtype=depth_feature.dtype)
    return depth

@make_traceable
def voxel_pooling_inference_ppj(geom_xyz, depth, depth_feature):
    x_bound = [-51.2, 51.2, 0.8]
    y_bound = [-51.2, 51.2, 0.8]
    z_bound = [-5, 3, 8]
    voxel_num = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]]).cuda()
    feature_map = voxel_pooling_inference(geom_xyz, depth, depth_feature[:, 112:(112 + 80)].contiguous(), voxel_num)

    feature_map = torch.load('/code/feature_map.pt')

    return feature_map.contiguous()

@make_traceable
def cat_feature(feature_map, feature_map_v2):
    return torch.cat([feature_map, feature_map_v2], 1).float()