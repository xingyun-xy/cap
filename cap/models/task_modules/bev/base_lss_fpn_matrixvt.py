# Copyright (c) Changan Auto. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cap.registry import OBJECT_REGISTRY

from .bevmatrixvtcommon import DepthNet, DepthReducer, HoriConv

__all__ = ["BaseLSSFPN_matrixvt, MatrixVT"]


@OBJECT_REGISTRY.register
class BaseLSSFPN_matrixvt(nn.Module):

    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        #  img_backbone_conf,
        #  img_neck_conf,
        depth_net_conf,
        use_da=False,
    ):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(BaseLSSFPN_matrixvt, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            "voxel_size",
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]),
        )
        self.register_buffer(
            "voxel_coord",
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]),
        )
        self.register_buffer(
            "voxel_num",
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]),
        )
        self.register_buffer("frustum", self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape
        # self.img_backbone = img_backbone_conf
        # self.img_neck = img_neck_conf
        self.depth_net = self._configure_depth_net(depth_net_conf)
        # self.img_neck.init_weights()
        # self.img_backbone.init_weights()
        self.use_da = use_da
        if self.use_da:
            self.depth_aggregation_net = (
                self._configure_depth_aggregation_net())

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf["in_channels"],
            depth_net_conf["mid_channels"],
            self.output_channels,
            self.depth_channels,
        )

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4,
                2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth).view(
                    n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = (torch.arange(*self.d_bound,
                                 dtype=torch.float).view(-1, 1,
                                                         1).expand(-1, fH, fW))
        D, _, _ = d_coords.shape
        x_coords = (torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW))
        y_coords = (torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(
            1, fH, 1).expand(D, fH, fW))
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self,
                     sensor2ego_mat,
                     intrin_mat,
                     ida_mat,
                     bda_mat,
                     use_onnx=None):
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
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        if (torch.onnx.is_in_onnx_export()
                or use_onnx):  ########## replace operator
            tmp = ida_mat.view(batch_size * num_cams, 1, ida_mat.size(2),
                               ida_mat.size(3))
            points = points.view(-1, 4, 1)
            values = []
            for i in range(tmp.size(0)):
                values.append(tmp[i, :, :, :].expand(78848, 4,
                                                     4).matmul(points))
            points = torch.stack(values, dim=0).squeeze(-1)
            points = torch.cat(
                (points[:, :, :2] * points[:, :, 2:3], points[:, :, 2:]), 2)
            combine = sensor2ego_mat.matmul(intrin_mat)
            combine = combine.view(batch_size * num_cams, 1, 4, 4)
            points = points.unsqueeze(-1)
            values = []
            for i in range(tmp.size(0)):
                values.append(combine[i, :, :, :].expand(78848, 4,
                                                         4).matmul(points[i]))
            points = torch.stack(values, dim=0)
            bda_mat = (bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4))
            points = points.view(batch_size, num_cams, 112, 16, 44, 4)
        else:
            points = (ida_mat.view(batch_size, num_cams, 1, 1, 1, 4,
                                   4).inverse().matmul(points.unsqueeze(-1)))
            points = torch.cat(
                (
                    points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                    points[:, :, :, :, :, 2:],
                ),
                5,
            )
            combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
            points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                                  4).matmul(points)
            bda_mat = (bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4))
            points = (bda_mat @ points).squeeze(
                -1)  # Regular matrix multiplication
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape
        if torch.onnx.is_in_onnx_export():  ###################
            imgs = imgs.flatten().view(batch_size * num_cams, num_channels,
                                       imH, imW)
        else:
            imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                       num_channels, imH, imW)
        # img_feats = self.img_neck(self.img_backbone(imgs))[0]
        # img_feats = self.img_backbone(imgs)[0]
        if torch.onnx.is_in_onnx_export():
            return img_feats
        img_feats = img_feats.reshape(
            batch_size,
            num_sweeps,
            num_cams,
            img_feats.shape[1],
            img_feats.shape[2],
            img_feats.shape[3],
        )
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def forward(
        self,
        sweep_imgs,
        features_seconfpn,
        # mats_dict,
        sensor2ego_mats,
        intrin_mats,
        ida_mats,
        sensor2sensor_mats,
        bda_mat,
        mlp_input=None,  # onnx only else none
        circle_map=None,  # onnx only else none
        ray_map=None,  # onnx only else none
        timestamps=None,
        is_return_depth=False,
    ):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape

        # print ('img22222', sweep_imgs.shape)

        if torch.onnx.is_in_onnx_export():  ###################
            key_frame_res = self._forward_single_sweep(
                0,
                sweep_imgs,
                features_seconfpn,
                # mats_dict,
                sensor2ego_mats,
                intrin_mats,
                ida_mats,
                sensor2sensor_mats,
                bda_mat,
                mlp_input,
                circle_map,  # onnx only else none
                ray_map,  # onnx only else none
                is_return_depth=is_return_depth,
            )
            return key_frame_res

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            features_seconfpn,
            # mats_dict,
            sensor2ego_mats,
            intrin_mats,
            ida_mats,
            sensor2sensor_mats,
            bda_mat,
            mlp_input,
            circle_map,  # onnx only else none
            ray_map,  # onnx only else none
            is_return_depth=is_return_depth,
        )

        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = (key_frame_res[0]
                             if is_return_depth else key_frame_res)

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    # mats_dict,
                    sensor2ego_mats,
                    intrin_mats,
                    ida_mats,
                    sensor2sensor_mats,
                    bda_mat,
                    mlp_input,
                    circle_map,  # onnx only else none
                    ray_map,  # onnx only else none
                    is_return_depth=False,
                )
                ret_feature_list.append(feature_map)

        if is_return_depth:
            # return torch.cat(ret_feature_list, 1), key_frame_res[1]
            return (
                0.9 * ret_feature_list[0] + 0.1 * ret_feature_list[1],
                key_frame_res[1],
            )
        else:
            # return torch.cat(ret_feature_list, 1)
            return 0.9 * ret_feature_list[0] + 0.1 * ret_feature_list[1]


@OBJECT_REGISTRY.register
class MatrixVT(BaseLSSFPN_matrixvt):

    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        # img_backbone_conf,
        # img_neck_conf,
        depth_net_conf,
        use_da=False,
    ):
        """Modified from LSSFPN.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """
        super().__init__(
            x_bound,
            y_bound,
            z_bound,
            d_bound,
            final_dim,
            downsample_factor,
            output_channels,
            # img_backbone_conf,
            # img_neck_conf,
            depth_net_conf,
            use_da=False,
        )

        self.register_buffer("bev_anchors",
                             self.create_bev_anchors(x_bound, y_bound))
        self.horiconv = HoriConv(self.output_channels, 512,
                                 self.output_channels)
        self.depth_reducer = DepthReducer(self.output_channels,
                                          self.output_channels)
        self.static_mat = None

    def create_bev_anchors(self, x_bound, y_bound, ds_rate=1):
        """Create anchors in BEV space

        Args:
            x_bound (list): xbound in meters [start, end, step]
            y_bound (list): ybound in meters [start, end, step]
            ds_rate (iint, optional): downsample rate. Defaults to 1.

        Returns:
            anchors: anchors in [W, H, 2]
        """
        x_coords = ((
            torch.linspace(
                x_bound[0],
                x_bound[1] - x_bound[2] * ds_rate,
                # self.voxel_num[0] // ds_rate,
                torch.div(self.voxel_num[0], ds_rate, rounding_mode='floor'),
                dtype=torch.float,
            ) + x_bound[2] * ds_rate / 2
        ).view(
            self.voxel_num[0] // ds_rate,
            # 1).expand(self.voxel_num[0] // ds_rate,
            #           self.voxel_num[1] // ds_rate))
            1).expand(
                torch.div(self.voxel_num[0], ds_rate, rounding_mode='floor'),
                torch.div(self.voxel_num[1], ds_rate, rounding_mode='floor'),
            ))
        y_coords = ((torch.linspace(
            y_bound[0],
            y_bound[1] - y_bound[2] * ds_rate,
            self.voxel_num[1] // ds_rate,
            dtype=torch.float,
        ) + y_bound[2] * ds_rate / 2).view(
            1,
            self.voxel_num[1] // ds_rate).expand(self.voxel_num[0] // ds_rate,
                                                 self.voxel_num[1] // ds_rate))

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0)
        return anchors

    def get_proj_mat(
        self,
        #  mats_dict=None
        sensor2ego_mats,
        intrin_mats,
        ida_mats,
        bda_mat,
        use_onnx=None,
    ):  # To generate ray matrix and ring matrix
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """
        if self.static_mat is not None:
            return self.static_mat

        bev_size = int(self.voxel_num[0])  # only consider square BEV
        if torch.onnx.is_in_onnx_export() or use_onnx:
            geom_sep = self.get_geometry(sensor2ego_mats, intrin_mats,
                                         ida_mats, bda_mat,
                                         use_onnx)  # [1, 6, 112, 16, 44, 3]
        else:
            geom_sep = self.get_geometry(
                sensor2ego_mats[:, 0,
                                ...],  # mats_dict['sensor2ego_mats'][:, 0, ...],
                intrin_mats[:, 0, ...],  # mats_dict['intrin_mats'][:, 0, ...],
                ida_mats[:, 0, ...],  # mats_dict['ida_mats'][:, 0, ...],
                bda_mat,  # mats_dict.get('bda_mat', None),
            )  # [1, 6, 112, 16, 44, 3]
        geom_sep = (geom_sep - (self.voxel_coord - self.voxel_size / 2.0)
                    ) / self.voxel_size  # [1, 6, 112, 16, 44, 3]
        geom_sep = (geom_sep.mean(3).permute(0, 1, 3, 2, 4).contiguous()
                    )  # B,Ncam,W,D,2   [1, 6, 44, 112, 3]
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D,
                                        -1)[..., :2]  # [1, 264, 112, 2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0],
                                    (geom_sep < 0)[..., 1])  # [1, 264, 112]
        invalid2 = torch.logical_or(
            (geom_sep > (bev_size - 1))[..., 0],
            (geom_sep > (bev_size - 1))[..., 1],
        )  # [1, 264, 112]

        geom_uni = self.bev_anchors[None].repeat(
            [B, 1, 1, 1]
        )  # B,128,128,2 self.bev_anchors:[128,128,2], self.bev_anchors[None].shape=[1,128,128,2]
        B, L, L, _ = geom_uni.shape  # 1 128 128 _
        null_point = int(
            (bev_size / 2) *
            (bev_size + 1))  # turn every thing out of the grid into 0
        if torch.onnx.is_in_onnx_export() or use_onnx:
            geom_sep[(invalid1 | invalid2).unsqueeze(-1).expand(
                1, 264, 112, 2)] = int(bev_size / 2)
            # if True:
            ray_map = geom_uni.new_zeros(
                (Nc * W, L * L)
            )  # Ray in the paper，only W shown in paper's fig，due to the non generality，consider Nc as 1, means only 1 camera
            circle_map = geom_uni.new_zeros((D, L * L))  # Ring in the paper
            geom_idx = geom_idx.view(264, 112)
            circle_map.scatter_(
                1,
                torch.transpose(geom_idx, 0, 1),
                torch.ones((112, 264)).to(circle_map),
            )
            ray_map.scatter_(1, geom_idx, torch.ones((264, 112)).to(ray_map))
            # ray_map = ray_map.view(264, 16384) # retore dimention
            # circle_map = circle_map.view(112, 16384)
            circle_map.scatter_(
                1,
                torch.from_numpy(
                    np.array([null_point for _ in range(112)]).reshape(
                        (112, 1))).to(circle_map).long(),
                torch.zeros((112, 1)).to(circle_map),
            )
            ray_map.scatter_(
                1,
                torch.from_numpy(
                    np.array([null_point for _ in range(264)]).reshape(
                        (264, 1))).to(ray_map).long(),
                torch.zeros((264, 1)).to(ray_map),
            )
            # circle_map = circle_map.view(112, L * L)
            # ray_map = ray_map.view(264, L * L)
            pass
        else:
            geom_sep[invalid1 | invalid2] = int(
                bev_size / 2
            )  # geom_sep.shape == [1, 264, 112, 2], (invalid1 | invalid2).shape == [1, 264, 112]
            geom_idx = (
                geom_sep[..., 1] * bev_size + geom_sep[..., 0]
            )  # [B,6*44=264,D=112] * 128 + [B,6*44,112] = [B,6*44,112]
            ray_map = geom_uni.new_zeros((B, Nc * W, L * L))
            circle_map = geom_uni.new_zeros((B, D, L * L))
            for b in range(B):
                for dir in range(Nc * W):
                    ray_map[b, dir, geom_idx[b, dir]] = 1
                for d in range(D):
                    circle_map[b, d, geom_idx[b, :, d]] = 1
            circle_map[..., null_point] = 0
            ray_map[..., null_point] = 0

        return circle_map, ray_map

    @autocast(False)
    def reduce_and_project(
            self,
            feature,
            depth,
            #    mats_dict
            sensor2ego_mats,
            intrin_mats,
            ida_mats,
            bda_mat,
            circle_map=None,  # only onnx else none
            ray_map=None,  # same as above
    ):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """
        # [N,112,H,W], [N,256,H,W]
        # depth attention and operation on depth, depth：same as Categorical Depth in paper, feature：Image Feature in paper
        # feature:[B*Camera 80 16 44]
        # depth:[B*Camera 112 16 44]
        # return：[B*Camera, depth112， width44]，same as Prime Depth in paper
        depth = self.depth_reducer(
            feature, depth)  # feature [6, 80, 16, 44] depth [6, 112, 16, 44]

        B = intrin_mats.shape[0]  # mats_dict['intrin_mats'].shape[0]

        feature = self.horiconv(
            feature
        )  # PFE process, return dimention: [B*camera, each pixel feature dim:80, width44]
        if torch.onnx.is_in_onnx_export():
            depth = depth.permute(0, 2, 1).reshape(-1, self.depth_channels)
            feature = feature.permute(0, 2, 1).reshape(-1,
                                                       self.output_channels)
            """Directly imported added by ZWJ"""
            circle_map = circle_map
            ray_map = ray_map
        else:
            depth = depth.permute(0, 2, 1).reshape(
                B, -1, self.depth_channels
            )  # Prime depth changed into [B, camera*Width44, Depth112] == [1, 264, 112]
            feature = feature.permute(0, 2, 1).reshape(
                B, -1, self.output_channels
            )  # feature changed into[B, camera*Width44, each pixel feature vector dim:80] == [1, 264, 80]
            circle_map, ray_map = self.get_proj_mat(
                # mats_dict
                sensor2ego_mats,
                intrin_mats,
                ida_mats,
                bda_mat,
            )  # circle_map:[B, depth112, BEVwidth*BEVheight=128^2=16384], ray_map:[B,camera6*width44=264, BEVwidth*BEVheight=128^2=16384]


        proj_mat = depth.matmul(
            circle_map
        )  # Distance encoding [B, camera*width44, depth112] * [B, depth112, BEVW*BEVH=128^2=16384] = [B, camera6*W44=264, BEVW*BEVH=128^2=16384], ENCODING Prime Depth
        if torch.onnx.is_in_onnx_export():
            proj_mat = (proj_mat * ray_map).permute(0, 2, 1)  # [16384, 264]
        else:
            proj_mat = (proj_mat * ray_map).permute(0, 2,
                                                    1)  # Direction encoding
            # [B, camera6*width44=264, BEVwidth*BEVheight=128^2=16384] * [B,camera6*width44=264, BEVwidth*BEVheight=128^2=16384] = [B, camera6*width44=264, BEVwidth*BEVheight=128^2=16384]
            # [B, BEVwidth*BEVheight=128^2=16384, camera6*wdith44=264]
        img_feat_with_depth = proj_mat.matmul(
            feature
        )  # [B, BEVwidth*BEVheight=128^2=16384, camera6*wdith44=264] * [B, camera6*wdith44=264, each pixel feature vector dim:80] = [B, 16384, 80]

        if torch.onnx.is_in_onnx_export():
            img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
                B, -1, *self.voxel_num[:2])
        else:
            img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
                B, -1, *self.voxel_num[:2])

        return img_feat_with_depth  # [B, 80, 128, 128]

    def _forward_single_sweep(
        self,
        sweep_index,
        sweep_imgs,
        #   mats_dict,
        features_seconfpn,
        sensor2ego_mats,
        intrin_mats,
        ida_mats,
        sensor2sensor_mats,
        bda_mat,
        mlp_input=None,  # onnx only else none
        circle_map=None,  # onnx only else none
        ray_map=None,  # onnx only else none
        is_return_depth=False,
    ):

        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape

        # img_feats = self.get_cam_feats(sweep_imgs) # feature extraction return[B, num_sweeps, camera 6, 512, 16, 44]
        ###################### Testing
        if torch.onnx.is_in_onnx_export():
            # features_seconfpn = features_seconfpn.reshape((6, 768, 16, 36))
            _, _, N, C, H, W = features_seconfpn.shape
            # features_seconfpn = features_seconfpn.view(N, C, H, W)
            features_seconfpn = features_seconfpn.reshape((N, C, H, W)) # 两次squezee会增加大量耗时
            # print (features_seconfpn.shape)
            depth_feature = self.depth_net(
                features_seconfpn,
                # mats_dict,
                sensor2ego_mats,
                intrin_mats,
                ida_mats,
                # sensor2sensor_mats,
                bda_mat,
                mlp_input,
            )  # [6, 192, 16, 44]
        else:
            source_features = features_seconfpn[:, 0, ...]

            depth_feature = self.depth_net(
                source_features.reshape(
                    batch_size * num_cams,
                    source_features.shape[2],
                    source_features.shape[3],
                    source_features.shape[4],
                ),
                # mats_dict,
                sensor2ego_mats,
                intrin_mats,
                ida_mats,
                # sensor2sensor_mats,
                bda_mat,
            )  # [6, 192, 16, 44]

        with autocast(enabled=False):
            feature = depth_feature[:, self.depth_channels:(
                self.depth_channels + self.output_channels), ].float()
            depth = depth_feature[:, :self.depth_channels].float().softmax(1)

            img_feat_with_depth = self.reduce_and_project(  # MatrixVT process，return BEV feature
                feature,
                depth,
                # mats_dict
                sensor2ego_mats,
                intrin_mats,
                ida_mats,
                # sensor2sensor_mats,
                bda_mat,
                circle_map,  # onnx only else none
                ray_map,  # onnx only else none
            )  # [b*n, c, d, w]

            if is_return_depth:
                return img_feat_with_depth.contiguous(), depth
            return img_feat_with_depth.contiguous()
