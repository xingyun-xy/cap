import os, sys
import os.path as osp
from functools import partial
import torch

from cap.utils.config import ConfigVersion
from cap.data.collates import collate_fn_bevdepth

VERSION = ConfigVersion.v2
from cap.engine.processors.loss_collector import collect_loss_by_regex

# basic config
H = 900
W = 1600 
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
samples_per_gpu = 16
# bev configuration
input_size = (512, 960)
depth_channels = 112
dbound = [2.0, 58.0, 0.5]
downsample_factor = 16
TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]
# data augmentation
ida_aug_conf = {
    # 'resize_lim': (0.386, 0.55),
    "resize_lim": (0.55, 0.65), # 512x960
    # "resize_lim": (0.94, 1.25), # 896x1600
    "final_dim": input_size,
    "rot_lim": (-5.4, 5.4),
    "H": H,
    "W": W,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.0),
    "cams": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT",
        "CAM_BACK", "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
}

bda_aug_conf = {
    "rot_lim": (-22.5, 22.5),
    "scale_lim": (0.95, 1.05),
    "flip_dx_ratio": 0.5,
    "flip_dy_ratio": 0.5,
}


# dataset
ds_path = osp(os.path.dirname(__file__), f"./ca_train_datasets.py")
datapaths = Config.fromfile(ds_path).datapaths

# model
model = dict(
    type="bev_matrixvt",
    backbone=dict(
        type='VoVNet',
        norm='BN',
        name='V-99-eSE',
        input_ch=3,
        out_features=['stage2', 'stage3', 'stage4', 'stage5'],
        with_cp=False,
        __graph_model_name="backbone",
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 768, 1024],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
        __graph_model_name="second_fpn",
    ),
    depthnet=dict(
        type="MatrixVT",
        x_bound=[-51.2, 51.2, 0.8],
        y_bound=[-51.2, 51.2, 0.8],
        z_bound=[-5, 3, 8],
        d_bound=[2.0, 58.0, 0.5],
        final_dim=input_size,
        output_channels=80,
        downsample_factor=16,
        # depth_net_conf=dict(in_channels=768, mid_channels=768), # for r18
        # depth_net_conf=dict(in_channels=512, mid_channels=512),   # for r50
        depth_net_conf=dict(in_channels=512, mid_channels=512),   # for VoVNetCP
        __graph_model_name="MatrixVT",
    ),
    head=dict(
        type="BEVDepthHead",
        __graph_model_name="BEVDepthHead",
        is_train=True,
        bev_backbone_conf=dict(
            type="ResNetBevDepth",
            in_channels=80,
            depth=18,
            num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=[0, 1, 2],
            norm_eval=False,
            base_channels=160,
        ),
        bev_neck_conf=dict(
            type="SECONDFPN",
            in_channels=[80, 160, 320, 640],
            upsample_strides=[1, 2, 4, 8],
            out_channels=[64, 64, 64, 64],
        ),
        tasks=TASKS,
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            code_size=9,
        ),
        train_cfg=dict(
            point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        ),
        test_cfg=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            nms_type="circle",
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        ),
        in_channels=256,  # Equal to bev_neck output_channels.
        min_radius=2,
    ),
    depth_channels=depth_channels,
    dbound=dbound,
    downsample_factor=downsample_factor,
)

# data_loader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    sampler=None,
    drop_last=True,
    num_workers=8,
    batch_size=samples_per_gpu,
    shuffle=False,
    collate_fn=partial(collate_fn_bevdepth, is_return_depth=True),
    dataset=dict(
        type="ComposeRandomDataset",
        sample_weights=[path.sample_weight for path in ds.train_data_paths],
        datasets=[
            dict(
                # type="NuscDetDataset",
                type="CaBev3dDataset",
                ida_aug_conf=ida_aug_conf,
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                data_root=path.img_path,
                info_paths=path.anno_path,
                is_train=True,
                use_cbgs=True,
                num_sweeps=1,
                img_conf=dict(
                    img_mean=[123.675, 116.28, 103.53],
                    img_std=[58.395, 57.12, 57.375],
                    to_rgb=True,
                ),
                return_depth=True,
                sweep_idxes=[],
                key_idxes=[],
                use_fusion=False,
            ) for path in ds.train_data_paths
        ],
    ),
)



# trainer