# flake8: noqa
from copy import deepcopy

from common import backbone, input_hw

# region common config
bn_kwargs = dict(eps=1e-5, momentum=0.1)

bev_backbone = dict(
    type="ResNetBevDepth",
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(
    type="SECONDFPN",
    in_channels=[80, 160, 320, 640],
    upsample_strides=[1, 2, 4, 8],
    out_channels=[64, 64, 64, 64],
)

TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]
common_heads = dict(
    reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
)
bbox_coder = dict(
    type="CenterPointBBoxCoder",
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)
train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)
test_cfg = dict(
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
)
# endregion
# for r18
# secondfpn = dict(
#     type="SECONDFPN",
#     in_channels=[64, 64, 128, 256, 512, 512],
#     # in_channels=[32, 32, 64, 128, 256, 256],
#     upsample_strides=[0.125, 0.25, 0.5, 1, 2, 4],
#     out_channels=[128, 128, 128, 128, 128, 128],
#     __graph_model_name="second_fpn",
# )

# for r50
secondfpn = dict(
    type='SECONDFPN',
    in_channels=[256, 512, 1024, 2048],
    upsample_strides=[0.25, 0.5, 1, 2],
    out_channels=[128, 128, 128, 128],
    __graph_model_name="second_fpn",
)


# region model component config
MatrixVT = dict(
    type="MatrixVT",
    x_bound=[-51.2, 51.2, 0.8],
    y_bound=[-51.2, 51.2, 0.8],
    z_bound=[-5, 3, 8],
    d_bound=[2.0, 58.0, 0.5],
    final_dim=input_hw,
    output_channels=80,
    downsample_factor=16,
    # depth_net_conf=dict(in_channels=768, mid_channels=768), # for r18
    depth_net_conf=dict(in_channels=512, mid_channels=512),   # for r50
    __graph_model_name="MatrixVT",
)
BEVDepthHead = dict(
    type="BEVDepthHead",
    __graph_model_name="BEVDepthHead",
    bev_backbone_conf=bev_backbone,
    bev_neck_conf=bev_neck,
    tasks=TASKS,
    common_heads=common_heads,
    bbox_coder=bbox_coder,
    train_cfg=train_cfg,
    test_cfg=test_cfg,
    in_channels=256,  # Equal to bev_neck output_channels.
    min_radius=2,
)

BEVDepthHead_loss = dict(
    type="BEVDepthHead_loss",
    # __graph_model_name = "BEVDepthHead_loss",
    tasks=TASKS,
    train_cfg=train_cfg,
    in_channels=256,  # Equal to bev_neck output_channels.
)
BEVDepthHead_loss_v2 = dict(
    type="BEVDepthHead_loss_v2",
    # __graph_model_name = "BEVDepthHead_loss_v2",
    tasks=TASKS,
    train_cfg=train_cfg,
    in_channels=256,  # Equal to bev_neck output_channels.
    loss_cls=dict(type="GaussianFocalLoss_bev", reduction="mean"),
    loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
)

depth_channels = 112
dbound = [2.0, 58.0, 0.5]
downsample_factor = 16

# endregion


# region matrixVT model config
def get_model(mode):
    return dict(
        type="bev_matrixvt",
        backbone=backbone,
        neck=secondfpn,
        depthnet=MatrixVT,
        head=BEVDepthHead,
        is_train=True if mode == "train" else False,
        depth_channels=depth_channels,
        dbound=dbound,
        downsample_factor=downsample_factor,
    )


# endregion

bev_matrixvt_train_model = deepcopy(get_model("train"))
bev_matrixvt_test_model = deepcopy(get_model("test"))
