import json
import os
from copy import deepcopy
from pathlib import Path

import torch

from cap.utils.config import Config

# env variables
num_machines = int(os.environ.get("CAP_NUM_MACHINES", "1"))
training_step = os.environ.get("CAP_TRAINING_STEP", "with_bn")
model_setting = os.getenv("CAP_PILOT_MODEL_SETTING", "changan_lmdb")
model_version = os.getenv("CAP_PILOT_MODEL_VERSION", "test")
model_thresh = os.getenv("CAP_PILOT_MODEL_THRESH", None)
tasks = os.getenv("CAP_PILOT_TASKS")

# model info
# model_type = "256_576_r50"
model_type = "256_576_vov99"
model_name = "_".join([model_type, model_setting, model_version])
if model_thresh is not None:
    model_thresh = json.loads(model_thresh)

if tasks is not None:
    tasks = json.loads(tasks)
else:
    # tasks
    semanticparsing_tasks = [
        dict(name="default_segmentation", important=True),
    ]

    laneparsing_tasks = [
        dict(name="lane_segmentation", important=True),
    ]

    cam_3d_tasks = [
        dict(name="vehicle_heatmap_3d_detection", important=True),
        # dict(name="ped_cyc_heatmap_3d_detection", important=True),
        # dict(name="static_objects_heatmap_3d_detection", important=True),
    ]
    bev_depth = [
        dict(name="singletask_bev", important=True),
        # dict(name="ped_cyc_heatmap_3d_detection", important=True),
        # dict(name="static_objects_heatmap_3d_detection", important=True),
    ]

    # tasks = (cam_3d_tasks + laneparsing_tasks + semanticparsing_tasks + bev_depth)
    # tasks = (laneparsing_tasks + semanticparsing_tasks + bev_depth)
    # tasks = (cam_3d_tasks + bev_depth) #推理BEV+real3D
    # tasks = laneparsing_tasks + semanticparsing_tasks  # 推理两分割
    # tasks = (cam_3d_tasks)
    # tasks =(laneparsing_tasks)
    # tasks =(semanticparsing_tasks)
    tasks = (bev_depth)

# dataset
ds_path = os.path.join(
    os.path.dirname(__file__),
    f"../datasets/{model_setting.lower()}_train_datasets.py",
)
if os.path.exists(ds_path):
    datapaths = Config.fromfile(ds_path).datapaths

is_local_train = not os.path.exists("/running_package")

if is_local_train:
    batch_size = 1  #设置batchsize
    bev_batch_size = 1  # refer to n samples
    bev_depth_loss_coeff = 3  # 始终为3，不需要再修改
    log_freq = 5
    # train_save_prefix = "/tmp/model"  #云平台训练保存路径
    train_save_prefix = "/tmp/model"  #本地训练保存路径
    infer_save_prefix = "/tmp/model"  #本地保存推理
    # infer_save_prefix = "/tmp/model"
    # save_prefix = "/code/cap_train_results/panorama/20230421_from_zzj"  #训练保存路径
    # save_prefix = "/code/cap_train_results/panorama/20230425_from_zzj"  #训练保存路径
else:
    batch_size = 1
    bev_batch_size = 1 # refer to n samples
    bev_depth_loss_coeff = 0.0  #默认3 降到 0.0
    log_freq = 25
    train_save_prefix = "/tmp/model"  # 训练保存路径
    infer_save_prefix = "/tmp/model"

ckpt_dir = Path(train_save_prefix) / model_type

pred_bev_batch_size = 8
pred_batch_size = 6
for task in tasks:
    if task["name"] == "vehicle_heatmap_3d_detection":
        pred_batch_size = 1

lmdb_data = "lmdb" in model_setting.lower()

input_hw = resize_hw = (256, 576)  # 修改输入尺寸
# input_hw = resize_hw = (512, 960)  # 修改输入尺寸
default_ori_img_shape = torch.tensor([[900, 1600, 3]] * pred_batch_size)
roi_region = (0, 0, input_hw[1], input_hw[0])
vanishing_point = (int(input_hw[1] / 2), int(input_hw[0] / 2))

bn_kwargs = dict(eps=1e-5, momentum=0.1)

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1

# 3d config
# undistort_depth_uv = True
undistort_depth_uv = False

# default_calib = torch.tensor(
#     [
#         [
#             [1114.34668, 0, 978.904541, 0],
#             [0, 1114.34668, 670.611328, 0],
#             [0, 0, 1, 0],
#         ]
#     ]
#     * pred_batch_size
# )
# default_distCoeffs = torch.tensor(
#     [
#         [
#             -0.586143374,
#             0.221308455,
#             1.17504729e-04,
#             1.71026128e-04,
#             7.10545704e-02,
#             -0.186895579,
#             -9.31752697e-02,
#             0.222488135,
#         ]
#     ]
#     * pred_batch_sizep
# )
default_calib = torch.tensor([[
    [1252.8131021185304, 0.0, 826.588114781398, 0.0],
    [0.0, 1252.8131021185304, 469.9846626224581, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]] * pred_batch_size)

default_distCoeffs = torch.tensor([[
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]] * pred_batch_size)
default_ori_img_shape = torch.tensor([[900, 1600, 3]] * pred_batch_size)

# default_ori_img_shape = torch.tensor([[900, 1600, 3]] * pred_batch_size)


# roi configs
def get_det_rpn_out_keys(mode):
    if "train" in mode:
        return None
    elif "val" in mode:
        return []
    elif "test" in mode:
        return ["pred_boxes"]
    else:
        raise KeyError(mode)


test_roi_num = 100


# backbone
# vovnet backbone
# 对应的out channels = [256, 512, 768, 1024]
# need a neck
# 同时要修改Vovnet代码，把forward的输出从dict改成list

# build_vovnet_backbone = {
#     'type': 'VoVNet',
#     'norm': 'BN',
#     'name': 'V-99-eSE',
#     'input_ch': 3,
#     'out_features': ['stage2', 'stage3', 'stage4', 'stage5'],
#     'with_cp': False,
# }
    
# # vovnet with fpn with p6
# backbone = dict(
#     type='VoVNetFPN',
#     bottom_up_config=build_vovnet_backbone,
#     in_features=['stage2', 'stage3', 'stage4', 'stage5'],
#     out_channels=256,
#     norm='BN',
#     top_block={'type': 'LastLevelP6',
#                   'in_channels_top': 256,
#                   'out_channels': 256,
#                   'in_features': 'p5',
#                   },
#     fuse_type='sum',
#     size_divisibility_mul_2=True,
#     checkpoint='/root/cap-xy/depth_pretrained_v99.pth',
#     __graph_model_name="backbone",
# )

backbone = dict(
    type='VoVNetCP',
    spec_name='V-99-eSE',
    norm_eval=True,
    frozen_stages=-1,
    input_ch=3,
    out_features=('stage2', 'stage3', 'stage4', 'stage5'),
    pretrained='/root/cap-xy/fcos3d_vovnet_imgbackbone-remapped.pth',
    __graph_model_name="backbone",
)

# fpn_neck=dict(
#     type='CPFPN',  ### remove unused parameters 
#     in_channels=[768, 1024],
#     out_channels=256,
#     num_outs=2,
#     norm_cfg=bn_kwargs,
#     __graph_model_name="fpn_neck",
# )

# backbone = dict(
#     type="ResNetBevDepth",
#     depth=50,
#     in_channels=3,
#     stem_channels=None,
#     base_channels=64,
#     num_stages=4,
#     strides=(1, 2, 2, 2),
#     dilations=(1, 1, 1, 1),
#     out_indices=[0, 1, 2, 3],
#     style='pytorch',
#     deep_stem=False,
#     avg_down=False,
#     frozen_stages=0,
#     conv_cfg=None,
#     norm_eval=False,
#     dcn=None,
#     stage_with_dcn=(False, False, False, False),
#     plugins=None,
#     with_cp=False,
#     zero_init_residual=True,
#     pretrained=None,
#     __graph_model_name="backbone",
# )


# fpn_neck = dict(
#     type="FPN",
#     in_strides=[2, 4, 8, 16, 32, 64],
#     in_channels=[64, 64, 128, 256, 512, 512],
#     out_strides=[4, 8, 16, 32, 64],
#     out_channels=[16, 32, 64, 128, 128],
#     bn_kwargs=bn_kwargs,
#     __graph_model_name="fpn_neck",
# )


fix_channel_neck = dict(
    type="FixChannelNeck",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_strides=[8, 16, 32, 64],
    out_channel=64,
    bn_kwargs=bn_kwargs,
    __graph_model_name="fix_channel_neck",
)

ufpn_seg_neck = dict(
    type="UFPN",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    group_base=1,
    __graph_model_name="ufpn_seg_neck",
)

ufpn_3d_neck = deepcopy(ufpn_seg_neck)
ufpn_3d_neck.update(
    dict(
        output_strides=[4],
        __graph_model_name="ufpn_3d_neck",
    ))

val_decoders = {}

val_transforms = [
    # dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

vis_tasks = []
