import os
from collections import defaultdict
from functools import partial
import torch
from copy import deepcopy
from common import input_hw, model_setting, model_type, vis_tasks, tasks, ckpt_dir, pred_bev_batch_size
#from models import val_model, test_model, TASK_CONFIGS
#from multitask import deploy_model, deploy_inputs
#from bev import bev_test_model
from cap.utils import Config
from cap.core.proj_spec.parsing import colormap
from cap.utils.config import ConfigVersion
from cap.data.collates import collate_fn_bevdepth, collate_fn_bevdepth_cooperate_pilot, collate_fn_bevdepth_onnx
from projects.panorama.configs.datasets.changan_lmdb_infer_datasets import pilot_data_path, bev_data_path

model_name = os.getenv("CAP_PILOT_MODEL_NAME", "changan_pilot")
assert model_name is not None
device_ids = [0]
VERSION = ConfigVersion.v2

task_name = model_type
visualize_callback = dict(type="ComposeVisualize",
                          callbacks=[dict(type="BevBBoxes", )])
callbacks = [visualize_callback]
H = 900
W = 1600
final_dim = input_hw
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)
ida_aug_conf = {
    'resize_lim': (0.35, 0.37),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}
bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}
CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]
# data_root='../../../datas/changan_data/nuScenes'

#infer+eval
data_root = bev_data_path.data_root
# data_root_changan = bev_data_path.data_root_changan
# train_info_paths = os.path.join(data_root, 'nuscenes_infos_train.pkl')
test_info_paths = bev_data_path.anno_path
TestInfoJsonPath = bev_data_path.TestInfoJsonPath

# is_train = True
is_train = False
use_cbgs = False
num_sweeps = 1
# return_depth = True
return_depth = False
sweep_idxes = []
key_idxes = []
use_fusion = False

data_loader_bev = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CaBev3dDataset",
        # type="",
        # type="NuscDetDatasetCooperatePilot",
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=CLASSES,
        data_root=data_root,
        # info_paths=train_info_paths,
        info_paths=test_info_paths,
        is_train=is_train,
        use_cbgs=use_cbgs,
        num_sweeps=num_sweeps,
        img_conf=img_conf,
        return_depth=return_depth,
        sweep_idxes=sweep_idxes,
        key_idxes=key_idxes,
        use_fusion=use_fusion,
    ),
    collate_fn=partial(collate_fn_bevdepth, is_return_depth=False),
    batch_size=pred_bev_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)

deploy_input = val = dict(
    #img = torch.zeros((6, 3, 320, 576)),
    # singletask_bev = dict(img = torch.zeros((6, 3, 320, 576)),
    sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
    intrin_mats=torch.rand((1, 1, 6, 4, 4)),
    ida_mats=torch.rand((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
    bda_mat=torch.rand((1, 4, 4)),
    img_metas_batch=[]),

#  when share dataloader with CAP original tasks，two places should be changed，keep rest the same   add by zmjppre
data_loader_bev_cooperate_pilot = deepcopy(data_loader_bev)
data_loader_bev_cooperate_pilot['dataset']['type'] = "CaBev3dDataset"
data_loader_bev_cooperate_pilot['collate_fn'] = partial(collate_fn_bevdepth,
                                                        is_return_depth=False)
"""ONNX ONLY!!! DONT MISSUSE THIS!!! ADDED BY ZWJ"""
data_loader_bev_ONNX = deepcopy(data_loader_bev)
data_loader_bev_ONNX['dataset']['type'] = "CaBev3dDataset"
data_loader_bev_ONNX['collate_fn'] = partial(collate_fn_bevdepth_onnx, is_return_depth=False)


def update_state_dict_bevdepth(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("model.", "")
        # new_k = k.replace("model.backbone.img_backbone", "backbone")
        # new_k = new_k.replace("model.backbone.img_neck", "bifpn_neck")
        # new_k = new_k.replace("model.backbone.depth_net", "depthnet")
        # new_k = new_k.replace("model.head.trunk", "head")
        # new_k = new_k.replace("model.head.neck", "bev_neck")
        # new_k = new_k.replace("model.head.task_heads", "bev_head.task_heads")
        # new_k = new_k.replace("model.head.shared", "bev_head.shared")
        # print(new_k)
        new_state_dict[new_k] = v
    return new_state_dict


# with_bn_predictor=dict(
#     type="Predictor",
#     # model=test_model,
#     model=bev_test_model,
#     model_convert_pipeline=dict(
#         type="ModelConvertPipeline",
#         converters=[
#             dict(
#                 type="LoadCheckpoint",
#                 checkpoint_path="../../../codebase/bev_depth_lss_r50_256x704_128x128_24e_2key.pth",
#                 state_dict_update_func=update_state_dict_bevdepth,
#                 allow_miss=True,
#                 ignore_extra=True,
#             ),
#         ],
#     ),
#     data_loader=data_loader_bev,
#     batch_processor=dict(
#         type="BasicBatchProcessor",
#         need_grad_update=False,
#         batch_transforms=[
#             # dict(type="BgrToYuv444", rgb_input=True),
#             # dict(
#             #     type="TorchVisionAdapter",
#             #     interface="Normalize",
#             #     mean=128.0,
#             #     std=128.0,
#             # ),
#         ]
#     ),
#     callbacks=callbacks,
#     num_epochs=1,
#     share_callbacks=False,
# )

# onnx_cfg = dict(
#     export_model=deploy_model,
#     onnx_name=task_name,
#     checkpoint_path=os.path.join(
#                 ckpt_dir, "with_bn-checkpoint-last.pth.tar"
#     ),
#     dummy_input=deploy_inputs,
#     output_names=[T['name'] for T in tasks],
# )
