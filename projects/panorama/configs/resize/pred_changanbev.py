import os
from collections import defaultdict
from functools import partial
import torch
from common import input_hw, model_setting, model_type, vis_tasks, tasks, ckpt_dir
from vismodels import val_model, test_model, TASK_CONFIGS
#from multitask import deploy_model, deploy_inputs
#from bev import bev_test_model

from cap.core.proj_spec.parsing import colormap
from cap.utils.config import ConfigVersion
from cap.data.collates import collate_fn_changanbev

model_name = os.getenv("CAP_PILOT_MODEL_NAME", "changan_pilot")
assert model_name is not None
device_ids = [0]
VERSION = ConfigVersion.v2

task_name = model_type
visualize_callback = dict(type="ComposeVisualize",
                          callbacks=[dict(type="BevBBoxes", )])
callbacks = [visualize_callback]
# H = 900
# W = 1600
# final_dim = (320, 576)
# img_conf = dict(img_mean=[123.675, 116.28, 103.53],
#                 img_std=[58.395, 57.12, 57.375],
#                 to_rgb=True)
changan_pre_process_infos = {
        "train_flag":False, # 训练模式
        'rot_lim'   : (-5.4, 5.4), # 旋转度数 非弧度
        'rand_flip' : True, # 是否水平翻转
        'FLC_FRC_RLC_RRC_RC_TOP_DOWN': (0.00, 0.10), # 前左 前右 后左 后 后右 上下的crop起始点范围 
        'FC_TOP_DOWN'                : (0.00, 0.01), # 前视 上下的crop起始点范围 
        'RLC_RRC_DROP'               : (0.10, 0.11), # 后左和后右因为需要把自车部分crop掉，这个是左或者右需要丢掉的范围
        
        'final_dim' : (320, 576),  # 非后视最终输入网络的参数
        'FC_H'       : 2160,        # 前视camera的高
        'FC_W'       : 3840,        # 前视camera的宽
        'NOT_FC_H'   : 1536,        # 非前视camera的高
        'NOT_FC_W'   : 1920,        # 非前视camera的宽
        'cams'       : ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'], # camera信息 实际暂没有用到 数据的处理顺序是按照这个来的
        'Ncams'      : 6, # camera个数
        "img_conf"   : dict(img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=False),# 图像均值、标准差以及是否转通道顺序
        "CLASS"      :['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian','traffic_cone',], # 检测的类别
        "data_root"  : "changan_car/500111/camera",# 数据的路径
        "visual_imgs": False, #是否可视化的标志 会把图像保存在本地
        "visual_save_path" : 'changan_visual_path', # 如果可视化，则保存在这里
        "return_undistort_imgs":True # 返回输入网络之前的图像可视化，用于端到端可视化
        }

# val_dataset = changanbevdataset(changan_pre_process_infos)
# val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=1,
#         sampler=None,
#     )
data_loader_changanbev = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        # type="NuscDetDataset",
        type="changanbevdataset",
        pre_process_info = changan_pre_process_infos
    ),
        collate_fn=partial(collate_fn_changanbev,
                       is_return_depth=False),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=None,
)

