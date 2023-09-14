from collections import OrderedDict
from copy import deepcopy
from importlib import import_module
import numpy as np 

import torch
from common import (default_calib, default_distCoeffs, default_ori_img_shape,
                    pred_batch_size, input_hw, tasks, val_decoders,
                    default_ori_img_shape)

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

inputs = dict(img=torch.zeros((6, 3, input_hw[0], input_hw[1]))) #修改输入尺寸
inputsx = dict(
    img=torch.zeros((6, 3, input_hw[0], input_hw[1])), #修改输入尺寸
    # img=torch.zeros((6, 3, 512, 960)), #修改输入尺寸
    sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
    intrin_mats=torch.rand((1, 1, 6, 4, 4)),
    ida_mats=torch.rand((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
    bda_mat=torch.rand((1, 4, 4)),  #),
    # gt_boxes_batch=torch.rand((17, 9)),
    # gt_labels_batch=torch.rand((17)),
    gt_boxes_batch=torch.from_numpy(np.load("bev_input/gt_boxes.npy", allow_pickle=True)),
    gt_labels_batch=torch.from_numpy(np.load("bev_input/gt_labels.npy", allow_pickle=True)),
    depth_labels_batch=torch.rand((1, 6, input_hw[0], input_hw[1])), #修改输入尺寸
    # depth_labels_batch=torch.rand((1, 6, 512, 960)), #修改输入尺寸
    # img_metas_batch = []
)
opt_inputs = dict(
    train=dict(),
    val=dict(
        calib=default_calib,
        distCoeffs=default_distCoeffs,
        ori_img_shape=default_ori_img_shape,
    ),
    test=dict(),
    onnx = dict()
)

traced_inputs = dict(img=torch.randn((1, 3, input_hw[0], input_hw[1])), )


def get_model(mode):
    converted_decoders = OrderedDict({
        (tuple(task_names), group): decoder
        for group, (task_names, decoder) in val_decoders.items()
    })
    return dict(
        type="MultitaskGraphModel",
        inputs=inputsx,
        # inputsx,  #[inputs if T.task_name != "singletask_bev" else inputsx for T in TASK_CONFIGS],#训练建图用
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.inputs[mode]
                     for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode)
                      for T in TASK_CONFIGS},
        funnel_modules=converted_decoders if "val" or "onnx" in mode else None,
        flatten_outputs="val" or "onnx" not in mode,
        lazy_forward=False,
        __build_recursive=False,
    )


model = get_model("train")
val_model = deepcopy(get_model("val"))
test_model = deepcopy(get_model("test"))
onnx_model = deepcopy(get_model("onnx"))
