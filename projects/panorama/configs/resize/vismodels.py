from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import torch
from common import (default_calib, default_distCoeffs, input_hw, tasks,
                    val_decoders, default_ori_img_shape)
"""Current the quick way when there is bev model only"""

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

inputs = dict(img=torch.zeros((6, 3, input_hw[0], input_hw[1])))
inputsx = dict(
    img=torch.zeros((6, 3, input_hw[0], input_hw[1])),
    sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
    intrin_mats=torch.rand((1, 1, 6, 4, 4)),
    ida_mats=torch.rand((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
    bda_mat=torch.rand((1, 4, 4)),  #),
    # gt_boxes_batch = torch.rand((17, 9)),
    # gt_labels_batch = torch.rand((17)),
    # depth_labels_batch = torch.rand((1, 6, 320, 576)),
    img_metas_batch=[])
"""newly created inputs for onnx conversion added by zwj"""
inputsonnx = dict(
    img=torch.rand((6, 3, input_hw[0], input_hw[1])),
    sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
    intrin_mats=torch.rand((1, 1, 6, 4, 4)),
    ida_mats=torch.rand((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
    bda_mat=torch.rand((1, 4, 4)),
    mlp_input = torch.rand((1,1,6,27)),
    circle_map = torch.rand((1,112,16384)),
    ray_map=torch.rand((1,216,16384))
)

opt_inputs = dict(
    train=dict(),
    val=dict(
        calib=default_calib,
        distCoeffs=default_distCoeffs,
        ori_img_shape=default_ori_img_shape,
    ),
    onnx=dict(
    ),
    test=dict(),
)

traced_inputs = dict(img=torch.randn((1, 3, input_hw[0], input_hw[1])), )

def get_model(mode):
    converted_decoders = OrderedDict({
        (tuple(task_names), group): decoder
        for group, (task_names, decoder) in val_decoders.items()
    })
    return dict(
        type="MultitaskGraphModel",
        #Auto choose inputs added by ZWJ
        inputs= inputsx if "singletask_bev" in task_names and mode != "onnx" else inputsonnx if "singletask_bev" in task_names and mode == "onnx" else inputs,
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.inputs[mode]
                     for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode)
                      for T in TASK_CONFIGS},
        funnel_modules=converted_decoders if "val" in mode else None,
        flatten_outputs="val" not in mode,
        lazy_forward=False,
        __build_recursive=False,
    )


model = get_model("train")
val_model = deepcopy(get_model("val"))
test_model = deepcopy(get_model("test"))
"""newly created onnx model added by zwj"""
onnx_model = deepcopy(get_model("onnx"))