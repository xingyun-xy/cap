"""
*******************************🤗IMPORTANT NOTICE🤗************************
            PLEASE DO READ THIS PART CARFULLY, BEFORE YOU RUN THE CODE!
Dataloader choice part will be modified later in a better way.
At current stage, please choose your dataloader carefully and respectively!
For Nuscences Json format, you can either use data_loader_bev or
data_loader_bev_cooperate_pilot (two choices, double the happiness)
For Changan image BEV visualization only, please use data_loader_changanbev
If you want to convert your model into ONNX format, please switch the
dataloader into data_loader_bev_ONNX
If your visualization task does not contain BEV, you can still use
data_loader from this page.
Also, if you want to visualize changan bev image, please go to
det_multitask.py and go to
def visulize() then read the temporary comments scrupulously!!!!!!
****************************************************************************
                                                        added by zwj 2023/05

*******************************🤗IMPORTANT NOTICE🤗************************
The dataloader now can be auto-selected based on your task (Nuscences based
visualization or onnx conversion)
If you need to visualize changanbev data, you should still read above note
related to changanbev data carefully.
****************************************************************************
                                                        added by zwj 2023/06

"""
import os
import sys

import torch
from common import (
    ckpt_dir,
    infer_save_prefix,
    input_hw,
    model_type,
    pred_batch_size,
    tasks,
    vis_tasks,
)

# from multitask import deploy_model, deploy_inputs
from pred_singletask_bev import (
    data_loader_bev_cooperate_pilot,
    data_loader_bev_ONNX,
)
from vismodels import onnx_model, val_model  # , test_model

from cap.core.proj_spec.parsing import colormap
from cap.utils.config import ConfigVersion
from projects.panorama.configs.datasets.changan_lmdb_infer_datasets import (
    pilot_data_path, )

from pred_changanbev import data_loader_changanbev

model_name = os.getenv("CAP_PILOT_MODEL_NAME", "changan_pilot")
assert model_name is not None
device_ids = [0]
VERSION = ConfigVersion.v2

# Fetch task file name added by ZWJ
majortask = sys.argv[0].split("/")[-1].split(".")[0]

task_name = model_type
visualize_callback = dict(
    type="ComposeVisualize",
    callbacks=[
        dict(
            type="DetMultitaskVisualize",
            out_keys=vis_tasks,
            output_dir=infer_save_prefix,  # f"{ckpt_dir}/tmp_viz_imgs/",
            vis_configs=dict(
                vehicle=dict(
                    color=(0, 255, 0),
                    thickness=2,
                    points2=dict(),
                ),
                rear=dict(
                    color=(0, 255, 255),
                    thickness=2,
                ),
                lane_segmentation=dict(
                    colormap=colormap,
                    alpha=0.5,
                ),
                default_segmentation=dict(
                    colormap=colormap,
                    alpha=0.7,
                ),
                vehicle_heatmap_3d_detection=dict(
                    color=(0, 255, 0),
                    thickness=2,
                    points2=dict(),
                    draw_bev=False,
                ),
            ),
            save_viz_imgs=True,
        ),
    ],
    bev_eval=True,  # add bev_eval flag   by zmj  评测时打开，推理时关闭
)

callbacks = [
    visualize_callback,
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    ###  Method 1  ###
    # dataset=dict(
    #     type="PilotTestDataset",
    #     # data_path=f"../data/test/pilot_test/data/",  #Infer Data Path
    #     # data_path=f"/data/test/pilot_test/nuscenes_test/test_cam_front/",
    #     data_path=f"/data/test/pilot_test/nuscenes_test/test/",
    #     # data_path=f"/data/test/pilot_test/nuscenes_test/test_one_img/",
    #     # data_path=f"/data/test/pilot_test/nuscenes_test/nuscenes_all/train",
    #     # data_path=f"/data/test/pilot_test/nuscenes_test/val/",
    #     im_hw=input_hw,
    #     to_rgb=False,
    #     return_orig_img=True,
    # ),
    ###  Method 2  ###
    dataset=dict(
        type="PilotTestDatasetSimple",
        # data_path=f"../../../datas/test/pilot_test/nuscenes_test/inference_imgs/",
        data_path=pilot_data_path,
        # data_path=f"/data/test/pilot_test/changan_data/20230425/",
        im_hw=input_hw,
        to_rgb=False,
        return_orig_img=True,
    ),
    batch_size=pred_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)

with_bn_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    "r50_256_576_12wtrain_6.5wstep.pth.tar",
                    # './train_res_ca_256_576/bev_batch1_320_576_202305_zxx/with_bn-checkpoint-last.pth.tar',
                    # '../data/train_save_dir/zhijuan/bev_fp16_v35_zhangzhijuan/with_bn-checkpoint-last.pth.tar',
                    
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    # auto-select data loder based on major task whether it is ONNX
    # conversion or normal visualization added by ZWJ
    data_loader=data_loader_bev_ONNX
    if majortask == "pytorch2onnx" else data_loader_bev_cooperate_pilot,
    # data_loader = data_loader_changanbev,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        batch_transforms=[
            dict(type="BgrToYuv444", rgb_input=True),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)

onnx_inputs = dict(
    img=torch.rand((6, 3, input_hw[0], input_hw[1])),
    sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
    intrin_mats=torch.rand((1, 1, 6, 4, 4)),
    ida_mats=torch.rand((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
    bda_mat=torch.rand((1, 4, 4)),
    mlp_input=torch.rand((1, 1, 6, 27)),
    circle_map=torch.rand((1, 112, 16384)),
    ray_map=torch.rand((1, 6*input_hw[1]//16, 16384)),
)
onnx_cfg = dict(
    export_model=onnx_model,
    onnx_name=task_name,
    checkpoint_path=os.path.join(
        "./256_576_nuscenes.pth.tar",
        # "./train_res_ca_256_576/bev_batch1_320_576_202305_zxx/with_bn-checkpoint-last.pth.tar",
        # 1660524174059188225_pengju_minloss_premodel_zwj/panorama_multitask_resize/",
        # "with_bn-checkpoint-step-22999-f3cb349d.pth.tar",
    ),
    dummy_input=onnx_inputs,
    output_names=[T["name"] for T in tasks],
)
