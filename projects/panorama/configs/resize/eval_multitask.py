import os
from collections import defaultdict
from functools import partial
from importlib import import_module

import torch
from common import (
    ckpt_dir,
    input_hw,
    model_name,
    model_setting,
    pred_batch_size,
    tasks,
    training_step,
    val_transforms,
    vis_tasks,
)
from schedule import freeze_bn_modules
from vismodels import val_model

from cap.core.data_struct.app_struct import (
    reformat_det_to_cap_eval,
    reformat_seg_to_cap_eval,
)
from cap.core.proj_spec.parsing import colormap
from cap.data.collates import (
    collate_fn_bevdepth,
    collate_fn_bevdepth_cooperate_pilot,
)
from cap.utils import Config
from projects.panorama.configs.datasets.changan_lmdb_eval_datasets import (
    bev_eval_datapath,
    dataset_ids,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
is_local_train = not os.path.exists("/running_package")
# bucket_root = "../data" if is_local_train else "/bucket/input"
bucket_root = "/data/temp" if is_local_train else "/bucket/input"

device_ids = [0, 1]
# num_processes = 2
log_rank_zero_only = False
# If eval_part_graph_model_only == True, forward only part graph model
# for diff task datasets, it usually speeds eval up
eval_part_graph_model_only = False

eval_tags = ["resize", "all_region"]

# task args
task_args = dict(
    vehicle_wheel_detection=dict(
        dump_obj_key="common_box",
    ),
    rear_detection=dict(
        dump_obj_key="vehicle",
    ),
    person_pose_classification=dict(
        dump_extra_obj_key="person_pose_classification",
    ),
    person_occlusion_classification=dict(
        dump_extra_obj_key="person_occlusion_classification"
    ),
    person_orientation_classification=dict(
        dump_extra_obj_key="person_orientation_classification"
    ),
)

task_yaml = dict(
    lane_segmentation="lane_segmentation.yaml",
    default_segmentation="default_segmentation.yaml",
    vehicle_heatmap_3d_detection="vehicle_heatmap_3d_detection.yaml",
    vehicle_detection="vehicle_detection.yaml",
)

dataset_type = dict(
    segmentation="PilotTestDatasetSimple", detection_real_3d="PilotEvalDataset"
)
# detection_real_3d="PilotTestDatasetSimple")

all_callbacks = []
all_data_loaders = []

# data
# 加载的是 changan_lmdb_eval_datasets.py

local_datapath = os.path.join(bucket_root, "eval/")

task_names = [task["name"] for task in tasks]

for eval_type, all_datasets in dataset_ids.items():
    for task_cfg, datasets in all_datasets.items():

        if isinstance(task_cfg, tuple):
            extra_task_key, task_key = task_cfg
        else:
            extra_task_key = None
            task_key = task_cfg

        # skip datasets of tasks not included in current model
        if extra_task_key is not None:
            if extra_task_key not in task_names:
                continue
        elif task_key not in task_names:
            continue

        if eval_type == "detection":
            _kwargs = {}
            if extra_task_key in task_args:
                _kwargs = task_args[extra_task_key]
            elif task_key in task_args:
                _kwargs = task_args[task_key]
            obj_key = import_module(task_key).object_type
            reformat_output_fn = reformat_det_to_cap_eval
            reformat_out_fn_kwargs = dict(
                obj_key=obj_key,
                det_task_key=task_key,
                extra_task_key=extra_task_key,
                **_kwargs,
            )

        elif eval_type == "segmentation":
            obj_key = import_module(task_key).task_name
            reformat_output_fn = reformat_seg_to_cap_eval
            reformat_out_fn_kwargs = dict(
                obj_key=obj_key,
            )

        elif eval_type == "detection_real_3d":
            obj_key = import_module(task_key).task_name
            reformat_output_fn = reformat_det_to_cap_eval
            reformat_out_fn_kwargs = dict(
                obj_key=obj_key,
            )

        for ds in datasets:
            data_path = os.path.join(local_datapath, str(ds))

            eval_callback = dict(
                type="CAPEval",
                cap_eval_dataset_id=ds,
                output_root=f"./eval_res/{ds}",
                input_yaml=f"./projects/panorama/configs/eval/{task_yaml[task_cfg]}",
                data_path=data_path,
                prediction_name=model_name,
                prediction_tags=["resize", "all_region"],
                reformat_output_fn=reformat_output_fn,
                reformat_out_fn_kwargs=reformat_out_fn_kwargs,
                cap_eval_type=eval_type,
            )

            visualize_callback = dict(
                type="ComposeVisualize",
                callbacks=[
                    dict(
                        type="DetMultitaskVisualize",
                        out_keys=[obj_key]
                        if eval_part_graph_model_only
                        else vis_tasks,
                        output_dir=f"./tmp_viz_imgs/{model_name}/{ds}",
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
                            person=dict(
                                color=(255, 0, 0),
                                thickness=2,
                            ),
                            cyclist=dict(
                                color=(255, 255, 0),
                                thickness=2,
                            ),
                            default_segmentation=dict(
                                colormap=colormap,
                                alpha=0.7,
                            ),
                            lane_segmentation=dict(
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
            )
            callbacks = [
                # visualize_callback,
                eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=5,
                ),
            ]
            all_callbacks.append(callbacks)

            data_loader = dict(
                type=torch.utils.data.DataLoader,
                sampler=dict(
                    type=torch.utils.data.DistributedSampler,
                    shuffle=False,
                ),
                dataset=dict(
                    type=dataset_type[eval_type],
                    data_path=data_path,
                    im_hw=input_hw,
                    to_rgb=False,
                    return_orig_img=True,
                ),
                batch_size=pred_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
                drop_last=False,
            )
            if eval_part_graph_model_only:
                all_data_loaders.append(
                    dict(
                        type="MultitaskLoader",
                        return_task=True,
                        loaders={obj_key: data_loader},
                    )
                )
            else:
                all_data_loaders.append(data_loader)

with_bn_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    #     "/data/temp/train_save_dir/zhijuan/multitask_v12_zhangzhijuan/",
                    #     "with_bn-checkpoint-step-50000-cae51968.pth.tar",
                    # ),
                    ckpt_dir,
                    "with_bn-checkpoint-last.pth.tar",
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=all_data_loaders,
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
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)
