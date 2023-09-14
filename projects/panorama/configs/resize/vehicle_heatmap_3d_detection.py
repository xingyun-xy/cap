# flake8: noqa

from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    datapaths,
    fpn_neck,
    input_hw,
    lmdb_data,
    log_freq,
    resize_hw,
    roi_region,
    ufpn_3d_neck,
    undistort_depth_uv,
    vanishing_point,
    vis_tasks,
    default_calib,
    default_distCoeffs,
    default_ori_img_shape,
)

from cap.callbacks.metric_updater import update_metric_using_regex
from cap.core.proj_spec.descs import frcnn_headmap_3d_detection_desc

task_name = "vehicle_heatmap_3d_detection"
vis_tasks.append(task_name)

classnames = ["vehicle"]
# 4cls
# classnames = ["vehicle", "person", "cyclist", "traffic_cone"]
num_classes = len(classnames)
classname2idxs = [i for i in range(1, num_classes + 1)]
flatten_classnames = reduce(lambda x, y: x + "_" + y, classnames)

# input
input_wh = input_hw[::-1]
down_stride = 4
keep_res = False  # whether use original image size
normalize_depth = True
# focal_length_default = 1114.3466796875
focal_length_default = 1252.8131021185304  # nuscenes CAM_FRONT
# ----- Seperate line -----

# model
max_depth = 50
max_objs = 100
use_depth_rotation_multiscale = False
output_head_with_2d_wh = True

# class_id map to new class_id
classid_map = {1: -1, 2: 0, 3: -1, 4: 0, 5: 0, 6: 0, 7: 0, 8: -1}
# classid_map = {1: 1, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 3}
# head
rot_channel = 2
if undistort_depth_uv:
    outputs = OrderedDict(
        hm=num_classes,
        dep_u=1,
        dep_v=1,
        rot=rot_channel,
        dim=3,
        loc_offset=2,
    )
    outputs_prefix = OrderedDict(
        hm="vehicle_hm",
        dep_u="dep_u",
        dep_v="dep_v",
        rot="rot",
        dim="dim",
        loc_offset="loc_offset",
        wh="wh",
    )
else:
    outputs = OrderedDict(
        hm=num_classes,
        dep=1,
        rot=rot_channel,
        dim=3,
        loc_offset=2,
    )
    outputs_prefix = OrderedDict(
        hm="vehicle_hm",
        dep="dep",
        rot="rot",
        dim="dim",
        loc_offset="loc_offset",
        wh="wh",
    )

if output_head_with_2d_wh:
    outputs["wh"] = 2

output_cfg = {
    out: {
        "out_channels": ch,
        "out_conv_channels": 32,
        "prefix": outputs_prefix[out],
    }
    for out, ch in outputs.items()
}


def get_model(mode):
    return dict(
        type="Camera3D",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, ufpn_3d_neck]),
        head=dict(
            type="ExtSequential",
            modules=[
                dict(
                    type="Camera3DHead",
                    output_cfg=output_cfg,
                    bn_kwargs=bn_kwargs,
                    in_strides=[4],
                    in_channels=[16],
                    out_stride=4,
                    __graph_model_name=f"{task_name}_head",
                ),
                dict(
                    type="AddDesc",
                    per_tensor_desc=frcnn_headmap_3d_detection_desc(
                        task_name="frcnn_camera_3d_detection",
                        classnames=classnames,
                        undistort_2dcenter=False,
                        undistort_depth_uv=undistort_depth_uv,
                        use_multibin=False,
                        score_threshold=0.48,  # score for software
                        roi_regions=roi_region,
                        vanishing_point=vanishing_point,
                        focal_length_default=focal_length_default,
                        output_head_with_2d_wh=output_head_with_2d_wh,
                    ),
                    __graph_model_name=f"{task_name}_desc",
                ),
            ],
        ),
        desc=dict(task_name=task_name),
        loss=dict(
            type="Camera3DLoss",
            hm_loss=dict(type="HMFocalLoss"),
            box2d_wh_loss=dict(type="HML1Loss", heatmap_type="weighted"),
            dimensions_loss=dict(type="HML1Loss", heatmap_type="weighted"),
            location_offset_loss=dict(type="HML1Loss",
                                      heatmap_type="weighted"),
            depth_loss=dict(type="HML1Loss", heatmap_type="weighted"),
            loss_weights=dict(
                heatmap=0.5,
                box2d_wh=0.01,
                depth=0.5,
                dimensions=0.5,
                rotation=2,
                location_offset=0.5,
            ),
            use_depth_rotation_multiscale=use_depth_rotation_multiscale,
            undistort_depth_uv=undistort_depth_uv,
            output_head_with_2d_wh=output_head_with_2d_wh,
            max_depth=max_depth,
            __graph_model_name=f"{task_name}_loss",
        ) if mode == "train" else None,
        postprocess=dict(
            type="Real3DDecoder",
            focal_length_default=focal_length_default,
            topk=40,
            nms_kwargs=dict(iou_threshold=0.7, replace=True),
            score_thres=0.2,
            __graph_model_name=f"{task_name}_real3d_decoder",
        ) if "val" in mode else None,
        convert_task_sturct=True,
    )


# inputs
inputs = dict(
    train=dict(
        heatmap=torch.zeros(
            (1, num_classes, resize_hw[0] // 4, resize_hw[1] // 4)),
        box2d_wh=torch.zeros((1, 2, resize_hw[0] // 4, resize_hw[1] // 4)),
        dimensions=torch.zeros((1, 3, resize_hw[0] // 4, resize_hw[1] // 4)),
        location_offset=torch.zeros(
            (1, 2, resize_hw[0] // 4, resize_hw[1] // 4)),
        depth=torch.zeros(
            (1, 2, resize_hw[0] // 4,
             resize_hw[1] // 4)) if undistort_depth_uv else torch.zeros(
                 (1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
        heatmap_weight=torch.zeros(
            (1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
        ignore_mask=torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
        index=torch.zeros((1, 100)),
        index_mask=torch.zeros((1, 100)),
        location=torch.zeros((1, 100, 3)),
        rotation_y=torch.zeros((1, 100, 1)),
        dimensions_=torch.zeros((1, 100, 3)),
    ),
    val=dict(
        # sensor2ego_mats = torch.rand((1, 1, 6, 4, 4)),
        # intrin_mats = torch.rand((1, 1, 6, 4, 4)),
        # ida_mats = torch.rand((1, 1, 6, 4, 4)),
        # sensor2sensor_mats = torch.rand((1, 1, 6, 4, 4)),
        # bda_mat = torch.rand((1, 4, 4)),
        # img_metas_batch = [],
        calib=torch.zeros((1, 3, 4)),
        distCoeffs=torch.zeros((1, 8)),
        ori_img_shape=(3, 900, 1600),
    ),
    test=dict(
        calib=torch.zeros((1, 3, 4)),
        distCoeffs=torch.zeros((1, 8)),
        ori_img_shape=(3, 900, 1600),
    ),
    #onnx input addded by zwj
    onnx=dict(
        calib=torch.zeros((1, 3, 4)),
        distCoeffs=torch.zeros((1, 8)),
        ori_img_shape=(3, 900, 1600),
    ),
)

loss_names = [
    "hm_loss",
    "rot_loss",
    "dep_loss",
    "wh_loss",
    "dim_loss",
    "loc_offset_loss",
]

metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_{name}$",
            ) for name in loss_names
        ]),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

ds = datapaths.vehicle_3d_detection
data_loader = dict(
    type=torch.utils.data.DataLoader,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=True,
    num_workers=0,
    batch_size=batch_size,
    dataset=dict(
        type="ComposeRandomDataset",
        sample_weights=[path.sample_weight for path in ds.train_data_paths],
        datasets=[
            dict(
                type="DetSeg2DAnnoDataset",
                idx_path=path.idx_path,
                img_path=path.img_path,
                anno_path=path.anno_path,
                transforms=[
                    dict(
                        type="Image3DTransform",
                        input_wh=input_wh,
                        keep_res=keep_res,
                        shift=np.array([0, 0], dtype=np.float32),
                        keep_aspect_ratio=False,
                    ),
                    dict(
                        type="Heatmap3DDetectionLableGenerate",
                        num_classes=num_classes,
                        classid_map=classid_map,
                        normalize_depth=normalize_depth,
                        focal_length_default=focal_length_default,
                        alpha_in_degree=False,
                        down_stride=down_stride,
                        use_bbox2d=True,
                        # use_project_bbox2d=False,
                        use_project_bbox2d=True,
                        enable_ignore_area=True,
                        shift=np.array([0, 0], dtype=np.float32),
                        filtered_name="__front_0820__",
                        min_box_edge=8,  # original image size
                        max_depth=max_depth,
                        max_objs=max_objs,
                        undistort_2dcenter=False,
                        undistort_depth_uv=undistort_depth_uv,
                        # keep_meta_keys=['calib', 'img_wh', 'file_name', 'distCoeffs', 'orgin_wh', 'ori_img']  # InputsVisualize需要的一些GT
                    ),
                ],
            ) for path in ds.train_data_paths
        ],
    ),
)