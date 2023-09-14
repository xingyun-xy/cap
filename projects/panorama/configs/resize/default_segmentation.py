# flake8: noqa

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
    ufpn_seg_neck,
    val_transforms,
    vanishing_point,
    vis_tasks,
)

from cap.callbacks.metric_updater import update_metric_using_regex
from cap.core.proj_spec.descs import get_default_parsing_desc

task_name = "default_segmentation"
vis_tasks.append(task_name)

# model
head_out_strides = [2, 8, 16, 32, 64]
num_classes = 21


def get_model(mode):
    out_strides = (
        head_out_strides if mode == "train" else [min(head_out_strides)]
    )

    return dict(
        type="Segmentor",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, ufpn_seg_neck]),
        head=dict(
            type="FRCNNSegHead",
            group_base=1,
            in_strides=[4, 8, 16, 32, 64],
            in_channels=[16, 32, 64, 128, 128],
            out_strides=out_strides,
            out_channels=[num_classes] * len(out_strides),
            bn_kwargs=bn_kwargs,
            argmax_output=mode != "train",
            dequant_output=mode == "train",
            with_extra_conv=False,
            __graph_model_name=f"{task_name}_head",
        ),
        loss=dict(
            type="MultiStrideLosses",
            num_classes=num_classes,
            out_strides=head_out_strides,
            loss=dict(
                type="WeightedSquaredHingeLoss",
                reduction="mean",
                weight_low_thr=0.6,
                weight_high_thr=1.0,
                hard_neg_mining_cfg=dict(
                    keep_pos=True,
                    neg_ratio=0.999,
                    hard_ratio=1.0,
                    min_keep_num=255,
                ),
            ),
            loss_weights=[4, 2, 2, 2, 2],
            __graph_model_name=f"{task_name}_loss",
        )
        if mode == "train"
        else None,
        desc=dict(
            type="AddDesc",
            per_tensor_desc=get_default_parsing_desc(
                desc_id=f"us_{num_classes}",
                roi_regions=roi_region,
                vanishing_point=vanishing_point,
            ),
            __graph_model_name=f"{task_name}_desc",
        )
        if mode is not "train"
        else None,
        postprocess=dict(
            type="VargNetSegDecoder",
            out_strides=out_strides,
            transforms=val_transforms,
            __graph_model_name=f"{task_name}_decoder",
        )
        if "val" in mode
        else None,
    )


# inputs
inputs = dict(
    train=dict(
        labels=[
            torch.zeros((6, 1, resize_hw[0] // s, resize_hw[1] // s))
            for s in head_out_strides
        ],
    ),
    val=dict(),
    test=dict(),
    onnx=dict()
)


loss_names = [f"stride_{s}_loss" for s in head_out_strides]

metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_{name}$",
            )
            for name in loss_names
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


ds = datapaths.default_parsing
data_loader = dict(
    type=torch.utils.data.DataLoader,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=True,
    num_workers=0,
    batch_size=batch_size,
    dataset=dict(
        type="ComposeRandomDataset",
        sample_weights=[
            path.sample_weight for path in ds.train_data_paths
        ],
        datasets=[
            dict(
                type="DetSeg2DAnnoDataset",
                idx_path=path.idx_path,
                img_path=path.img_path,
                anno_path=path.anno_path,
                transforms=[
                    dict(
                        type="SemanticSegAffineAugTransformerEx",
                        target_wh=input_hw[::-1],
                        inter_method=10,
                        label_scales=[
                            1.0 / stride_i for stride_i in head_out_strides
                        ],
                        use_pyramid=True,
                        pyramid_min_step=0.7,
                        pyramid_max_step=0.8,
                        flip_prob=0.5,
                        label_padding_value=-1,
                        rand_translation_ratio=0.0,
                        center_aligned=False,
                        rand_scale_range=(0.8, 1.3),
                        resize_wh=resize_hw[::-1],
                        adapt_diff_resolution=False,
                    ),
                ],
            )
            for path in ds.train_data_paths
        ],
    ),
)