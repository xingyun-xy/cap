import os
import sys
from importlib import import_module

import torch
from common import (
    ckpt_dir,
    is_local_train,
    log_freq,
    num_machines,
    tasks,
    training_step,
    input_hw
)
from data_loaders import data_loader
from models import model, test_model
from schedule import (
    base_lr,
    interval_by,
    num_steps,
    save_interval,
    warmup_steps,
)

from cap.engine.processors.loss_collector import collect_loss_by_regex
from cap.utils.config import ConfigVersion

VERSION = ConfigVersion.v2

# step-specific args
num_steps = num_steps[training_step]
base_lr = base_lr[training_step]

# -------------------------- multitask --------------------------
seed = None  # random seed
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = True

device_ids = [0, 1, 2, 3]

# ckpt_dir = Path(save_prefix) / model_type
log_dir = ckpt_dir / "logs"
redirect_config_logging_path = (log_dir /
                                f"config_log_{training_step}.log").as_posix()

# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]
# -------------------------- batch processor --------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    # enable_apex=True,
    enable_amp=enable_amp,
    grad_scaler=dict(
        type=torch.cuda.amp.GradScaler,
        growth_interval=200,
    ),
)

deploy_model = (
    test_model  # if "bev" not in task_names else bev_matrixvt_test_model
)
# onnx_model = onnx_model
deploy_inputs = dict(
    img=torch.rand((6, 3, input_hw[0], input_hw[1])),  # 导onnx需要 #修改输入尺寸
    # img = torch.rand((6, 3, 512, 960)),  #导onnx需要 #修改输入尺寸
    sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
    intrin_mats=torch.rand((1, 1, 6, 4, 4)),
    ida_mats=torch.rand((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
    bda_mat=torch.rand((1, 4, 4)),
)
# mlp_input = torch.rand((1,1,6,27)),
# circle_map = torch.rand((1,112,16384)),
# ray_map=torch.rand((1,216,16384)))
# -------------------------- callbacks --------------------------
stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="PolyLrUpdater",
    save_dir=log_dir.as_posix() if is_local_train else "/job_tboard/",
    max_update=num_steps // num_machines,
    power=1.0,
    warmup_len=warmup_steps,
    step_log_interval=25,
)

checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir.as_posix(),
    name_prefix=training_step + "-",
    deploy_model=deploy_model,
    deploy_inputs=deploy_inputs,
    strict_match=False,
    best_refer_metric=None,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir.as_posix() if is_local_train else "/job_tboard/",  # noqa
    update_freq=log_freq,
    tb_update_funcs=[
        T.tb_update_func for T in TASK_CONFIGS
        if getattr(T, "tb_update_func", None) is not None
    ],
    model=None,  # 如果不显示梯度变化，设为None ---add by zxx
    # model =model, # 训练时tensorboard显示梯度更新，但训练速度会严重降低
)

metric_updaters = [T.metric_updater for T in TASK_CONFIGS]

# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
] + metric_updaters

profiler = dict(type="SimpleProfiler",
                # dirpath=log_dir.as_posix(),
                # filename='profile.log',
                )


def update_state_dict_bevdepth(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("model.backbone.img_backbone", "backbone")
        new_k = new_k.replace("model.backbone.img_neck", "second_fpn")
        new_k = new_k.replace("model.backbone", "MatrixVT")
        new_k = new_k.replace("model.head.task_heads",
                              "BEVDepthHead.task_heads")
        new_k = new_k.replace("model.head.shared_conv",
                              "BEVDepthHead.shared_conv")
        new_k = new_k.replace("model.head.trunk", "BEVDepthHead.trunk")
        new_k = new_k.replace("model.head.neck", "BEVDepthHead.neck")

        new_state_dict[new_k] = v
    return new_state_dict


def update_state_dict_bevdepth_v2(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = "backbone." + k
        new_state_dict[new_k] = v
    return new_state_dict


# -------------------------- float trainer --------------------------
with_bn_trainer = dict(
    type="DistributedDataParallelTrainer", 
    model=model, #if "bev" not in task_names else bev_matrixvt_train_model,                        
    resume_epoch_or_step = False,            #resume 
    resume_optimizer = False,                #resume 
    model_convert_pipeline=dict(            #resume
        type="ModelConvertPipeline",
        qat_mode="with_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                allow_miss=True,
                ignore_extra=True, # TODO
                # checkpoint_path=os.path.join(
                # #     # "../pth/loss_clip_gridmask_hsv/with_bn-checkpoint-step-60999-d047b463.pth.tar"
                # #     # ckpt_dir, "with_bn-checkpoint-step-22999-f3cb349d.pth.tar" #预训练模型
                #     "/root/cap-xy/ckpts/official/fcos3d_vovnet_imgbackbone-remapped.pth"  #当前最优BEV预训练模型
                # ),
            ),
        ],
    ),
    data_loader=data_loader,              
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-5)},
        lr=base_lr * num_machines,  # 机器多学习率会相应变大
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(num_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
    # profiler=profiler,
    find_unused_parameters=True,  # DDP设置提高性能
)

if is_local_train and len(device_ids) == 1:
    with_bn_trainer.pop("sync_bn")
    with_bn_trainer["type"] = "Trainer"

for t in task_names + ["common", "data_loaders", "models", "schedule"]:
    sys.modules.pop(t)
