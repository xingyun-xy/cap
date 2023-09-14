from functools import partial
from importlib import import_module

import torch
from bev_matrixvt import (
    BEVDepthHead,
    BEVDepthHead_loss,
    BEVDepthHead_loss_v2,
    MatrixVT,
    bev_matrixvt_test_model,
    bev_matrixvt_train_model,
    dbound,
    depth_channels,
    downsample_factor,
    secondfpn,
)
from common import (
    backbone,
    bev_batch_size,
    ckpt_dir,
    datapaths,
    input_hw,
    is_local_train,
    log_freq,
    num_machines,
    tasks,
    training_step,
    vis_tasks,
)

# from data_loaders import data_loader
# from models import model, val_model, test_model, traced_inputs
from schedule import (
    base_lr,
    interval_by,
    num_steps,
    save_interval,
    warmup_steps,
)

from cap.callbacks.metric_updater import update_metric_using_regex
from cap.data.collates import collate_fn_bevdepth
from cap.engine.processors.loss_collector import collect_loss_by_regex
from cap.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
task_name = "singletask_bev"
vis_tasks.append(task_name)

# step-specific args
num_steps = num_steps[training_step]
base_lr = base_lr[training_step]

# -------------------------- multitask --------------------------
seed = None  # random seed
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

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
        # dict(type="BgrToYuv444", rgb_input=True),
        # dict(
        #     type="TorchVisionAdapter",
        #     interface="Normalize",
        #     mean=128.0,
        #     std=128.0,
        # ),
    ],
    loss_collector=collect_loss_by_regex("^.*singletask_bev.*"),
    enable_amp=enable_amp,
)

deploy_model = bev_matrixvt_test_model
# data_shape = (3, input_hw[0], input_hw[1])
# deploy_inputs = {"img": torch.randn((1,) + data_shape)}
deploy_inputs = dict(
    imgs_batch=torch.randn((1, 1, 6, 3, input_hw[0], input_hw[1])),
    sensor2ego_mats=torch.randn((1, 1, 6, 4, 4)),
    intrin_mats=torch.randn((1, 1, 6, 4, 4)),
    ida_mats=torch.randn((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.randn((1, 1, 6, 4, 4)),
    bda_mat=torch.randn((1, 4, 4)),
)

# -------------------------- callbacks --------------------------
stat_callback = dict(type="StatsMonitor", log_freq=log_freq)

lr_callback = dict(
    type="PolyLrUpdater",
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
    save_dir=log_dir.as_posix() if is_local_train else "/tmp/model",  # noqa
    update_freq=log_freq,
    tb_update_funcs=[
        T.tb_update_func for T in TASK_CONFIGS
        if getattr(T, "tb_update_func", None) is not None
    ],
)

# metric_updaters = [T.metric_updater for T in TASK_CONFIGS]
loss_names = [
    "loss",
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

# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
    metric_updater,
]

profiler = dict(type="SimpleProfiler")
# dirpath=log_dir.as_posix(),
# filename='profile.log',)

# region bev train dataloader config
H = 900  # 900
W = 1600  # 1600
# -------------------------------Input------------------------------------
# H,W = input_hw
final_dim = input_hw
ida_aug_conf = {
    # 'resize_lim': (0.386, 0.55),
    "resize_lim": (0.36, 0.38),
    "final_dim":
    final_dim,
    "rot_lim": (-5.4, 5.4),
    "H":
    H,
    "W":
    W,
    "rand_flip":
    True,
    "bot_pct_lim": (0.0, 0.0),
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams":
    6,
}
bda_aug_conf = {
    "rot_lim": (-22.5, 22.5),
    "scale_lim": (0.95, 1.05),
    "flip_dx_ratio": 0.5,
    "flip_dy_ratio": 0.5,
}
CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
# data_root='../../../datas/nuScenes'
# train_info_paths = os.path.join(data_root,'nuscenes_infos_train.pkl')
is_train = True
use_cbgs = True
num_sweeps = 1
img_conf = dict(
    img_mean=[123.675, 116.28, 103.53],
    img_std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
sweep_idxes = []
key_idxes = []
use_fusion = False
return_depth = True
# endregion
ds = datapaths.bev_depth
data_root = ds.train_data_paths[0].img_path
# data_root = '/code/data/nuScenes'p
train_info_paths = ds.train_data_paths[0].anno_path
# train_info_paths = '/code/data/nuScenes/nuscenes_infos_train.pkl'
# data_root = '/tmp/dataset/dataset-803-version-1/trainval'
# train_info_paths = '/tmp/dataset/dataset-803-version-2/nuscenes_infos_train.pkl'
data_loader = dict(
    type=torch.utils.data.DataLoader,
    sampler=None,
    drop_last=True,
    num_workers=8,
    batch_size=bev_batch_size,
    shuffle=False,
    collate_fn=partial(collate_fn_bevdepth, is_return_depth=True),
    dataset=dict(
        type="ComposeRandomDataset",
        sample_weights=[path.sample_weight for path in ds.train_data_paths],
        datasets=[
            dict(
                # type="NuscDetDataset",
                type="CaBev3dDataset",
                ida_aug_conf=ida_aug_conf,
                bda_aug_conf=bda_aug_conf,
                classes=CLASSES,
                data_root=path.img_path,
                info_paths=path.anno_path,
                is_train=is_train,
                use_cbgs=use_cbgs,
                num_sweeps=num_sweeps,
                img_conf=img_conf,
                return_depth=return_depth,
                sweep_idxes=sweep_idxes,
                key_idxes=key_idxes,
                use_fusion=use_fusion,
            ) for path in ds.train_data_paths
        ],
    ),
)

# batch_size = 1,
# shuffle = False,
# num_workers = 0,
# drop_last = True,
# sampler = None,))
# -------------------------- trainer --------------------------
with_bn_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=bev_matrixvt_train_model,
    # resume_epoch_or_step = True,            #resume
    # resume_optimizer = True,                #resume
    # model_convert_pipeline=dict(            #resume
    #     type="ModelConvertPipeline",
    #     qat_mode="with_bn",
    #     converters=[
    #         dict(
    #             type="LoadCheckpoint",
    #             checkpoint_path=os.path.join(
    #                 ckpt_dir, "with_bn-checkpoint-last.pth.tar"
    #             ),
    #         ),
    #     ],
    # ),
    data_loader=data_loader,
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-5)},
        lr=base_lr * num_machines,
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(num_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
)
# profiler=profiler,)

inputs = dict(
    train=dict(
        img=torch.zeros((6, 3, input_hw[0], input_hw[1])),
        sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
        intrin_mats=torch.rand((1, 1, 6, 4, 4)),
        ida_mats=torch.rand((1, 1, 6, 4, 4)),
        sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
        bda_mat=torch.rand((1, 4, 4)),
    ),
    val=dict(
        img=torch.zeros((6, 3, input_hw[0], input_hw[1])),
        sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
        intrin_mats=torch.rand((1, 1, 6, 4, 4)),
        ida_mats=torch.rand((1, 1, 6, 4, 4)),
        sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
        bda_mat=torch.rand((1, 4, 4)),
        img_metas_batch=[],
    ),
    test=dict(
        img=torch.zeros((6, 3, input_hw[0], input_hw[1])),
        sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
        intrin_mats=torch.rand((1, 1, 6, 4, 4)),
        ida_mats=torch.rand((1, 1, 6, 4, 4)),
        sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
        bda_mat=torch.rand((1, 4, 4)),
        # mlp_input = torch.randn((1,1,6,27)),
        # circle_map = torch.randn((1,112,16384)),
        # ray_map=torch.randn((1,216,16384))
    ),
    onnx=dict(
        img=torch.zeros((6, 3, input_hw[0], input_hw[1])),
        sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
        intrin_mats=torch.rand((1, 1, 6, 4, 4)),
        ida_mats=torch.rand((1, 1, 6, 4, 4)),
        sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
        bda_mat=torch.rand((1, 4, 4)),
        # mlp_input = torch.randn((1,1,6,27)),
        # circle_map = torch.randn((1,112,16384)),
        # ray_map=torch.randn((1,216,16384))
    ),
)

bev_matrixvt_depth_loss = dict(
    type="bev_matrixvt_depth_loss",
    BEVDepthHead_loss=BEVDepthHead_loss,
    BEVDepthHead_lossv2=BEVDepthHead_loss_v2,
    __graph_model_name="bev_matrixvt_depth_loss",
    depth_channels=depth_channels,
    dbound=dbound,
    downsample_factor=downsample_factor,
)


def get_model(mode):
    return dict(type="bev_matrixvt",
                backbone=backbone,
                neck=secondfpn,
                depthnet=MatrixVT,
                head=BEVDepthHead,
                mode=mode,
                loss_v3=bev_matrixvt_depth_loss if mode == "train" else None,
                depth_channels=depth_channels,
                dbound=dbound,
                downsample_factor=downsample_factor)


if is_local_train and len(device_ids) == 1:
    with_bn_trainer.pop("sync_bn")
    with_bn_trainer["type"] = "Trainer"

# for t in task_names + ["common", "data_loaders", "models", "schedule"]:
#     sys.modules.pop(t)
