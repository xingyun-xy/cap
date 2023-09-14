from __future__ import absolute_import, print_function
import os

from cap.core.proj_spec.parsing import colormap
from cap.data.packer.utils import (
    get_default_pack_args_from_environment,
    get_pack_list,
    init_logger,
)

# idx2cam = {
#     "0": "front_right",
#     "1": "rear_right",
#     "2": "front_left",
#     "3": "rear_left",
#     "4": "rear",
#     "5": "front_0820",
# }

idx2cam = {
    "0": "FRONT",
    "1": "FRONT_RIGHT",
    "2": "FRONT_LEFT",
    "3": "BACK",
    "4": "BACK_LEFT",
    "5": "BACK_RIGHT",
}

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose

# logger
logger = init_logger(log_file="log/train.log", overwrite=True)

task_name = "vehicle_3d_detection"

# input_root_dir = f"data/train/{task_name}"
# input_root_dir = f"/data/train_save_dir/zoumingjie/vehicle_3d/nuscenes_all/train"
input_root_dir = f"/data/train_save_dir/zoumingjie/vehicle_3d/val"
# output_dir = f"data/lmdb/{task_name}/"
output_dir = f"/code/data/{task_name}/"
excluded_annos_pattern = None

idx_path = os.path.join(output_dir, "idx")
img_path = os.path.join(output_dir, "img")
anno_path = os.path.join(output_dir, "anno")

folder_anno_pairs = get_pack_list(
    input_root_dir, [idx2cam[str(idx)] for idx in idx2cam]
)

data_packer = dict(
    type="DetSeg2DPacker",
    folder_anno_pairs=folder_anno_pairs,
    output_dir=output_dir,
    num_workers=num_workers,
    anno_transform=None,
)

# TODO  3d visualization 

CLASS_NAMES = [
    "pedestrian",
    "car",
    "cyclist",
    "bus",
    "truck",
    "specialcar",
    "tricycle",
    "dontcare",
]

viz_num_show = 20
viz_save_flag = True
viz_save_path = os.path.join(output_dir, "viz")

# dataset for visualization
viz_dataset = dict(
    type="DetSeg2DAnnoDataset",
    idx_path=idx_path,
    img_path=img_path,
    anno_path=anno_path,
)

vis_configs = dict(
    pedestrian=dict(
        color=colormap[8],
        thickness=1,
    ),
    car=dict(
        color=colormap[9],
        thickness=1,
    ),
    cyclist=dict(
        color=colormap[2],
        thickness=1,
    ),
    bus=dict(
        color=colormap[3],
        thickness=1,
    ),
    truck=dict(
        color=colormap[4],
        thickness=1,
    ),
    specialcar=dict(
        color=colormap[7],
        thickness=1,
    ),
    tricycle=dict(
        color=colormap[5],
        thickness=1,
    ),
    dontcare=dict(
        color=colormap[6],
        thickness=1,
    ),

)
# funcation for visualization
viz_fn = dict(
    type="VizDenseBoxDetAnno",
    save_flag=viz_save_flag,
    save_path=viz_save_path,
    viz_class_id=list(range(1, len(CLASS_NAMES) + 1)),
    class_name=CLASS_NAMES,
    lt_point_id=0,
    rb_point_id=2,
    vis_configs=vis_configs
)
