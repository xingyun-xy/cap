from __future__ import absolute_import, print_function
import os

from cap.data.packer.utils import (
    get_default_pack_args_from_environment,
    get_pack_list,
    init_logger,
)

idx2cam = {
    "0": "front_right",
    "1": "rear_right",
    "2": "front_left",
    "3": "rear_left",
    "4": "rear",
    "5": "front_0820",
}


# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose

# logger
logger = init_logger(log_file="log/train.log", overwrite=True)

task_name = "ped_3d_detection"

input_root_dir = f"data/train/{task_name}"
output_dir = f"data/lmdb/{task_name}/"
excluded_annos_pattern = None


idx_path = os.path.join(output_dir, "idx")
img_path = os.path.join(output_dir, "img")
anno_path = os.path.join(output_dir, "anno")

viz_num_show = 20
viz_save_flag = True
viz_save_path = os.path.join(output_dir, "viz")

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
