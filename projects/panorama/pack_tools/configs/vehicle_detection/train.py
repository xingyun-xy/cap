from __future__ import absolute_import, print_function
import os

import yaml

from cap.data.packer.utils import (
    get_default_pack_args_from_environment,
    init_logger,
    ls_img_folder_and_anno_file,
)

CLASS_NAMES = [
    "CLASS_01",
    "CLASS_02",
    "CLASS_03",
    "vehicle",
    "CLASS_05",
    "CLASS_06",
    "cyclist",
    "CLASS_08",
    "CLASS_09",
    "CLASS_10",
    "CLASS_11",
    "CLASS_12",
    "CLASS_13",
    "CLASS_14",
    "CLASS_15",
    "CLASS_16",
    "CLASS_17",
    "CLASS_18",
    "CLASS_19",
    "CLASS_20",
]

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose

# logger
logger = init_logger(log_file="log/train.log", overwrite=True)

task_name = "vehicle_detection"

anno_ts_fn_config_path = f"configs/{task_name}/det_vehicle_config.yaml"
with open(anno_ts_fn_config_path) as fin:
    anno_ts_fn_config = yaml.load(fin, Loader=yaml.FullLoader)

input_root_dir = f"data/train/{task_name}/zhoushi_horizon_format/horizon_format_946"
output_dir = f"data/lmdb/vehicle_detection/lmdb_zhoushi_0946"
excluded_annos_pattern = None

folder_anno_pairs = ls_img_folder_and_anno_file(
    root_path=input_root_dir,
    anno_ext=".json",
    recursive=True,
    excluded_annos_pattern=excluded_annos_pattern,
)

idx_path = os.path.join(output_dir, "idx")
img_path = os.path.join(output_dir, "img")
anno_path = os.path.join(output_dir, "anno")

viz_num_show = 20
viz_save_flag = True
viz_save_path = os.path.join(output_dir, "viz")

anno_transformer = dict(
    type="DenseBoxDetAnnoTs",
    anno_config=anno_ts_fn_config,
    root_dir=input_root_dir,  # noqa
    verbose=verbose,
    __build_recursive=False,
)

data_packer = dict(
    type="DetSeg2DPacker",
    folder_anno_pairs=folder_anno_pairs,
    output_dir=output_dir,
    num_workers=num_workers,
    anno_transform=anno_transformer,
)


# dataset for visualization
viz_dataset = dict(
    type="DetSeg2DAnnoDataset",
    idx_path=idx_path,
    img_path=img_path,
    anno_path=anno_path,
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
)
