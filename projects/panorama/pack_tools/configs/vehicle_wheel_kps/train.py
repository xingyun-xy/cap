from __future__ import absolute_import, print_function
import os
import time

from cap.data.packer.utils import (
    get_default_pack_args_from_environment,
    init_logger,
    ls_img_folder_and_anno_file,
)

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose

# logger
logger = init_logger(
    log_file=f"log/packing_kps2_{int(time.time())}.log", overwrite=True
)

# ##############################################
# kps data config

# number of kps in annos
num_kps = 2

# [
#     bbox_type,
#     bbox_clur, bbox_occ, bbox_ignore,
#     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
#     kp_back_x, kp_back_y, kp_front_x, kp_front_y,
#     occ_back, occ_front
# ]

obj_ele_num_2kps = 1 + 3 + 4 + 4 + 2

# [
#     bbox_type,
#     bbox_clur, bbox_occ, bbox_ignore,
#     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
#     kp_back_right_x, kp_back_right_y, kp_back_left_x, kp_back_left_y,
#     kp_front_x, kp_front_y, occ_back_right, occ_back_left, occ_front
# ]

obj_ele_num_3kps = 1 + 3 + 4 + 6 + 3

# [
#     bbox_type,
#     bbox_clur, bbox_occ, bbox_ignore,
#     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
#     (kps_i_x, kps_i_y) * num_kps,
#     occ_i * num_kps
# ]

obj_ele_num_xkps = 1 + 3 + 4 + num_kps * 2 + num_kps

# bbox name
class_name = "vehicle"
# kps name
sub_class_name = "vehicle_kps_8"
# 指定为8点数据转换为2点打包
task_name = "vehicle_2_kps"
task_name_list = ["vehicle_8_kps", "vehicle_2_kps"]
assert task_name in task_name_list

save_pic = True

valid_types = [
    "Sedan_Car",
    "Car",
    "SUV",
    "MiniVan",
    "Van",
    "Tram",
    "Bus",
    "Misc",
    "BigTruck",
    "Truck",
    "Lorry",
    "MotorcycleWithPerson",
    "BikeWithPerson",
    "other",
    "unknown",
    "Trucks",
    "Small_Medium_Car",
]


# 关键点遮挡属性
kps_occ_id_dict = {
    "full_visible": 2,
    "self_occluded": 1,
    "other_occluded": 1,
    "outside": 0,
    "ignore": 3,
}  # noqa
conf_ignore_list = []  # 'super_low' or 'low'

# 目标框遮挡属性
occ_ignore_list = ["invisible"]

min_width = 30
min_height = 30
# ##############################################

input_task_type = "vehicle_keypoints"
task_type = "vehicle_wheel_kps"


# input_root_dir = f"data/train/{input_task_type}"
# output_dir = f"data/lmdb/{task_type}/"
input_root_dir = f"data/train/{input_task_type}/zhoushi_horizon_format/horizon_format_971"
output_dir = f"data/lmdb/{task_type}/lmdb_zhoushi_0971"
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

anno_transformer = [
    dict(
        type="VehicleKps8to2Ts",
        class_name=class_name,
        sub_class_name=sub_class_name,
        ts_sub_class_name="p_WheelKeyPoints_2",
    ),
    dict(
        type="KeyPointAnnoJsonTs",
        num_kps=num_kps,
        conf_ignore_list=conf_ignore_list,
    ),
    dict(
        type="KeyPointAnnoTs",
        num_kps=num_kps,
        valid_types=valid_types,
        min_width=min_width,
        min_height=min_height,
        kps_occ_id_dict=kps_occ_id_dict,
        occ_ignore_list=occ_ignore_list,
        obj_ele_num_2kps=obj_ele_num_2kps,
        obj_ele_num_3kps=obj_ele_num_3kps,
        obj_ele_num_xkps=obj_ele_num_xkps,
    ),
]

# pack image data
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
    type="VizKpsDetAnno",
    num_kps=num_kps,
    task_name=task_name,
    save_flag=viz_save_flag,
    save_path=viz_save_path,
)
