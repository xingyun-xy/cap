dataset_ids = dict(
    # detection_real_3d={
    #     "vehicle_heatmap_3d_detection": [
    #         # "temp/nuscenes_eval",
    #         # "nuscenes_test/val"
    #         # "nuscenes/common_6000/camera_front_1000"
    #         "temp/eval/detection_3D/nuscenes/common_6000/camera_front_1000"
    #     ],
    # },
    segmentation={
        "default_segmentation": [
            "semantic_parsing/common_6000/camera_back_1000",
            "semantic_parsing/common_6000/camera_back_left_1000",
            "semantic_parsing/common_6000/camera_back_right_1000",
            "semantic_parsing/common_6000/camera_front_875",
            "semantic_parsing/common_6000/camera_front_left_1000",
            "semantic_parsing/common_6000/camera_front_right_998",
        ],
        "lane_segmentation": [
            "lane_parsing/common_6000/camera_back_1000",
            "lane_parsing/common_6000/camera_back_left_1000",
            "lane_parsing/common_6000/camera_back_right_1000",
            "lane_parsing/common_6000/camera_front_1000",
            "lane_parsing/common_6000/camera_front_left_1000",
            "lane_parsing/common_6000/camera_front_right_1000",
        ],
    }, )

bev_eval_datapath = dict(
    # data_root=f"../data/changan_data/nuScenes",
    # img_path=f"/tmp/dataset/dataset-803-version-1/trainval",
    # anno_path=f"/tmp/dataset/dataset-803-version-2/nuscenes_infos_val.pkl",
    data_root=f"../data/changan_data/BEV_changan/DTS000003054_V5/annotation",
    img_path=f"../data/changan_data/BEV_changan/DTS000003054_V5",
    anno_path=f"../data/changan_data/BEV_changan/DTS000003054_V5/annotation",
    ca_instrins=f"./ca_instrains",
)

from easydict import EasyDict

dataset_ids = EasyDict(dataset_ids)
bev_eval_datapath = EasyDict(bev_eval_datapath)
