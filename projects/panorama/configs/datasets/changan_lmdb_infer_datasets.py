import os

from easydict import EasyDict

# pilot_data_path = '/tmp/dataset/dataset-803-version-1/trainval'   #云平台路径
# pilot_data_path = f"../data/eval/detection_3D/nuscenes/common_6000/camera_front_1000/"  #BEV+real3D本地推理路径
pilot_data_path = (
    f"../data/eval/lane_parsing/common_6000/camera_front_1000/"  # 两个分割本地推理路径
)

# infer 和 eval 都是用的这个
bev_data_path = dict(
    data_root=f'/tmp/dataset/dataset-3480-version-1',                 # 7980 samples
    anno_path=f'/tmp/dataset/dataset-3480-version-1/annotation',
    # data_root=f'/tmp/dataset/dataset-3698-version-1',                 # 5640 samples
    # anno_path=f'/tmp/dataset/dataset-3698-version-1/annotation',   #云平台路径
    # data_root=f'../data/changan_data/nuScenes',                                 #本地路径
    # anno_path=f'../data/changan_data/nuScenes/nuscenes_infos_val.pkl',          #本地路径
    data_root_changan="../data/cadata/",
    ca_instrins=f'./ca_instrains.txt',
    TestInfoJsonPath=f'"/data/changan_data/nuScenes/uniteddata.json"')

bev_vis_threshold = 0.4  # BEV可视化阈值

bev_data_path = EasyDict(bev_data_path)
