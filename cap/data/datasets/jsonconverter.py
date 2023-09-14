import os
import torch
import mmcv
import numpy as np
from PIL import Image
import torch.utils.data as data
# from pyquaternion import Quaternion
import cv2
import copy
'''
data_root = '../../../datas/changan_data/nuScenes'
# train_info_paths = os.path.join(data_root, 'nuscenes_infos_train.pkl')
test_info_paths = os.path.join(data_root, 'nuscenes_infos_val.pkl')

print("load plk")
cam_direction = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT','CAM_BACK', 'CAM_BACK_RIGHT']
cam_info_exclude = ["filename", "height", "width","sample_token"]#camera_intrinsic
bev_pkl =  mmcv.load(test_info_paths) 
image_dict = dict()
img_list = []
anno_infos_list = []
print("plk loaded")
for i, databev in enumerate(bev_pkl):
    sample_token = databev["sample_token"]
    timestamp = databev["timestamp"]
    anno_infos = databev["ann_infos"]
    anno_dict = dict()
    anno_dict["sample_token"] = sample_token
    for i,x in enumerate(anno_infos):
        anno_infos[i]["velocity"]=x["velocity"].tolist()
    anno_dict["bev_anno"] = anno_infos
    anno_infos_list.append(anno_dict)
    bev_img = dict()
    for cd in cam_direction:
        bev_img["sample_token"] = sample_token
        #filenames =  databev["cam_infos"][cd]["filename"]
        cam_info_dict = copy.deepcopy(dict(filter(lambda x:x[0] not in cam_info_exclude , databev["cam_infos"][cd].items())))
        del cam_info_dict["calibrated_sensor"]["camera_intrinsic"]
        bev_img["cam_infos"] = {f"{cd}":cam_info_dict}
        bev_img["lidar_infos"] = databev["lidar_infos"]
        image_key =  databev["cam_infos"][cd]["filename"].split("/")
        bev_img["image_key"] = image_key[2]
        bev_img["image_source"] = databev["cam_infos"][cd]["filename"]
        bev_img["id"] = sample_token # Not sure if the id is the sample_token, pls check
        camera_intrinsic =databev["cam_infos"][cd]["calibrated_sensor"]["camera_intrinsic"]
        calib_x = copy.deepcopy(camera_intrinsic[0])
        calib_y = copy.deepcopy(camera_intrinsic[1])
        calib_z = copy.deepcopy(camera_intrinsic[2])
        calib_x.append(0.0),calib_y.append(0.0),calib_z.append(0.0)
        bev_img["calib"] =[calib_x, calib_y, calib_z]
        bev_img["timestamp"] = timestamp
        bev_img["distCoeffs"] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        bev_img["shape"] =[databev["cam_infos"][cd]["height"],  databev["cam_infos"][cd]["width"]]
        img_list.append(bev_img)
       
image_dict["images"] = img_list
image_dict["ann_infos"] = anno_infos_list
image_dict["cam_sweeps"] = []
image_dict["lidar_sweeps"] = []
import json
j = "/root/codebase/zwj-capv2/cap_develop/cap/data/datasets/data.json"

with open(j, "w") as fp: 
    json.dump(image_dict, fp,indent=2)
# json_object = json.dumps(img_list, 'data.json', indent = 4) 
# print(json_object)
print("pkl to json conversion process finished")
'''
import os
import struct 
import glob
image_data_all  = "../../../datas/cadata/images/*.jpg"
img_list = sorted(glob.glob(image_data_all))
intri =  "../../../datas/cadata/instrinsic.txt"
with open(intri, "r") as f:
    x = f.read().splitlines()
from itertools import islice  
x = [i for i in x if i !=''] 
x = iter(x) 
it = [4,4,4,4,4,4]
Output = [list(islice(x,i)) for i in it
       ]
l = []
t = []
s = []
for i in range(len(Output)):
    singleintric = Output[i][1:]
    print(singleintric)
    for id in range(len(singleintric)):
        print(len(singleintric))
        intric = singleintric[id].split(' ')
 
        l.append(intric)
    l.append(['0','0','0','1'])
print(l)        
l = iter(l) 
it = [4,4,4,4,4,4]
Output = [list(islice(l,i)) for i in it
       ]
print(l)
print(torch.from_numpy(np.array(Output).astype(float)))
print(torch.from_numpy(np.array(Output).astype(float))[0])
# for i in img_list:
#     img = Image.open(
#                     i) #
    

# dep =  "../../../datas/cadata/dep.bin"
# intri = "../../../datas/cadata/intristic.txt"
# dim  = "../../../datas/cadata/dim.bin"
# rot = "../../../datas/cadata/rot.bin"
# offset = "../../../datas/cadata/offset.bin"
# wh = "../../../datas/cadata/wh.bin"
# rotbin = open(rot,"rb")
# size = os.path.getsize(rot)
# for i in range(size):
#     rot_data = rotbin.read(1)
#     rot_data = struct.unpack("B",rot_data)[0]/180

#     print(rot_data)