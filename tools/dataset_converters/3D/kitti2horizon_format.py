#此脚本将mmdet转换的coco 格式的nuscenes数据集转换成cap可以打包的格式也类似coco格式

import json
import shutil
import os
import os.path as osp
import copy
import numpy as np
import math
from argparse import ArgumentParser
from tqdm import tqdm
import torch


# CAM_NAME_LIST = [
#     'CAM_FRONT',
#     'CAM_FRONT_LEFT',
#     'CAM_FRONT_RIGHT',
#     'CAM_BACK_LEFT',
#     'CAM_BACK_RIGHT',
#     'CAM_BACK',
# ]

CATEGORIES_KITTI = [
    {
        "id": 0,
        "name": "Pedestrian"
    },
    {
        "id": 1,
        "name": "Cyclist"
    },
    {
        "id": 2,
        "name": "Car"
    },
]

CATEGORIES_AUTO_LABEL = [
    {
        "name": "pedestrian",
        "id": 1
    },
    {
        "name": "car",
        "id": 2
    },
    {
        "name": "cyclist",
        "id": 3
    },
    {
        "name": "bus",
        "id": 4
    },
    {
        "name": "truck",
        "id": 5
    },
    {
        "name": "specialcar",
        "id": 6
    },
    {
        "name": "tricycle",
        "id": 7
    },
    {
        "name": "dontcare",
        "id": 8
    },
]

# categories mapping from kitti to auto_label
CATEGORIES_MAP = {
    0: 1,  # car -> car
    1: 3,  # truck -> truck
    2: 2,  # trailer -> specialcar
}

# auto_label annotation file structure
COCO_STRUCT = {
    "annotations": [],
    "images": [],
    "categories": CATEGORIES_AUTO_LABEL,
}

ANNO_INSTANCE = {
    "timestamp": "",  #时间戳
    "image_id": "",  #同一张图片image_id一样
    "id": "",  #目标id=image_id+000编号
    "category_id": 0,
    "bbox": [],  #//左上宽高
    "in_camera": {
        "dim": [],  #3D目标h w l
        "depth": 0,  #3D目标中心深度z
        "alpha": 0,  #相机坐标系下观测角
        "location": [],  #相机坐标系下3D中心坐标
        "location_offset": [],  #"2d bbox"中心点反投影到3D后与3D中心点的offset,单位m
        "rotation_y": 0  #相机坐标系下航向角
        # location_offset_2d 和 alpha_2d 字段根据是否使用 bbox_2d 进行添加
    },
    # "in_lidar": {},  # in_lidar 字段没有用到
    "ignore": False,  #是否忽略掉
    "occlusion": "",  #遮挡情况  #kitti 有0，1，2，3 四种情况，遮挡占比依次增大
    "bbox_2d": None  # 默认都是NULL，2D大模型刷出来的，用不到
}

IMG_INSTANCE = {
    "image_key": "",
    "image_source": "",
    "id": "",
    "calib": [[], [], []],  #相机内参：3*4矩阵
    # "Tr_vel2cam": [[],[],[],[]], #雷达到相机的转换矩阵：4*4矩阵
    "timestamp": "",
    "distCoeffs": [0.0] * 8,  #畸变系数! nuscenes 没有 distCoeffs，但是CAP训练代码里用到，但是我看代码这个字段没有的话就不做去畸变，也可以都设为0
    "shape": [900, 1600],
    # "Tr_vcs2cam": [[],[],[],[]], #车到相机的转换矩阵：4*4矩阵
    # "calib_all":{}  #所有标定信息
    # "ignore_mask": {}     #! nuscenes 没有，但是CAP训练代码里用到，但是我看代码这个字段没有的话就不做
}


class KittiToAutoLabel(object):

    def __init__(self, json_path, img_base_path, save_path, use_bbox_2d=True) -> None:
        self.json_path = json_path
        self.img_base_path = img_base_path
        self.save_path = save_path
        self.coco_struct = COCO_STRUCT
        self.anno_instance = ANNO_INSTANCE
        self.img_instance = IMG_INSTANCE
        self.use_bbox_2d = use_bbox_2d
        self.copy_img_list = []

    def convert(self):
        with open(self.json_path, 'r') as f:
            json_anno = json.load(f)
            annotations, images = json_anno["annotations"], json_anno["images"]
            print('start convertting labels...')

            img_ids_calib_map = dict()
            for img in tqdm(images):
                img_dst = copy.deepcopy(self.img_instance)
                file_name = osp.basename(img["file_name"])
                img_dst["image_key"] = osp.splitext(file_name)[0]
                img_dst["image_source"] = 'data/' + file_name
                img_dst["id"] = img["id"]
                # kitti 4*4  ==>  auto_label 3*4
                src_calib = img["cam_intrinsic"][:3]
                img_ids_calib_map[img["id"]] = src_calib  #3x4
                img_dst["calib"] = src_calib

                img_dst["timestamp"] = img_dst["image_key"] #随便给一个
                img_dst["shape"] = [img["height"], img["width"]]

                self.coco_struct["images"].append(img_dst)
            print(f'img nums: {len(img_ids_calib_map)}')

            for anno in tqdm(annotations):
                if anno["image_id"] not in img_ids_calib_map:
                    continue
                anno_dst = copy.deepcopy(self.anno_instance)
                file_name = osp.basename(anno["file_name"])
                anno_dst["timestamp"] = osp.splitext(file_name)[0] #随便给一个
                anno_dst["image_id"] = anno["image_id"]
                anno_dst["id"] = str(anno["id"])
                anno_dst["category_id"] = CATEGORIES_MAP[anno["category_id"]]
                # nuscenes: top_left w h ,   autolabel: top_left w h
                anno_dst["bbox"] = anno["bbox"]
                # nuscenes: [l, h, w]   autolabel" [h, w, l]
                anno_dst["in_camera"]["dim"] = [anno["bbox_cam3d"][4], anno["bbox_cam3d"][5], anno["bbox_cam3d"][3]]
                anno_dst["in_camera"]["depth"] = anno["bbox_cam3d"][2]  # z
                # anno["bbox_cam3d"][-1]是（全局）偏航角，而anno_dst["in_camera"]["alpha"]是观测角，需要转换
                anno_dst["in_camera"]["alpha"] = rotation_y2alpha(anno["bbox_cam3d"][0:3], anno["bbox_cam3d"][-1])
                # xyz  nuscenes: 经过转换的coco style的nuscenes结果的location是3dbox的几何中心点； autolabel：3dbox的底面中心点，与kitti3d定义的相同，需要转换
                anno_dst["in_camera"]["location"] = np.array(anno["bbox_cam3d"][0:3]).tolist()
                # location_offset是bbox中心点反投影到3D后与3D中心点的offset，需要进行转换
                bbox = np.array(anno_dst["bbox"])
                center_img_coord = np.array(bbox[:2] + bbox[2:] / 2) #2D中心点坐标
                calib = np.array(img_ids_calib_map.get(anno_dst["image_id"]))

                #内参矩阵3x4的处理方式
                image_points = np.append(center_img_coord,anno_dst["in_camera"]["depth"])
                batch1_image_points = torch.from_numpy(np.expand_dims(image_points,axis=0))
                calib = torch.from_numpy(calib)
                cam_coord_back_proj = points_img2cam(batch1_image_points, calib)
                cam_coord_back_proj = cam_coord_back_proj.numpy().squeeze()

                #内参矩阵3x3的处理方式
                # cam_coord_back_proj = img2cam(center_img_coord, calib, anno_dst["in_camera"]["depth"])#TODO 是图像坐标系转相机坐标系么？


                #! 这二者是谁减去谁要弄清楚，我看代码中发现应该是location_offset = 3dbox底面中心点 - 2d框中心点反投影到相机坐标再移到底面，或者是3dbox中心点-2d中心点反投影
                # anno_dst["in_camera"]["location_offset"] = (anno_dst["in_camera"]["location"] - cam_coord_back_proj).tolist()
                anno_dst["in_camera"]["location_offset"] = (np.array(anno["bbox_cam3d"][0:3]) - cam_coord_back_proj).tolist()
                anno_dst["in_camera"]["rotation_y"] = anno["bbox_cam3d"][-1]
                # 默认都是False
                anno_dst["ignore"] = False
                # occlusion这个字段在原始的nuscenes里有（visibility.json），但是转成coco格式后忽略掉了，需要改mmdet3d的数据转换脚本#TODO改了么？
                anno_dst["occlusion"] = "full_visible"

                if self.use_bbox_2d:
                    anno_dst["in_camera"]["location_offset_2d"] = anno_dst["in_camera"]["location_offset"] #TODO 为什么不一样
                    anno_dst["in_camera"]["alpha_2d"] = anno_dst["in_camera"]["alpha"]
                    anno_dst["bbox_2d"] = anno_dst["bbox"]

                self.coco_struct["annotations"].append(anno_dst)

                if osp.join(self.img_base_path, anno["file_name"]) not in self.copy_img_list:
                    self.copy_img_list.append(osp.join(self.img_base_path, anno["file_name"]))

    def save_all_imgs(self):
        target_path = osp.join(self.save_path, 'data')
        os.makedirs(target_path, exist_ok=True)
        print('start copying images...')
        for img_path in tqdm(self.copy_img_list):
            img_path = img_path.replace("\\","/")
            assert osp.isfile(img_path),"image file not exist! "
            shutil.copy(img_path, target_path)

    def save_json(self):
        os.makedirs(self.save_path, exist_ok=True)
        json_path = osp.join(self.save_path, 'data.json')
        with open(json_path, 'w+') as f:
            json.dump(self.coco_struct, f)


def alpha2rotation_y(location, alpha):
    '''
    alpha = rotation_y - arctan(x/z)
    '''
    x, y, z = location
    return alpha + math.atan2(x, z)


def rotation_y2alpha(location, rotation_y):
    '''
    alpha = rotation_y - arctan(x/z)
    '''
    x, y, z = location
    return rotation_y - math.atan2(x, z)


def alpha2rotation_y_test():
    # autolabel 前两行instance
    x = 11.106876445266911
    y = 1.0380369607608357
    z = 6.484840995618121
    alpha = -2.3086372774517923
    rotation_y = -1.6771085259929368

    arctan = math.atan2(x, z)
    error = rotation_y - arctan - alpha
    print(error)
    print(error / math.pi * 180)

    x, y, z = -13.200766528364175, 1.4469503397055474, 16.13510603646059
    alpha = -1.1566576114519906
    rotation_y = -1.6954958002049858

    # # KITTI  alpha = rotation_y - arctan(x/z)
    # # Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
    # x = -16.53
    # y = 2.39
    # z = 58.49
    # alpha = 1.85
    # rotation_y = 1.57
    # arctan = math.atan(x/z)

    # Math.atan()只能返回一个角度值,因此确定他的角度非常的复杂，因此通常采用math.atan2
    # arctan = math.atan(x/z)

    arctan = math.atan2(x, z)

    error = rotation_y - arctan - alpha

    # KITTI算出来是对的，但是autolabel算出来不对  #TODO 解决了么？

    print(error)
    print(error / math.pi * 180)


def cam2img(cam_coord: np.ndarray, P: np.ndarray):
    '''
    Args:
        cam_coord: 3 * n
        P: 3 * 3
    Return:
        img_coord: 3 * n (e.g. np.array([[u v 1], [u v 1]...]))
    '''
    assert cam_coord.shape[0] == P.shape[1]
    img_coord = 1 / cam_coord[-1] * np.dot(P, cam_coord)
    img_coord = img_coord / img_coord[-1]
    return img_coord

def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

def img2cam(img_coord: np.ndarray, P: np.ndarray, depth: float):
    '''
    inv(K) * [u,v,1].T = 1/Zc * [Xc, Yc, Zc].T

    Args:
        img_coord: 3 * n (e.g. np.array([[u v 1], [u v 1]...]))
        P: 3 * 3
        depth: float
    Return:
        cam_coord_back_proj: 3 * n
    '''
    assert img_coord.shape[0] == P.shape[1]
    cam_coord_back_proj = np.dot(np.linalg.inv(P), img_coord)
    cam_coord_back_proj = cam_coord_back_proj * depth
    return cam_coord_back_proj


def cam2img_test():
    # # nuscenes val第一个目标
    # # nuscenes 应该是把3d框中心点相机坐标投影到2d图像上得到center2d
    # inner = np.array([[1257.8625342125129,
    # 	0.0,
    # 	827.2410631095686],
    # 	[0.0,
    # 	1257.8625342125129,
    # 	450.915498205774],
    # 	[0.0,
    # 	0.0,
    # 	1.0]])
    # bbox = np.array([553.4947253760256,
    # 	490.5584879921071,
    # 	30.329964533936845,
    # 	62.24330538453637])
    # # 2d框的中心点
    # center = bbox[:2] + 1/2 * bbox[2:]
    # # 3d框中心点投影到图像上的点
    # center2d = np.array([568.765380859375,
    # 	521.4768676757812])
    # print(f'error between center and center2d: {center - center2d}')
    # cam_coord = np.array([-7.516707974170363,	1.5012318792386194,	36.525215135341675])
    # img_coord = 1/cam_coord[-1] * np.dot(inner, cam_coord)
    # # img_coord = img_coord/img_coord[-1]
    # print(img_coord)
    # # TODO 投影算出来的图像坐标与center2d坐标还是有误差的，需要看看mmdet3d数据转换的代码是怎样做的
    # print(f'error between bbox_cam3d proj to img and center2d: {img_coord[:2] - center2d}')

    # nuscenes train第一个目标
    # nuscenes 应该是把3d框中心点相机坐标投影到2d图像上得到center2d
    # inner = np.array([[1266.417203046554,
    # 	0.0,
    # 	816.2670197447984],
    # 	[0.0,
    # 	1266.417203046554,
    # 	491.50706579294757],
    # 	[0.0,
    # 	0.0,
    # 	1.0]])
    # bbox = np.array([1206.5693751819117,
    # 	477.86111828160216,
    # 	19.31993062031279,
    # 	35.78389940122628])
    # # 2d框的中心点
    # center = bbox[:2] + 1/2 * bbox[2:]
    # # 3d框中心点投影到图像上的点
    # center2d = np.array([1216.1754150390625,
    # 	495.6607666015625])
    # print(f'error between center and center2d: {center - center2d}')
    # cam_coord = np.array([18.63882979851619,
    # 	0.19359276352412746,
    # 	59.02486732065484])
    # img_coord = 1/cam_coord[-1] * np.dot(inner, cam_coord)
    # # img_coord = img_coord/img_coord[-1]
    # print(img_coord)
    # # TODO 投影算出来的图像坐标与center2d坐标还是有误差的，需要看看mmdet3d数据转换的代码是怎样做的
    # print(f'error between bbox_cam3d proj to img and center2d: {img_coord[:2] - center2d}')

    # autolabel 第一行instance
    # bbox = np.array([3617.740734656658, 979.0386911438773, 190.30124187523325, 299.3098354956669])
    # img_coord = np.append(bbox[:2] + bbox[2:] / 2, 1)
    # cam_coord = np.array([11.106876445266911,
    #                 1.0380369607608357,
    #                 6.484840995618121])
    # inner = np.array([[
    #                 2450.01,
    #                 0.0,
    #                 1920.81,
    #             ],
    #             [
    #                 0.0,
    #                 2450.04,
    #                 1085.07,
    #             ],
    #             [
    #                 0.0,
    #                 0.0,
    #                 1.0,
    #             ]])

    # img_coord_proj = 1/cam_coord[-1] * np.dot(inner, cam_coord)
    # print(f'error between bbox_cam3d proj to img and center2d: {img_coord - img_coord_proj}')

    # cam_coord_back_proj = np.dot(np.linalg.inv(inner), img_coord)
    # # 这样反投影回去得到的location_offset不对
    # print(cam_coord_back_proj[:2] - cam_coord[:2])

    # autolabel 第三行instance，也就是有bbox_2d字段的
    bbox = np.array([2479.137451171875, 1129.5220336914062, 402.9375, 202.70550537109375])
    img_coord = np.append(bbox[:2] + bbox[2:] / 2, 1)
    cam_coord = np.array([5.5278952624810955, 1.6767569789993098, 16.322349359187847])
    inner = np.array([[
        2450.01,
        0.0,
        1920.81,
    ], [
        0.0,
        2450.04,
        1085.07,
    ], [
        0.0,
        0.0,
        1.0,
    ]])

    # img_coord_proj = 1/cam_coord[-1] * np.dot(inner, cam_coord)
    # print(f'error between bbox_cam3d proj to img and center2d: {img_coord - img_coord_proj}')
    cam_coord_back_proj = np.dot(np.linalg.inv(inner), img_coord)
    # 这样反投影回去得到的location_offset不对
    print(cam_coord_back_proj[:2] / cam_coord_back_proj[-1] - cam_coord[:2])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--json_path', type=str, default='/workspace/data/changan_data/kitti/kitti_infos_train_mono3d.coco.json', help='json_path')
    parser.add_argument('--img_base_path', type=str, default='/workspace/data/changan_data/kitti', help='img_base_path')
    parser.add_argument('--save_path', type=str, default='/workspace/data/changan_data/kitti_horizon_format/train', help='dir to save results')
    args = parser.parse_args()
    return args

  

if __name__ == '__main__':
    args = parse_args()
    converter = KittiToAutoLabel(args.json_path, args.img_base_path, args.save_path)
    converter.convert() #转换标注格式
    converter.save_json() #保存转换后的json
    converter.save_all_imgs() #复制图片到data文件夹
    print("done!")

