from argparse import ArgumentParser
import json
import shutil
import os
import os.path as osp
import copy
import numpy as np
import math


'''
Folder structure:
nuscenes:
data
--samples
----CAM_FRONT
----CAM_BACK
...
...
--nuscenes_infos_train_mono3d.coco.json
--nuscenes_infos_val_mono3d.coco.json

auto_label_pack:
--vehicle_3d
----data
------xxx.jpg
------xxx.jpg
...
...
----xxx.json

'''


CAM_NAME_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

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
        }
    ]

CATEGORIES_NUSCENES = [{
		"id": 0,
		"name": "car"
	},
	{
		"id": 1,
		"name": "truck"
	},
	{
		"id": 2,
		"name": "trailer"
	},
	{
		"id": 3,
		"name": "bus"
	},
	{
		"id": 4,
		"name": "construction_vehicle"
	},
	{
		"id": 5,
		"name": "bicycle"     # 这里是自行车而不是骑行人
	},
	{
		"id": 6,
		"name": "motorcycle"
	},
	{
		"id": 7,
		"name": "pedestrian"
	},
	{
		"id": 8,
		"name": "traffic_cone"
	},
	{
		"id": 9,
		"name": "barrier"
	}]

# categories mapping from nuscenes to auto_label
# nuscenes donnot have categories: cyclist and tricycle, which are included in auto_label
CATEGORIES_MAP = {
    0: 2,         # car -> car
    1: 5,         # truck -> truck
    2: 6,         # trailer -> specialcar
    3: 4,         # bus -> bus
    4: 6,         # construction_vehicle -> specialcar
    5: 8,         # bicycle -> dontcare
    6: 8,         # motorcycle -> dontcare
    7: 1,         # pedestrian -> pedestrian
    8: 8,         # traffic_cone -> dontcare
    9: 8,         # barrier -> dontcare
}

# auto_label annotation file structure
COCO_STRUCT = {
    "annotations":[],
    "images": [],
    "categories": CATEGORIES_AUTO_LABEL
}
OBJECT_INSTANCE = {
            "timestamp": "",
            "image_id": "",
            "id": "",
            "category_id": 0,
            "bbox": [],
            "in_camera": {
                "dim": [],
                "depth": 0,
                "alpha": 0,
                "location": [],
                "location_offset": [],
                "rotation_y": 0
                # location_offset_2d 和 alpha_2d 字段根据是否使用 bbox_2d 进行添加
            },
            # "in_lidar": {},            # in_lidar 字段没有用到
            "ignore": False,
            "occlusion": "",
            "bbox_2d": None             # 默认都是NULL
        }

IMG_META_INSTANCE = {
        "image_key": "",
        "image_source": "",
        "id": "",
        "calib": [[],[],[]],
        # "Tr_vel2cam": [[],[],[],[]],
        "timestamp": "",
        "distCoeffs": [0.0] * 8,     #! nuscenes 没有 distCoeffs，但是CAT训练代码里用到，但是我看代码这个字段没有的话就不做去畸变，也可以都设为0
        "shape": [900, 1600],
        # "Tr_vcs2cam": [[],[],[],[]],
        # "ignore_mask": {}     #! nuscenes 没有，但是CAT训练代码里用到，但是我看代码这个字段没有的话就不做
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--json_path', type=str, default='/data/train_save_dir/zoumingjie/vehicle_3d/nuscenes_all/nuscenes_infos_train_mono3d.coco.json', help='json_path')
    parser.add_argument(
        '--img_base_path', type=str, default='/data/nuscenes_mini/mini/samples', help='img_base_path')
    parser.add_argument(
        '--save_path', type=str, default='/data/train_save_dir/zoumingjie/vehicle_3d/nuscenes_all/train/', help='dir to save results')

    args = parser.parse_args()
    return args

class NuscenesToAutoLabel(object):
    '''
    Convert coco format Nuscenes dataset annotations and images to
    auto_label format(coco format)
    '''
    def __init__(self, json_path, img_base_path, save_path, use_bbox_2d=True) -> None:
        self.json_path = json_path
        self.img_base_path = img_base_path
        self.save_path = save_path
        self.coco_struct = COCO_STRUCT
        self.object_instance = OBJECT_INSTANCE
        self.img_meta_instance = IMG_META_INSTANCE
        self.use_bbox_2d = use_bbox_2d


    def convert(self):
        with open(self.json_path, 'r') as f:
            json_anno = json.load(f)
            annotations, images = json_anno["annotations"], json_anno["images"]
            print('start convertting labels...')

            img_ids_calib_map = dict()
            for img in images:
                img_dst = copy.deepcopy(self.img_meta_instance)
                # file_name example: 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
                file_name = osp.basename(img["file_name"])
                img_dst["image_key"] = osp.splitext(file_name)[0]
                # img_dst["image_source"] = osp.join('data', file_name)
                img_dst["image_source"] = 'data/' + file_name
                img_dst["id"] = img["id"]
                # nuscenes 3*3   auto_label 3*4
                src_calib = img["cam_intrinsic"]
                img_ids_calib_map[img["id"]] = src_calib
                src_calib = [i + [0.0] for i in src_calib]
                img_dst["calib"] = src_calib
                img_dst["timestamp"] = img_dst["image_key"].split('__')[-1]
                img_dst["shape"] = [img["height"], img["width"]]

                self.coco_struct["images"].append(img_dst)
            print(f'img nums: {len(img_ids_calib_map)}')

            for anno in annotations:
                if anno["image_id"] not in img_ids_calib_map:
                    continue
                anno_dst = copy.deepcopy(self.object_instance)
                # file_name example: 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
                file_name = osp.basename(anno["file_name"])
                anno_dst["timestamp"] = osp.splitext(file_name)[0].split('__')[-1]
                anno_dst["image_id"] = anno["image_id"]
                # id的方式不同，nuscenes是从0开始依次计数，不区分图像，autolabel是每个图像里的目标计数
                anno_dst["id"] = str(anno["id"])
                anno_dst["category_id"] = CATEGORIES_MAP[anno["category_id"]]
                # nuscenes: top_left w h ,   autolabel: top_left w h
                anno_dst["bbox"] = anno["bbox"]
                # nuscenes: [l, h, w]   autolabel" [h, w, l]
                anno_dst["in_camera"]["dim"] = [anno["bbox_cam3d"][4], anno["bbox_cam3d"][5], anno["bbox_cam3d"][3]]
                anno_dst["in_camera"]["depth"] = anno["bbox_cam3d"][2]    # z
                # anno["bbox_cam3d"][-1]是（全局）偏航角，而anno_dst["in_camera"]["alpha"]是观测角，需要转换
                anno_dst["in_camera"]["alpha"] = rotation_y2alpha(anno["bbox_cam3d"][0:3], anno["bbox_cam3d"][-1])
                # xyz  nuscenes: 经过转换的coco style的nuscenes结果的location是3dbox的几何中心点； autolabel：3dbox的底面中心点，与kitti3d定义的相同，需要转换
                anno_dst["in_camera"]["location"] = (np.array(anno["bbox_cam3d"][0:3]) + np.array([0, anno_dst["in_camera"]["dim"][0] / 2.0, 0])).tolist()
                # location_offset是bbox中心点反投影到3D后与3D中心点的offset，需要进行转换
                bbox = np.array(anno_dst["bbox"])
                center_img_coord = np.append(bbox[:2] + bbox[2:] / 2, 1)
                calib = np.array(img_ids_calib_map.get(anno_dst["image_id"]))
                cam_coord_back_proj = img2cam(center_img_coord, calib, anno_dst["in_camera"]["depth"])
                #! 这二者是谁减去谁要弄清楚，我看代码中发现应该是location_offset = 3dbox底面中心点 - 2d框中心点反投影到相机坐标再移到底面，或者是3dbox中心点-2d中心点反投影
                # anno_dst["in_camera"]["location_offset"] = (anno_dst["in_camera"]["location"] - cam_coord_back_proj).tolist()
                anno_dst["in_camera"]["location_offset"] = (np.array(anno["bbox_cam3d"][0:3]) - cam_coord_back_proj).tolist()
                anno_dst["in_camera"]["rotation_y"] = anno["bbox_cam3d"][-1]
                # 默认都是False
                anno_dst["ignore"] = False
                # occlusion这个字段在原始的nuscenes里有（visibility.json），但是转成coco格式后忽略掉了，需要改mmdet3d的数据转换脚本
                anno_dst["occlusion"] = "full_visible"

                if self.use_bbox_2d:
                    anno_dst["in_camera"]["location_offset_2d"] = anno_dst["in_camera"]["location_offset"]
                    anno_dst["in_camera"]["alpha_2d"] = anno_dst["in_camera"]["alpha"]
                    anno_dst["bbox_2d"] = anno_dst["bbox"]


                self.coco_struct["annotations"].append(anno_dst)



    def save_all_imgs(self):
        target_path = osp.join(self.save_path, 'data')
        os.makedirs(target_path, exist_ok=True)
        print('start copying images...')
        for cam in CAM_NAME_LIST:
            cam_folders = osp.join(self.img_base_path, cam)
            for img in os.listdir(cam_folders):
                full_file_name = osp.join(cam_folders, img)
                if osp.isfile(full_file_name):
                    shutil.copy(full_file_name, target_path)

    def save_json(self):
        self.convert()
        os.makedirs(self.save_path, exist_ok=True)
        json_path = osp.join(self.save_path, 'data.json')
        with open(json_path, 'w+') as f:
            json.dump(self.coco_struct, f)


def alpha2rotation_y(location, alpha):
    '''
    alpha = rotation_y - arctan(x/z)
    '''
    x, y, z= location
    return alpha + math.atan2(x, z)

def rotation_y2alpha(location, rotation_y):
    '''
    alpha = rotation_y - arctan(x/z)
    '''
    x, y, z= location
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
    print(error/math.pi * 180)

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

    # KITTI算出来是对的，但是autolabel算出来不对

    print(error)
    print(error/math.pi * 180)

def cam2img(cam_coord: np.ndarray, P: np.ndarray):
    '''
    Args:
        cam_coord: 3 * n
        P: 3 * 3
    Return:
        img_coord: 3 * n (e.g. np.array([[u v 1], [u v 1]...]))
    '''
    assert cam_coord.shape[0] == P.shape[1]
    img_coord = 1/cam_coord[-1] * np.dot(P, cam_coord)
    img_coord = img_coord/img_coord[-1]
    return img_coord

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
    bbox = np.array([2479.137451171875,
                1129.5220336914062,
                402.9375,
                202.70550537109375])
    img_coord = np.append(bbox[:2] + bbox[2:] / 2, 1)
    cam_coord = np.array([5.5278952624810955,
                    1.6767569789993098,
                    16.322349359187847])
    inner = np.array([[
                    2450.01,
                    0.0,
                    1920.81,
                ],
                [
                    0.0,
                    2450.04,
                    1085.07,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ]])

    # img_coord_proj = 1/cam_coord[-1] * np.dot(inner, cam_coord)
    # print(f'error between bbox_cam3d proj to img and center2d: {img_coord - img_coord_proj}')
    cam_coord_back_proj = np.dot(np.linalg.inv(inner), img_coord)
    # 这样反投影回去得到的location_offset不对
    print(cam_coord_back_proj[:2]/cam_coord_back_proj[-1] - cam_coord[:2])


def main():
    args = parse_args()
    cl = NuscenesToAutoLabel(args.json_path, args.img_base_path, args.save_path)
    cl.save_json()
    # cl.save_all_imgs()

if __name__ == '__main__':
    main()
    # alpha2rotation_y_test()



