import os
from argparse import ArgumentParser

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from cap.data.datasets.bevdepth import Box, LidarPointCloud
from cap.data.datasets.bevdepth import Quaternion
from PIL import Image
from cap.utils.distributed import rank_zero_only


from  cap.data.datasets.bevdepth import \
    map_name_from_general_to_detection
from projects.panorama.configs.datasets.changan_lmdb_infer_datasets import bev_data_path,bev_vis_threshold

def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('idx',
                        type=int,
                        help='Index of the dataset to be visualized.')
    parser.add_argument('result_path', help='Path of the result json file.')
    parser.add_argument('target_path',
                        help='Target path to save the visualization result.')

    args = parser.parse_args()
    return args


def get_ego_box(box_dict,
                ego2global_rotation=None,
                ego2global_translation=None,
                is_ego=False):
    box = Box(
        box_dict['translation'],
        box_dict['size'],
        Quaternion(box_dict['rotation']),
    )
    # if is_ego == False:  # 不是ego，默认是global
    #     trans = -np.array(ego2global_translation)
    #     rot = Quaternion(ego2global_rotation).inverse
    #     box.translate(trans)
    #     box.rotate(rot)
    # else:  # 是ego，则不用从global ==> ego
    #     pass

    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
        axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    template = (np.array((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2)

    corners3d = np.tile(boxes3d[:, None, 3:6],
                        [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3),
                                      boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def get_bev_lines(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:

        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]],
                        [corners[st, 1], corners[ed, 1]]])
    return ret


def get_cam_corners(corners, translation, rotation, cam_intrinsics):
    # ego2pixel
    # rotation输入为旋转矩阵
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    # cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ rotation
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners
def changan_visual(results,
                   sensor2ego_trans,
                   sensor2ego_rot,
                   intrin_mats,
                   visual_save_path,
                   cameras_paths,
                   return_undistort_imgs = True,
                   score_threshold = 0.6,
                   limit_range = 60,
                   class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian','traffic_cone',],
                   IMG_KEYS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'],
                   DefaultAttribute = {'car': 'vehicle.parked', 'pedestrian': 'pedestrian.moving', 'trailer': 'vehicle.parked', 'truck': 'vehicle.parked', 
                                       'bus': 'vehicle.moving', 'motorcycle': 'cycle.without_rider', 'construction_vehicle': 'vehicle.parked',
                                       'bicycle': 'cycle.without_rider', 'barrier': '', 'traffic_cone': ''} # 本来是evaluator.DefaultAttribute
                   ):
    # 拿到本sample的sample_token，以便能够在infos_dict中拿到gt
    for idx, result in enumerate(results):# 遍历本次所有sample的结果 每次取得一个sample，因为一般我们转模型用batchsize==1，所以这里大概率就是循环一次
        pred_corners, pred_class = [], [] # Get prediction corners 拿到预测的corners
        assert len(result[0]) == len(result[1]) and len(result[0]) == len(result[2])
        for ind, (box, score, label) in enumerate(zip(result[0], result[1], result[2])): # 遍历每个样本的所有的bbox信息
            name = class_names[label]
            center = box[:3].cpu().tolist()
            wlh = box[[4, 3, 5]].cpu().tolist()
            box_yaw = box[6].cpu()
            box_vel = box[7:].cpu().tolist()
            box_vel.append(0)
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw)
            nusc_box = Box(center, wlh, quat, velocity=box_vel)
            # import pdb
            # pdb.set_trace()
            if np.sqrt(nusc_box.velocity[0]**2 + nusc_box.velocity[1]**2) > 0.2: # 速度的值
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer',]:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = DefaultAttribute[name]
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = DefaultAttribute[name]
            
            box = dict(
                sample_token=os.path.basename(visual_save_path), # 因为没有token，这里使用图像名字替代
                translation=nusc_box.center.tolist(),
                size=nusc_box.wlh.tolist(),
                rotation=nusc_box.orientation.elements.tolist(),
                velocity=nusc_box.velocity[:2],
                detection_name=name,
                detection_score=float(score),
                attribute_name=attr,
            )
            
            if box['detection_score'] >= score_threshold and box['detection_name'] in class_names: # 阈值和类别
                box3d = get_ego_box(box, None, None, is_ego=True)# 这个结果本来就是自车坐标系下的，不用再进行转换, 所以设置 is_ego = True
                if np.linalg.norm(box3d[:2]) <= limit_range:# 
                    corners = get_corners(box3d[None])[0]
                    pred_corners.append(corners)
                    pred_class.append(box['detection_name'])

        # Set figure size
        plt.figure(figsize=(24, 8))

        for i, k in enumerate(IMG_KEYS):
            # Draw camera views
            fig_idx = i + 1 if i < 3 else i + 2
            plt.subplot(2, 4, fig_idx)

            # Set camera attributes
            plt.title(k)
            plt.axis('off')
            if "CAM_FRONT" == k: # 因为500111这辆车前视FOC较大
                plt.xlim(0, 3840)
                plt.ylim(2160, 0)
            else:
                plt.xlim(0, 1920)
                plt.ylim(1536, 0)
            if return_undistort_imgs: # 拿到畸变矫正后的图像
                img = cameras_paths[i].astype(np.uint8) ################# 这里要读取图像 因为是单bacth 所以用0作为下标
            else:
                img = mmcv.imread(cameras_paths[i][0]) ################# 这里要读取图像 因为是单bacth 所以用0作为下标
                # TODO
                # ...                                        
            # 因为图像变化了，这里需要对应更改内参。预测出来的值都是针对畸变矫正后的图像，并非针对 256 * 704 那个输入网络之前的图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw images
            plt.imshow(img)

            # Draw 3D predictions
            for corners, cls in zip(pred_corners, pred_class):
                cam_corners = get_cam_corners(corners, 
                                                list(sensor2ego_trans[0,i,:].cpu().numpy()), 
                                                list(sensor2ego_rot[0,i,:].cpu().numpy()), 
                                                intrin_mats[0,0,i,0:3,0:3].cpu().numpy())
                lines = get_3d_lines(cam_corners)
                for line in lines:
                    plt.plot(line[0], line[1], c=cm.get_cmap('tab10')(class_names.index(cls)))

        # Draw BEV
        plt.subplot(1, 4, 4)

        # Set BEV attributes
        plt.title('LIDAR_TOP')
        plt.axis('equal')
        plt.xlim(-40, 40)
        plt.ylim(-40, 40)

        # Draw BEV predictions
        for corners in pred_corners:
            lines = get_bev_lines(corners)
            for line in lines:
                plt.plot([-x for x in line[1]], line[0], c='g', label='prediction')

        # Set legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=1)

        # Save figure
        plt.tight_layout(w_pad=0, h_pad=2)
        
        plt.savefig(visual_save_path.replace(".bmp", ".jpg")) # 保存图像


@rank_zero_only
def bev_vis(
    # idx,
    nusc_results_file,
    img_metas,
    dump_file,
    threshold=bev_vis_threshold,
    show_range=60,
    show_classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ],
):
    # Set cameras
    IMG_KEYS = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    # infos = mmcv.load('../../../datas/nuScenes/nuscenes_infos_val.pkl')

    # assert idx < len(infos)
    # Get data from dataset
    results = nusc_results_file['results']
    # info = infos[idx]
    # print(info)
    # print(nusc_results_file)
    # print(results)
    for img_meta in img_metas:

        token = img_meta["token"]
        result = results[token]
        # print([x for x in result[token]])
        # assert(0)
        # lidar_path = info['lidar_infos']['LIDAR_TOP']['filename']
        # lidar_points = np.fromfile(os.path.join('data/nuScenes', lidar_path), dtype=np.float32, count=-1).reshape(-1, 5)[..., :4]
        # lidar_calibrated_sensor = info['lidar_infos']['LIDAR_TOP']['calibrated_sensor']
        # Get point cloud
        # pts = lidar_points.copy()
        # ego2global_rotation = np.mean([info['cam_infos'][cam]['ego_pose']['rotation'] for cam in IMG_KEYS],0)
        # ego2global_translation = np.mean([info['cam_infos'][cam]['ego_pose']['translation'] for cam in IMG_KEYS], 0)
        # ego2global_rotation = img_meta["ego2global_rotation"]
        # ego2global_translation = img_meta["ego2global_translation"]
        # lidar_points = LidarPointCloud(lidar_points.T)
        # lidar_points.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        # lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))
        # pts = lidar_points.points.T

        # Get GT corners
        # gt_corners = []
        # for i in range(len(info['ann_infos'])):
        #     if map_name_from_general_to_detection[info['ann_infos'][i]['category_name']] in show_classes:
        #         box = get_ego_box(dict(size=info['ann_infos'][i]['size'], rotation=info['ann_infos'][i]['rotation'], translation=info['ann_infos'][i]['translation'],),
        #                           ego2global_rotation,
        #                           ego2global_translation)

        #         if np.linalg.norm(box[:2]) <= show_range:
        #             corners = get_corners(box[None])[0]
        #             gt_corners.append(corners)

        # Get prediction corners
        pred_corners, pred_class = [], []
        for box in result:
            if box['detection_score'] >= threshold and box[
                    'detection_name'] in show_classes:
                box3d = get_ego_box(box, None,
                                    None)
                # box3d[2] += 0.5 * box3d[5]  # NOTE
                if np.linalg.norm(box3d[:2]) <= show_range:
                    corners = get_corners(box3d[None])[0]
                    pred_corners.append(corners)
                    pred_class.append(box['detection_name'])

        # Set figure size
        plt.figure(figsize=(24, 8))

        for i, k in enumerate(IMG_KEYS):
            # Draw camera views
            fig_idx = i + 1 if i < 3 else i + 2
            plt.subplot(2, 4, fig_idx)

            # Set camera attributes
            plt.title(k)
            plt.axis('off')
            # plt.xlim(0, 1600)
            # plt.ylim(900, 0)

            data_root = bev_data_path.data_root
            img = mmcv.imread(
                os.path.join(data_root, img_meta['file_name'][i]))
            h,w,c = img.shape
            plt.xlim(0, w)
            plt.ylim(h, 0)
            # img = Image.open(os.path.join(data_root, img_meta['file_name'][i]))
            # w,h = img.size
            # print (h,w)
            # ratio = max(900/h, 1600/w)
            # print (int(w*ratio), int(h*ratio))
            # img = img.resize((int(w*ratio), int(h*ratio)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw images
            plt.imshow(img)
            # print (img_meta['cam_infos'][k].keys())

            # Draw 3D predictions
            for corners, cls in zip(pred_corners, pred_class):
                cam2car = np.array(img_meta['cam_infos'][k]['cam2car'])
                translation = cam2car[:3, -1]
                rotation = cam2car[:3, :3]
                instrin = np.array(img_meta['cam_infos'][k]['intrinsic'])
                cam_corners = get_cam_corners(
                    corners,
                    translation,
                    rotation,
                    instrin,
                    # img_meta['cam_infos'][k]['calibrated_sensor']['translation'],
                    # #img_meta['ego2global_translation_corn'][i],
                    # img_meta['cam_infos'][k]['calibrated_sensor']['rotation'],
                    # #img_meta['ego2global_rotation_corn'][i],
                    # img_meta['cam_infos'][k]['calibrated_sensor']
                    # ['camera_intrinsic']
                    #img_meta['camera_intrinsic'][i],
                )
                lines = get_3d_lines(cam_corners)
                for line in lines:
                    plt.plot(line[0],
                            line[1],
                            c=cm.get_cmap('tab10')(show_classes.index(cls)))

        # Draw BEV
        plt.subplot(1, 4, 4)

        # Set BEV attributes
        plt.title('LIDAR_TOP')
        plt.axis('equal')
        plt.xlim(-40, 40)
        plt.ylim(-40, 40)

        # Draw point cloud
        # plt.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')

        # # Draw BEV GT boxes
        # for corners in gt_corners:
        #     lines = get_bev_lines(corners)
        #     for line in lines:
        #         plt.plot([-x for x in line[1]], line[0], c='r', label='ground truth')

        # Draw BEV predictions
        for corners in pred_corners:
            lines = get_bev_lines(corners)
            for line in lines:
                plt.plot([-x for x in line[1]], line[0], c='g', label='prediction')

        # Set legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(),
                by_label.keys(),
                loc='upper right',
                framealpha=1)

        # Save figure
        plt.tight_layout(w_pad=0, h_pad=2)
        # plt.show()
        d_f = os.path.join(dump_file, f"{token}.png")
        print(d_f)
        plt.savefig(d_f)
        plt.close() # 避免内存泄漏

