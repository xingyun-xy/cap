import argparse
import json
import os
import pickle
import shutil
import warnings

import cv2
import numpy as np

np.set_printoptions(suppress=True)
CATEGORIES = [{"name": "pedestrian", "id": 1},
              {"name": "car", "id": 2},
              {"name": "cyclist", "id": 3},
              {"name": "bus", "id": 4},
              {"name": "truck", "id": 5},
              {"name": "specialcar", "id": 6},
              {"name": "tricycle", "id": 7},
              {"name": "dontcare", "id": 8}]
CLS_ARG2CAP = {"box_truck": "truck",
               "railed_vehicle": "dontcare",
               "official_signaler": "pedestrian",
               "vehicular_trailer": "truck",
               "animal": "dontcare",
               "traffic_light_trailer": "truck",
               "motorcycle": "tricycle",
               "pedestrian": "pedestrian",
               "bicyclist": "cyclist",
               "school_bus": "bus",
               "regular_vehicle": "car",
               "message_board_trailer": "truck",
               "construction_barrel": "tricycle",
               "large_vehicle": "truck",
               "truck": "truck",
               "wheeled_device": "specialcar",
               "bicycle": "tricycle",
               "motorcyclist": "cyclist",
               "truck_cab": "truck",
               "sign": "dontcare",
               "dog": "dontcare",
               "bollard": "dontcare",
               "mobile_pedestrian_crossing_sign": "dontcare",
               "stop_sign": "dontcare",
               "stroller": "pedestrian",
               "wheelchair": "specialcar",
               "wheeled_rider": "specialcar",
               "articulated_bus": "bus",
               "bus": "bus",
               "construction_cone": "dontcare"}
CAT_MAP = {i['name']: i['id'] for i in CATEGORIES}


def collect_cls(data):
    """收集argoverse2数据集中所有的类别集合，并返回id"""
    cls = set()
    for d in data:
        cls.update(d['gt_names'])
    categories = [{'name': k.lower(), 'id': i + 1} for i, k in enumerate(cls)]
    return cls, categories


def _draw_box_3d(image, corner3d, c=(0, 0, 255), show_arrow=True, thickness=1):
    face_idx = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(
                image,
                (corner3d[f[j], 0], corner3d[f[j], 1]),
                (corner3d[f[(j + 1) % 4], 0], corner3d[f[(j + 1) % 4], 1]),
                c,
                thickness,
                lineType=cv2.LINE_AA,
            )

        if not show_arrow:
            if ind_f == 0:
                cv2.line(
                    image,
                    (corner3d[f[0], 0], corner3d[f[0], 1]),
                    (corner3d[f[2], 0], corner3d[f[2], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    image,
                    (corner3d[f[1], 0], corner3d[f[1], 1]),
                    (corner3d[f[3], 0], corner3d[f[3], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        # show an arrow to indicate 3D orientation of the object
        if show_arrow:
            # 4,5,6,7
            p1 = (
                         corner3d[0, :] + corner3d[1, :] + corner3d[2, :] + corner3d[3, :]
                 ) / 4
            p2 = (corner3d[0, :] + corner3d[1, :]) / 2
            p3 = p2 + (p2 - p1) * 0.5

            p1 = p1.astype(np.int32)
            p3 = p3.astype(np.int32)

            cv2.line(
                image,
                (p1[0], p1[1]),
                (p3[0], p3[1]),
                c,
                thickness,
                lineType=cv2.LINE_AA,
            )
    return image


def mini_file(images, img_root, out_dir, json_name):
    """转换一个mini数据集出来"""
    for image in images:
        ori_path = os.path.join(img_root, image['image_source'])
        out_path = os.path.join(out_dir, json_name.rstrip('.json'), f"{image['image_key']}.jpg")
        image['image_source'] = f"{json_name.rstrip('.json')}/{image['image_key']}.jpg"
        out_d = os.path.split(out_path)[0]
        if not os.path.exists(out_d):
            os.makedirs(out_d)
        shutil.copyfile(ori_path, out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("argoverse", help="argoverse data root")
    parser.add_argument("out", help="out data dir")
    parser.add_argument("--mode", default='val', help="data mode")
    parser.add_argument("--show", action='store_true', help="show result")
    return parser.parse_args()


def yaw_vector(yaw):
    """根据数据集坐标系，使用向量来描述yaw角度"""
    x = np.sin(yaw)
    y = np.cos(yaw)
    z = np.zeros_like(x)
    return np.vstack([x, y, z])


def show(img):
    cv2.namedWindow("img", 0)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cam2img(cam_coord: np.ndarray, calib: np.ndarray):
    """
    Args:
        cam_coord: n * 3 * k， n个空间上的点。 一列表示一个点
        calib: 3 * 3
    """
    assert cam_coord.shape[1] == calib.shape[1]
    img_coord = calib @ cam_coord / cam_coord[:, -1:]  # 结果除z
    return img_coord[:, :-1]  # n, 2, k


def img2cam(img_coord: np.ndarray, calib: np.ndarray, depth: np.ndarray):
    """图片像素坐标转换到相机坐标系。
     img_coord: 像素坐标 (2,n)
     calib: 相机内参 (3,3)
     depth: 深度参数, 当作Z_c (n,)
    """
    cam_coord_back_proj = np.vstack([img_coord, np.ones_like(depth)])  # 3,n
    cam_coord_back_proj = np.linalg.inv(calib) @ cam_coord_back_proj  # 3,n
    cam_coord_back_proj = cam_coord_back_proj * depth  # 3,n

    return cam_coord_back_proj


def corners(gt, src=(0.5, 1.0, 0.5)):
    """根据location计算出3D框的8个点，在argoverse中，location是3D框的正中心. gt:7 * n"""
    assert gt.shape[0] == 7, 'gt的shape应该为(7,n)'
    location = gt[:3]
    l, w, h, yaw = gt[3:]  # argoverse的dim顺序为l,w,h
    c_, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[[c, 0, s], [0, 1, 0], [-s, 0, c]] for c, s in zip(c_, s_)], dtype=np.float32)  # n个旋转矩阵 n,3,3
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
    z_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]
    y_corners = [np.zeros_like(h)] * 4 + [-h, -h, -h, -h]
    # y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corner = np.array([x_corners, y_corners, z_corners], dtype=np.float32).transpose((2, 0, 1))  # (n,3,8)
    corners_3d = R @ corner  # n,3,8
    dst = np.array((0.5, 1.0, 0.5))  # cap中x,y,z为3D框底面中心
    src = np.array(src)
    location += np.vstack([w, h, l]) * (dst - src).reshape(3, 1)

    return location[None] + corners_3d.T  # 8,3,n. 相机坐标系


def corners_2d(corners_3d, calib):
    """根据3D框坐标生成2D包围框. corners_3d: (8,3,n); calib: (3,3)"""
    corner_2d = cam2img(corners_3d.T, calib)  # n,2,8
    x1, y1 = corner_2d.min(axis=-1).T
    x2, y2 = corner_2d.max(axis=-1).T
    center = np.vstack([(x2 + x1) / 2, (y1 + y2) / 2]).T
    return np.vstack((x1, y1, x2, y2)).T, center


def roy_y2alpha(location, yaw):
    """通过航向角生成alpha角"""
    x, y, z = location
    return yaw - np.arctan2(x, z)


def is_in_picture(img_wh, depth, corner_2d):
    """根据中心点判断目标是否在当前相机的图像内, 返回对应位置的布尔值"""
    width, height = img_wh

    area_img = width * height
    area = (corner_2d[:, 2] - corner_2d[:, 0]) * (corner_2d[:, 3] - corner_2d[:, 1])
    a = area < area_img / 2

    xb1, yb1, xb2, yb2 = corner_2d.T
    xi1, yi1, xi2, yi2 = np.array([0., 0., width, height])[None].repeat(corner_2d.shape[0], axis=0).T
    x1 = np.maximum(xb1, xi1)
    y1 = np.maximum(yb1, yi1)
    x2 = np.minimum(xb2, xi2)
    y2 = np.minimum(yb2, xi2)
    area_inter = np.maximum(0, (x2 - x1 + 1)) * np.maximum(0, (y2 - y1 + 1))
    aig = area_inter > area / 5  # gt与图片交集大于1/5的gt
    aii = area_inter < area_img / 3  # gt与图片交集小于图片的1/3

    # x0, y0 = (center_2d > 0).T
    # x_w, y_h = center_2d[:, 0] < width, center_2d[:, 1] < height  # 2D 中心点落在图片内
    d_ = depth > 0.
    return d_ * a * aig * aii


def parse_one_annotation(data, annotations, images, root, show_res=False):
    """解析一个data，返回每个gt在图像中的2D框、在相机中的3D框及对应信息、在雷达中的3D框及对应信息"""
    gt = data['gt_boxes'].T  # (7,n)
    dim = gt[3:-1]
    yaw = gt[-1]
    yaw_ego = yaw_vector(yaw)
    names = data['gt_names']
    assert gt.shape[-1] == len(names), '不规范的标注文件'
    gt_loc = np.vstack([gt[:3], np.ones((1, gt.shape[-1]))])

    Tl = data['ego2global_translation']
    Rl = data['ego2global_rotation']
    lidar2global = np.vstack([np.hstack([Rl, Tl[..., None]]), np.array([0., 0., 0., 1.])])

    for cam, anno in data['cams'].items():
        idx = 1
        calib = anno['cam_intrinsic']
        timestamp = str(anno['timestamp'])
        img_path = anno['data_path']
        img = cv2.imread(os.path.join(root, img_path))
        img_wh = img.shape[:2][::-1]  # cv2的shape为(h,w,3), 反转为(w,h)
        img_id = os.path.split(img_path)[-1][:-4]

        R1 = anno['sensor2ego_rotation']
        T1 = anno['sensor2ego_translation']
        sensor2ego = np.vstack([np.hstack([R1, T1[..., None]]), np.array([0., 0., 0., 1.])])
        R2 = anno['ego2global_rotation']
        T2 = anno['ego2global_translation']
        ego2global = np.vstack([np.hstack([R2, T2[..., None]]), np.array([0., 0., 0., 1.])])
        translation = np.linalg.inv(ego2global @ sensor2ego) @ lidar2global  # lidar -> global -inv-> sensor
        R = np.linalg.inv(R2 @ R1) @ Rl

        # translation = np.linalg.inv(sensor2ego)  # ego就是雷达坐标系，所以不用转到global下。这样做也是正确的
        # R = np.linalg.inv(R1)

        gt_loc_cam = translation @ gt_loc
        gt_loc_cam = gt_loc_cam[:3]
        gt_loc_cam[1] += (dim[-1] / 2)  # 将x,y,z从3D框集合中心平移到底面中心
        yaw_cam = R @ yaw_ego
        yaw_cam = -np.arctan2(yaw_cam[-1], yaw_cam[0]) + np.pi / 2
        alpha = roy_y2alpha(gt_loc_cam, yaw_cam)
        corner_3d = corners(np.vstack([gt_loc_cam, dim, yaw_cam]))
        corner_2d, center_2d = corners_2d(corner_3d, calib)
        in_pic_mask = is_in_picture(img_wh, gt_loc_cam[-1], corner_2d)
        if in_pic_mask.sum() < 1:
            warnings.warn(f"No object founded in {img_path}")
            continue

        gt_loc_cam = gt_loc_cam.T[in_pic_mask]  # n,3
        dim_cam = dim.T[in_pic_mask][:, ::-1]  # n,3  转为h,w,l
        yaw_cam = yaw_cam[in_pic_mask]  # n
        alpha = alpha[in_pic_mask]  # n
        corner_2d = corner_2d[in_pic_mask]  # n,4
        center_2d = center_2d[in_pic_mask]  # n,2
        center_3d = img2cam(center_2d.T, calib, gt_loc_cam.T[-1]).T  # n,3
        center_3d[:, 1] += (dim[-1][in_pic_mask] / 2)  # 反投影到相机坐标然后平移到底面中心
        loc_offset = gt_loc_cam - center_3d
        names_cam = names[in_pic_mask]

        if show_res:
            c3d = corner_3d.T[in_pic_mask]
            c3d = cam2img(c3d, calib).astype(np.int32)
            for x1, y1, x2, y2 in corner_2d:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (102, 333, 222), 1)
            for c3 in c3d:
                _draw_box_3d(img, c3.T)
            show(img)

        for loc, dim_, yaw_, alpha_2d, cor_2d, cent_2d, loc_off, cat in zip(gt_loc_cam,
                                                                            dim_cam,
                                                                            yaw_cam,
                                                                            alpha,
                                                                            corner_2d,
                                                                            center_2d,
                                                                            loc_offset,
                                                                            names_cam):
            res = dict(
                timestamp=timestamp,
                image_id=img_id,
                id='{}{:03d}'.format(timestamp, idx),
                category_id=CAT_MAP[CLS_ARG2CAP[cat.lower()]],
                bbox=cor_2d.tolist(),
                in_camera=dict(
                    dim=dim_.tolist(),
                    depth=loc[-1],
                    alpha=alpha_2d,
                    location=loc.tolist(),
                    location_offset=loc_off.tolist(),
                    rotation_y=yaw_,
                    location_offset_2d=loc_off.tolist(),
                    alpha_2d=alpha_2d
                ),
                ignore=False,
                occlusion='full_visible',
                bbox_2d=cor_2d.tolist()
            )
            annotations.append(res)
            idx += 1
        images.append(dict(
            image_key=img_path.lstrip('sensors_data/').rstrip('.jpg').replace('/', '_'),
            id=img_id,
            image_source=img_path,
            calib=calib.tolist(),
            timestamp=timestamp,
            distCoeffs=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
            shape=img_wh[::-1],  # l,w,h -> h,w,l
        ))
        print(f"\rimage: {img_id}.jpg done!", end=' ')


def infos(root, mode='train'):
    path = os.path.join(root, f"argoverse2_infos_{mode}.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def cvt_argoverse2cap():
    """将argoverse标注文件转为cap需要的标注文件并保存"""
    args = parse_args()
    data = infos(args.argoverse, args.mode)
    print(f"start convert! dataset: {args.mode}. total frames: {len(data)}...")
    annotations = list()
    images = list()
    result = dict(
        annotations=annotations,
        images=images,
        categories=CATEGORIES
    )
    for p in data[:500]:
        parse_one_annotation(p, annotations, images, args.argoverse, args.show)

    print("\nall done! save begin...")
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    json_name = f"argoverse2_infos_{args.mode}.json"
    mini_file(result['images'], args.argoverse, args.out, json_name)
    with open(os.path.join(args.out, json_name), 'w') as f:
        f.write(json.dumps(result))


if __name__ == '__main__':
    cvt_argoverse2cap()
