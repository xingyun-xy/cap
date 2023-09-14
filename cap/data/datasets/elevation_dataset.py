# Copyright (c) Changan Auto. All rights reserved.

import json
import os
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch.utils.data as data
import yaml
from PIL import Image

from cap.data.utils import pil_loader
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .pack_type.lmdb import Lmdb

try:
    import mxnet as mx
except ImportError:
    mx = None

__all__ = ["Elevation", "ElevationRec", "ElevationFromImage"]

DEPTH_COVER_SCALE = 256
GAMMA_COVER_SCALE = 8192
HEIGHT_COVER_SCALE = 1024


def load_parameters(cfg: Mapping, invert: bool = False):
    """Load Parameters.

    Args:
        cfg: parameters of one timestamp.
        invert: if invert the pose (R, t).
    """
    R1, T1 = load_extrinsics(cfg["transform1"])
    N, camH = load_ground(cfg["ground"])
    K, fx, fy = load_intrinsics(cfg["intrinsics"])
    H1 = load_homogarphy(cfg["H1"]).astype("float32")
    R1 = R1.astype("float32")
    T1 = T1.astype("float32")
    K = K.astype("float32")
    N = N.astype("float32")
    camH = camH.astype("float32")
    if invert:
        R1 = R1.transpose(0, 1)  # [b,3,3]
        T1 = R1 @ T1 * -1
    return K, R1, T1, N, camH, H1


def load_parameters_lmdb(
    ground_dict: Mapping, intrinsics_dict: Mapping, invert: bool = False
):
    """Load Parameters.

    Args:
        ground_dict: ground and pose parameters.
        intrinsics_dict: camera intrinsics.
        invert: if invert the pose (R, t).
    """
    R1, T1 = load_extrinsics(ground_dict["transform"])
    N, camH = load_ground(ground_dict["ground"])
    K, _, _ = load_intrinsics(intrinsics_dict["intrinsics"])
    R1 = R1.astype("float32")
    T1 = T1.astype("float32")
    K = K.astype("float32")
    N = N.astype("float32")
    camH = camH.astype("float32")
    H = K.copy()  # Placeholder

    if invert:
        R1 = R1.transpose(0, 1)  # [b,3,3]
        T1 = R1 @ T1 * -1
    return K, R1, T1, N, camH, H


def load_intrinsics(dict_intrinsics: Mapping):
    """Load camera intrinsics.

    Args:
        dict_intrinsics: intrinsics of one timestamp.
    """
    intrinsics = np.zeros((3, 3))
    intrinsics[0][2] = float(dict_intrinsics["cx"])
    intrinsics[1][2] = float(dict_intrinsics["cy"])
    intrinsics[0][0] = float(dict_intrinsics["fx"])
    intrinsics[1][1] = float(dict_intrinsics["fy"])
    intrinsics[2][2] = float(1)
    fx = np.array(dict_intrinsics["fx"])
    fy = np.array(dict_intrinsics["fy"])
    return intrinsics, fx, fy


def load_extrinsics(list_extrinsics: Mapping):
    """Load camera extrinsics.

    Args:
        list_extrinsics:
            relavive pose of two adjacent timestamp.
    """
    if isinstance(list_extrinsics, dict):
        R = np.zeros((3, 3))
        T = np.zeros((3, 1))
        for i in range(9):
            x = i // 3
            y = i % 3
            R[x][y] = list_extrinsics["R"][i]
        for j in range(3):
            T[j][0] = list_extrinsics["t"][j]
    else:
        transform = np.array(list_extrinsics).reshape(4, 4)
        R = transform[:3, :3]
        T = transform[:3, 3:]
    return R, T


def load_homogarphy(list_homo: Mapping):
    """Load camera extrinsics.

    Args:
        list_homo:
            homography of two adjacent timestamp.
    """
    H = np.zeros((3, 3))
    for i in range(9):
        x = i // 3
        y = i % 3
        H[x][y] = list_homo[i]
    return H


def load_ground(dict_plane: Mapping):
    """Load ground plane norm.

    Args:
        dict_plane:
            ground norm of current frame.
    """
    nx = float(dict_plane["nx"])
    ny = float(dict_plane["ny"])
    nz = float(dict_plane["nz"])
    camH = np.array([float(dict_plane["d"])])
    N = np.array([[nx], [ny], [nz]])
    return N, camH


def load_cfg(
    ann_root: str,
    data_root: str,
    cfg_file: str,
    source: str = "FRONT_camera_infos_oft.yaml",
):
    """Load parameter dict.

    Load config of data list, contains pack name and timestamps.

    Args:
        ann_root: annotation root.
        data_root: data root.
        cfg_file: file with data packs.
        source: parametrs file.
    """
    f = open(ann_root + cfg_file, "r")
    lines = f.readlines()
    dict_cfg = {}
    for line in lines:
        pack_name = line.strip()
        pack_name_key = pack_name
        print(data_root, pack_name)
        cfg_path = os.path.join(data_root, pack_name, source)
        with open(cfg_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            dict_cfg[pack_name_key] = cfg
    f.close()
    return dict_cfg


def load_list(ann_root: str, list_file: str):
    """Load data list.

    Args:
        ann_root: annotation root.
        list_file: data list file.
    """
    # read train list from filenames(txt)
    list_file_path = os.path.join(ann_root, list_file)
    images = []
    with open(list_file_path, "r") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            # format：file_root fro_id cur_id pre_id ppre_id
            lines = [i.strip() for i in line.strip().split(" ")]
            images.append(lines)
    return images


def read_points(path: str):
    """Read pointcloud.

    Args:
        path: pointcloud path, end with '.asc'
    """
    f = open(path, "r")
    lines = f.readlines()
    lists_3 = []
    lists_2 = []
    for line in lines:
        a = line.strip().split(" ")
        tmp_3 = []
        tmp_2 = []
        for b in a[:3]:
            tmp_3.append(float(b))
        for b in a[3:]:
            tmp_2.append(float(b))
        lists_3.append(tmp_3)
        lists_2.append(tmp_2)
    return lists_3, lists_2


def point2area_distance(
    point: np.ndarray,
    N: np.ndarray,
    camH: np.ndarray,
):
    """Calculate distance to ground plane.

    Args:
        point: point in camera coordinate system.
        N: ground norm.
        camH: camera height.
    """
    if N[1] > 0:
        Nx, Ny, Nz = N[0], N[1], N[2]
    else:
        Nx, Ny, Nz = -N[0], -N[1], -N[2]
    if camH > 0:
        h = -camH
    else:
        h = camH
    d = -Nx * point[0] - Ny * point[1] - Nz * point[2] - h
    return d


def load_lidar2camere(data_root: str, pack_name: str, filename: str):
    """Load pose of lidar to camera.

    Args:
        data_root: data root.
        pack_name: pack name.
        filename: attrbute file name.
    """
    attr_path = os.path.join(data_root, pack_name, filename)
    cam_attr = json.load(open(attr_path))
    T = np.array(cam_attr["calibration"]["lidar_top_2_camera_front"])
    R_l2c = T[:3, :3]
    t_l2c = T[:3, 3:]
    return R_l2c, t_l2c


def normalize_K(K: np.ndarray, img_shape: Sequence):
    """Normalize intrinsics.

    Args:
        K: camera intrinsics.
        img_shape: image shape as (h, w).
    """
    img_h, img_w = img_shape
    K[0] = K[0] / img_w
    K[1] = K[1] / img_h
    return K


def gen_gamma(points: np.ndarray, K: np.ndarray, N: np.ndarray, camH: float):
    """Generate gamma from pointcloud.

    Args:
        points:  points in camera coordinate system.
        K: camera intrinsics.
        N: ground norm.
        camH: camera height.
    """
    uv_undist = points.T
    height_list = []
    gamma_list = []

    uv_undist_for_depth = np.transpose(np.dot(K, uv_undist))
    depth = np.reshape(uv_undist_for_depth[..., 2], (-1, 1))
    uv_undist_nor = np.true_divide(uv_undist_for_depth, np.tile(depth, [1, 3]))
    if N[1] < 0:
        N = -N
    if camH < 0:
        camH = -camH
    N = N.reshape((1, 3))
    height_list = np.dot(-N, uv_undist) + camH
    height_list = height_list.T
    gamma_list = height_list / depth
    return uv_undist_nor, gamma_list, height_list, depth


def get_ori_gamma(
    data_root: str,
    pack_name: str,
    ids: str,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    N: np.ndarray,
    camH: float,
):
    """Generate gamma, depth, height list..

    Args:
        data_root: data root.
        pack_name: pack name.
        ids: timestamp
        R: rotation of lidar to camera.
        t: transition of lidar to camera.
        K: camera intrinsics.
        N: ground norm.
        camH: camera height.
    """
    asc_path = os.path.join(data_root, pack_name, ids + ".asc")

    lists_world, lists_camera = read_points(asc_path)
    pointcoord = np.array(lists_camera)
    pointcloud = np.array(lists_world)
    pointcloud = np.dot(R, pointcloud.T) + t
    pointcoord = pointcoord / pointcoord[:, 2][:, np.newaxis][:, :2]
    pointcloud = pointcloud.T
    uv_undist_nor, gamma_list, high_list, depth_list = gen_gamma(
        pointcloud, K, N, camH
    )
    return (
        uv_undist_nor,
        pointcloud,
        pointcoord,
        gamma_list,
        high_list,
        depth_list,
    )


def get_depth_height_gamma(
    data_root: str,
    pack_name: str,
    ids: str,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    N: np.ndarray,
    camH: float,
    img_shape: Sequence,
):
    """Generate gt maps from points.

    Args:
        data_root: data root.
        pack_name: pack name.
        ids: timestamp
        R: rotation of lidar to camera.
        t: transition of lidar to camera.
        K: camera intrinsics.
        N: ground norm.
        camH: camera height.
        img_shape: image shape, like (h, w).
    """
    img_h, img_w = img_shape
    (
        uv_undist_nor,
        _,
        _,
        gamma_list,
        high_list,
        depth_list,
    ) = get_ori_gamma(data_root, pack_name, ids, K, R, t, N, camH)

    def _points_filter(gt_list, gt, gt_type):
        MINMAX = {
            "gamma": (-0.1, 0.4),
            "height": (-0.2, 5),
            "depth": (-0.2, 256),
        }
        gt_min = MINMAX[gt_type][0]
        gt_max = MINMAX[gt_type][1]
        if h >= img_h or w >= img_w:
            pass
        elif gt_list[i] > gt_max:
            gt[0][h][w] = gt_max
        elif gt_list[i] < gt_min:
            gt[0][h][w] = gt_min
        else:
            gt[0][h][w] = gt_list[i]
        return gt

    img_gamma = np.ones((1, img_h, img_w), dtype=np.float32) * (-1)
    img_high = np.ones((1, img_h, img_w), dtype=np.float32) * (-1)
    img_depth = np.ones((1, img_h, img_w), dtype=np.float32) * (-1)
    for i in range(len(uv_undist_nor)):
        pt = uv_undist_nor[i][:2]
        w = int(np.round(pt[0]))
        h = int(np.round(pt[1]))
        img_gamma = _points_filter(
            gamma_list,
            img_gamma,
            gt_type="gamma",
        )
        img_high = _points_filter(
            high_list,
            img_high,
            gt_type="height",
        )
        img_depth = _points_filter(
            depth_list,
            img_depth,
            gt_type="depth",
        )
    return img_gamma, img_high, img_depth


def get_gt(
    data_root: str,
    pack_name: str,
    ids: str,
):
    """Generate gt maps from img.

    Args:
        data_root: data root.
        pack_name: pack name.
        ids: timestamp
    """
    depth_path = os.path.join(data_root, pack_name, "depth_gt", ids + ".png")
    height_path = os.path.join(data_root, pack_name, "height_gt", ids + ".png")
    gamma_path = os.path.join(data_root, pack_name, "gamma_gt", ids + ".png")
    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    height_gt = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
    gamma_gt = cv2.imread(gamma_path, cv2.IMREAD_UNCHANGED)

    depth_gt = depth_gt.astype(np.float) / DEPTH_COVER_SCALE - 1.0
    height_gt = height_gt.astype(np.float) / HEIGHT_COVER_SCALE - 1.0
    gamma_gt = gamma_gt.astype(np.float) / GAMMA_COVER_SCALE - 1.0
    return depth_gt, height_gt, gamma_gt


def get_image_path(
    data_root: str, pack_name: str, name: str, postfix: str = ".jpg"
):
    """Get image path.

    Args:
        data_root: data root.
        pack_name: pack name.
        name: timestamp.
        postfix: img postfix, '.jpg' or '.png'.
    """
    image_path = os.path.join(data_root, pack_name, name + postfix)
    assert os.path.exists(image_path), image_path
    return image_path


def get_image(
    data_root: str,
    img_path: str,
    name: str,
    postfix: str,
    mode: str = "RGB",
    size: Sequence = None,
):
    """Get image.

    Args:
        data_root: data root.
        img_path: image path.
        name: timestamp.
        postfix: img postfix, '.jpg' or '.png'.
        mode: image mode.
        size: image size.
    """
    image = pil_loader(
        get_image_path(data_root, img_path, name, postfix), mode, size=size
    )
    return image


def gt_processer(gt: np.ndarray):
    # parsing gt
    gamma_mask = gt[:, :, 0:1]
    height_img = gt[:, :, 1:2] / HEIGHT_COVER_SCALE - 1
    depth_img = gt[:, :, 2:3] / DEPTH_COVER_SCALE - 1

    # set no pointcloud -1
    gamma_img = height_img / depth_img
    gamma_img[gamma_mask == 0] = -1
    height_img[gamma_mask == 0] = -1
    depth_img[gamma_mask == 0] = -1

    return gamma_img, height_img, depth_img, gamma_mask


# TODO: () AutoElevation will be deprecated when rec data enough.
@OBJECT_REGISTRY.register
class Elevation(data.Dataset):
    """A dataset for generating data to support elevation task.

    Args:
        root: Path of data.
        ann_file: root of annotation.
        list_file: frame stamps file.
        cfg_file: file of a config file, the content of which are
            parameters file path.
        source: yaml file of parameters.
        gt_source: the gt source, default is 'img'.
        transforms: List of transform.
        load_data_types: which type data to load.
        img_shape: image shape
        load_all_cfg: if load all cfg one time, set 'True' in old
            version data.

    """

    def __init__(
        self,
        root: str,
        ann_root: str,
        list_file: str,
        cfg_file: str,
        source: str,
        gt_source: str,
        transforms: Optional[Sequence] = None,
        load_data_types: Optional[Sequence] = None,
        img_shape: Optional[Sequence] = None,
        load_all_cfg: bool = True,
    ):
        self.source = source
        self.gt_source = gt_source
        self.transforms = transforms
        self.load_data_types = load_data_types
        self.img_shape = img_shape

        self.data_root = root
        self.ann_root = ann_root
        self.load_all_cfg = load_all_cfg

        if self.load_all_cfg:
            self.cfg_dict = load_cfg(ann_root, root, cfg_file)
        self.file_list = load_list(ann_root, list_file)

        if self.gt_source == "img":
            self.folder = "FRONT_undist"
        else:
            self.folder = "FRONT"
        self.attr_dict = {}

    def load_cam_pos(self, data_root: str, pack_name: str, filename: str):
        """Load camera pos for each pack.

        Args:
            data_root: data root.
            pack_name: pack name.
            filename: timestamp.
        """
        if pack_name in self.attr_dict.keys():
            R_l2c = self.attr_dict[pack_name]["R_l2c"]
            t_l2c = self.attr_dict[pack_name]["t_l2c"]
        else:
            R_l2c, t_l2c = load_lidar2camere(data_root, pack_name, filename)
            self.attr_dict[pack_name] = {"R_l2c": R_l2c, "t_l2c": t_l2c}
        return R_l2c, t_l2c

    def load_elevation_imgs(
        self,
        data_root: str,
        img_path: str,
        img_name: str,
    ):
        """Load images.

        Args:
            data_root: data root.
            img_path: image path.
            img_name: image names.
        """
        pil_imgs = [
            [get_image(data_root, img_path, name, ".png", "RGB")]
            for name in img_name
        ]
        masks = [
            [
                get_image(
                    data_root,
                    img_path.replace(self.folder, "SEGMENT"),
                    name,
                    "_label.png",
                    "F",
                )
            ]
            for name in img_name
        ]
        return pil_imgs, masks

    def load_elevation_params(
        self,
        pack_name: str,
        filename: str,
    ):
        """Load parameters.

        Args:
            pack_name: pack name.
            filename: timestamp.
        """
        if self.load_all_cfg:
            cfg = self.cfg_dict[pack_name][filename]
        else:
            # TODO: load cfg of index data
            pass
        K, R_p2c, T_p2c, N_c, camH_c, H_p2c = load_parameters(cfg)
        return K, R_p2c, T_p2c, N_c, camH_c, H_p2c

    def load_elevation_gt(
        self,
        data_root: str,
        img_path: str,
        cur_id: str,
        K: np.ndarray,
        N_c: np.ndarray,
        camH_c: float,
    ):
        """Load camera pos for each pack.

        Args:
            data_root: data root.
            img_path: image path.
            cur_id: current frame timestamp.
            K: camera intrinsics.
            N_c: ground norm of current frame.
            camH_c: camera height of current frame.
        """
        if self.gt_source == "img":
            depth_cur, height_cur, gamma_cur = get_gt(
                data_root, img_path.replace(self.folder, ""), cur_id
            )
        else:
            R_l2c, t_l2c = self.load_cam_pos(
                data_root, img_path.replace(self.folder, ""), "attribute.json"
            )
            gamma_cur, height_cur, depth_cur = get_depth_height_gamma(
                data_root,
                img_path.replace(self.folder, "pc_with_project"),
                cur_id,
                K,
                R_l2c,
                t_l2c,
                N_c,
                camH_c,
                self.img_shape,
            )
        return gamma_cur, height_cur, depth_cur

    def load_data(self, index):
        # load imgs id
        (
            pack_name,
            fro_id,
            cur_id,
            pre_id,
            _,
        ) = self.file_list[index]
        data_root = self.data_root
        img_path = os.path.join(pack_name, self.folder)
        img_name = [cur_id, pre_id, fro_id]  # t,t-1,t+1
        # load parameters of current img
        K, R_p2c, T_p2c, N_c, camH_c, H_p2c = self.load_elevation_params(
            pack_name, cur_id
        )
        # load imgs and masks
        pil_imgs, masks = self.load_elevation_imgs(
            data_root, img_path, img_name
        )
        # load gt of cur img
        gamma_cur, height_cur, depth_cur = self.load_elevation_gt(
            data_root,
            img_path,
            cur_id,
            K,
            N_c,
            camH_c,
        )
        imgs = (pil_imgs, masks)
        gts = (gamma_cur, height_cur, depth_cur)
        params = (K, R_p2c, T_p2c, N_c, camH_c, H_p2c)
        timestamp = np.array([img_name[0]], dtype="float64") / 1000
        return imgs, gts, params, timestamp

    def get_sample_data(
        self,
        imgs: Sequence,
        gts: Sequence,
        params: Sequence,
        timestamp: Sequence,
    ):
        """Load data sample.

        Args:
            imgs: pil_imgs, masks.
            gts: gamma gt, height gt, depth gt.
            params: current frame params, contains K,
                R_p2c, T_p2c, N_c, camH_c, H_p2c.
            timestamp: current timestamp.
        """
        pil_imgs, masks = imgs
        gamma_cur, height_cur, depth_cur = gts
        K, R_p2c, T_p2c, N_c, camH_c, H_p2c = params

        data_dict = {}
        data_dict["pil_imgs"] = pil_imgs
        if "ground_homo" in self.load_data_types:
            data_dict["ground_homo"] = [H_p2c.copy()]
        if "rotation" in self.load_data_types:
            data_dict["rotation"] = [R_p2c.copy()]
        if "transition" in self.load_data_types:
            data_dict["transition"] = [T_p2c.copy()]
        if "intrinsics" in self.load_data_types:
            data_dict["intrinsics"] = normalize_K(K.copy(), self.img_shape)
        if "gt_depth" in self.load_data_types:
            data_dict["gt_depth"] = [depth_cur]
        if "gt_height" in self.load_data_types:
            data_dict["gt_height"] = [height_cur]
        if "gt_gamma" in self.load_data_types:
            data_dict["gt_gamma"] = [gamma_cur]
        if "ground_norm" in self.load_data_types:
            data_dict["ground_norm"] = [N_c]
        if "mask" in self.load_data_types:
            data_dict["mask"] = masks
        if "obj_mask" in self.load_data_types:
            assert (
                "mask" in self.load_data_types
            ), "'obj_mask' rely on 'mask',  \
                plz add 'mask' in load_data_types"
            data_dict["obj_mask"] = None
        if "ground_mask" in self.load_data_types:
            # ground mask for homographynet and end2end
            assert (
                "mask" in self.load_data_types
            ), "'ground_mask' rely on 'mask', \
                plz add 'mask' in load_data_types"
            data_dict["ground_mask"] = None
        if "camera_high" in self.load_data_types:
            data_dict["camera_high"] = [camH_c]
        if "timestamp" in self.load_data_types:
            data_dict["timestamp"] = timestamp
        return data_dict

    def __getitem__(self, index):
        try:
            imgs, gts, params, timestamp = self.load_data(index)
            data_dict = self.get_sample_data(imgs, gts, params, timestamp)

            if self.transforms is not None:
                data_dict = self.transforms(data_dict)
            return data_dict
        except KeyError:
            print("Parameters of index '{}' does exist".format(index))
            if index >= len(self.file_list):
                index = 0
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.file_list)

    def __repr__(self):
        return "Elevation dataset"


class ReaderElevation(object):
    """Read data from origin file, rec or lmdb.

    Args:
        img_rec_path: path of image rec.
        seg_rec_path: path of seg rec.
        gt_rec_path: path of gt rec.
        ele_lmdb_path: path of elevation lmdb.
        intrinsics_lmdb_path: path of intrinsics lmdb.
        pack2idx_path: mapping of pack to index.
        gt_shape: shape of ground true, (1080, 1920, 3).
    """

    def __init__(
        self,
        root: str,
        img_rec_path: Optional[Union[str, Sequence[str]]] = None,
        seg_rec_path: Optional[Union[str, Sequence[str]]] = None,
        gt_rec_path: Optional[Union[str, Sequence[str]]] = None,
        ele_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        intrinsics_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        pack2idx_path: Optional[Union[str, Sequence[str]]] = None,
        gt_shape: Optional[Sequence] = None,
    ):
        self.root = root
        self.image_lst, self.image_rec = self._load_rec(img_rec_path)
        self.seg_lst, self.seg_rec = self._load_rec(seg_rec_path)
        self.gt_lst, self.gt_rec = self._load_rec(gt_rec_path)

        self.pack2idx = self.read_pack2idx(pack2idx_path)
        self.ele_lmdb_path = ele_lmdb_path
        self.intrinsics_lmdb_path = intrinsics_lmdb_path

        if self.ele_lmdb_path:
            self.ele_lmdb = self._load_lmdb(ele_lmdb_path)
        if self.intrinsics_lmdb_path:
            self.intrinsics_lmdb = self._load_lmdb(intrinsics_lmdb_path)

        self.gt_shape = gt_shape

    def _load_rec(self, rec_path: str):
        if rec_path:
            idx_path = rec_path.replace(".rec", ".idx").replace(
                "test_idx_and_lmdb", "test_rec_and_lmdb"
            )
            lst_path = rec_path.replace(".rec", ".lst").replace(
                "test_lst_and_lmdb", "test_rec_and_lmdb"
            )
            image_lst = self.read_lst(lst_path)
            image_rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
        else:
            image_lst, image_rec = None, None
        return image_lst, image_rec

    def _load_lmdb(self, lmdb_path: str):
        lmdb = Lmdb(lmdb_path, False, True, readonly=True)
        return lmdb

    def __getstate__(self):
        state = self.__dict__
        state["ele_lmdb"] = None
        state["intrinsics_lmdb"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.ele_lmdb_path is not None:
            self.ele_lmdb = Lmdb(
                self.ele_lmdb_path, False, True, readonly=True
            )
        if self.intrinsics_lmdb_path:
            self.intrinsics_lmdb = Lmdb(
                self.intrinsics_lmdb_path, False, True, readonly=True
            )

    def read_lst(self, lst_path: str):
        # Read the lst (.lst) file
        image_lst = {}
        with open(lst_path, "r") as fin:
            for line in iter(fin.readline, ""):
                try:
                    line = line.decode("utf-8")
                except AttributeError:
                    pass
                line = line.strip().split("\t")
                image_index = line[0]
                image_name = line[-1]
                assert image_name not in image_lst
                image_lst[image_name] = int(image_index)
        return image_lst

    def read_pack2idx(self, pack2idx_path: str):
        # Get mapping info of pack, use to load intrinsics.
        with open(pack2idx_path, "r") as f:
            pack2idx = yaml.load(f, Loader=yaml.SafeLoader)
        return pack2idx

    def get_rec_img(self, rec: str, idx: str):
        _, img = mx.recordio.unpack_img(rec.read_idx(idx), cv2.IMREAD_COLOR)
        return img

    def get_rec_gt(self, rec: str, idx: str):
        header, img = mx.recordio.unpack(rec.read_idx(idx))
        img = header.label.reshape(self.gt_shape)
        return img

    def select_rec_by_mode(self, mode: str):
        if mode == "RGB":
            rec = self.image_rec
            lst = self.image_lst
        elif mode == "F":
            rec = self.seg_rec
            lst = self.seg_lst
        return rec, lst

    def get_img(
        self,
        pack_name: str,
        img_name: str,
        prefix: str,
        postfix: str,
        mode="RGB",
        size=None,
    ):
        rec, lst = self.select_rec_by_mode(mode)
        if rec:
            img_key = os.path.join(pack_name, prefix, img_name + postfix)
            image_np = self.get_rec_img(rec, lst[img_key])
            image_np = image_np[:, :, ::-1]
            image = Image.fromarray(image_np).convert(mode)
        else:
            image = pil_loader(
                get_image_path(
                    self.root, pack_name + "/" + prefix, img_name, postfix
                ),
                mode,
                size=size,
            )
        return image

    def get_gt(self, pack_name: str, img_name: str, prefix: str, postfix: str):
        # postfix = '.png'
        if self.gt_rec:
            frame_path = os.path.join(pack_name, prefix, img_name + postfix)
            gt = self.get_rec_gt(self.gt_rec, self.gt_lst[frame_path])
        else:
            frame_path = os.path.join(
                self.root, pack_name, prefix, img_name + postfix
            )
            gt = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        return gt

    def get_ground(self, idx: int):
        # idx should be same as image list.
        data = self.ele_lmdb.read(idx).decode()
        ground_dict = json.loads(data)
        return ground_dict

    def get_intrinsics(self, pack_name: str):
        idx = self.pack2idx[pack_name]
        data = self.intrinsics_lmdb.read(idx).decode()
        intrinsics_dict = json.loads(data)
        return intrinsics_dict


@OBJECT_REGISTRY.register
class ElevationRec(data.Dataset):
    """A dataset for generating data to support elevation task.

    Args:
        root: Path of data.
        ann_file: root of annotation.
        list_file: frame stamps file.
        img_rec_path: path of image rec.
        seg_rec_path: path of seg rec.
        gt_rec_path: path of gt rec.
        ele_lmdb_path: path of elevation lmdb.
        intrinsics_lmdb_path: path of intrinsics lmdb.
        pack2idx_path: mapping of pack to index.
        transforms: List of transform.
        load_data_types: which type data to load.
        load_all_cfg: if load all cfg one time, set 'True' in old
            version data.
        img_shape: image shape.
        gt_shape: ground truth shape, 3 channel for rec format.
    """

    def __init__(
        self,
        root: Optional[Union[str, Sequence[str]]] = None,
        ann_root: Optional[Union[str, Sequence[str]]] = None,
        list_file: Optional[Union[str, Sequence[str]]] = None,
        img_rec_path: Optional[Union[str, Sequence[str]]] = None,
        seg_rec_path: Optional[Union[str, Sequence[str]]] = None,
        gt_rec_path: Optional[Union[str, Sequence[str]]] = None,
        ele_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        intrinsics_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        pack2idx_path: Optional[Union[str, Sequence[str]]] = None,
        transforms: Optional[Sequence] = None,
        load_data_types: Optional[Sequence] = None,
        img_shape: Optional[Sequence] = None,
        gt_shape: Optional[Sequence] = None,
    ):
        self.transforms = transforms
        self.load_data_types = load_data_types
        self.img_shape = img_shape
        self.gt_shape = gt_shape

        self.data_root = root
        self.ann_root = ann_root
        self.file_list = load_list(ann_root, list_file)
        self.reader_ele = ReaderElevation(
            root,
            img_rec_path,
            seg_rec_path,
            gt_rec_path,
            ele_lmdb_path,
            intrinsics_lmdb_path,
            pack2idx_path,
            gt_shape=self.gt_shape,
        )

    def load_elevation_imgs(
        self,
        pack_name: str,
        img_name: str,
    ):
        """Load images.

        Args:
            pack_name: pack_name (image path).
            img_name: image names.
        """
        pil_imgs = [
            [
                self.reader_ele.get_img(
                    pack_name,
                    name,
                    prefix="camera_front",
                    postfix=".jpg",
                    mode="RGB",
                )
            ]
            for name in img_name
        ]
        masks = []
        if "mask" in self.load_data_types:

            masks = [
                [
                    self.reader_ele.get_img(
                        pack_name,
                        name,
                        prefix="seg_camera_front",
                        postfix=".png",
                        mode="F",
                    )
                ]
                for name in img_name
            ]
        return pil_imgs, masks

    def load_elevation_params(
        self,
        pack_name: str,
        index: str,
    ):
        """Load parameters.

        Args:
            pack_name: pack_name (image path).
            cur_id: timestamp of current image.
        """

        ground_dict = self.reader_ele.get_ground(index)
        intrinsics_dict = self.reader_ele.get_intrinsics(pack_name)

        K, R_p2c, T_p2c, N_c, camH_c, H_p2c = load_parameters_lmdb(
            ground_dict, intrinsics_dict, invert=False
        )
        return K, R_p2c, T_p2c, N_c, camH_c, H_p2c

    def load_elevation_gt(
        self,
        pack_name: str,
        cur_id: str,
    ):
        """Load ground truth depth, height, gamma for each current img.

        Args:
            pack_name: pack_name (image path).
            cur_id: current frame timestamp.
        """
        gt = self.reader_ele.get_gt(
            pack_name, cur_id, prefix="dense_ele_camera_front", postfix=".png"
        )
        gamma_img, height_img, depth_img, _ = gt_processer(gt)
        return gamma_img, height_img, depth_img

    def load_data(self, index):
        # load imgs id
        (
            pack_name,
            fro_id,
            cur_id,
            pre_id,
            _,
        ) = self.file_list[index]
        img_name = [cur_id, pre_id, fro_id]  # t,t-1,t+1
        # load imgs and masks
        pil_imgs, masks = self.load_elevation_imgs(pack_name, img_name)
        # load parameters of current img
        K, R_p2c, T_p2c, N_c, camH_c, H_p2c = self.load_elevation_params(
            pack_name, index
        )
        # load gt of cur img
        gamma_cur, height_cur, depth_cur = self.load_elevation_gt(
            pack_name,
            cur_id,
        )
        imgs = (pil_imgs, masks)
        gts = (gamma_cur, height_cur, depth_cur)
        params = (K, R_p2c, T_p2c, N_c, camH_c, H_p2c)
        timestamp = np.array([img_name[0]], dtype="float64") / 1000
        return imgs, gts, params, timestamp

    def get_sample_data(
        self,
        imgs: Sequence,
        gts: Sequence,
        params: Sequence,
        timestamp: Sequence,
    ):
        """Load data sample.

        Args:
            imgs: pil_imgs, masks.
            gts: gamma gt, height gt, depth gt.
            params: current frame params, contains K,
                R_p2c, T_p2c, N_c, camH_c, H_p2c.
            timestamp: current timestamp.
        """
        pil_imgs, masks = imgs
        gamma_cur, height_cur, depth_cur = gts
        K, R_p2c, T_p2c, N_c, camH_c, H_p2c = params

        data_dict = {}
        data_dict["pil_imgs"] = pil_imgs

        if "ground_homo" in self.load_data_types:
            data_dict["ground_homo"] = [H_p2c.copy()]
        if "rotation" in self.load_data_types:
            data_dict["rotation"] = [R_p2c.copy()]
        if "transition" in self.load_data_types:
            data_dict["transition"] = [T_p2c.copy()]
        if "intrinsics" in self.load_data_types:
            data_dict["intrinsics"] = normalize_K(
                K.copy(), (self.img_shape[0], self.img_shape[1])
            )
        if "gt_depth" in self.load_data_types:
            data_dict["gt_depth"] = [depth_cur]
        if "gt_height" in self.load_data_types:
            data_dict["gt_height"] = [height_cur]
        if "gt_gamma" in self.load_data_types:
            data_dict["gt_gamma"] = [gamma_cur]
        if "ground_norm" in self.load_data_types:
            data_dict["ground_norm"] = [N_c]
        if "mask" in self.load_data_types:
            data_dict["mask"] = masks
        if "obj_mask" in self.load_data_types:
            assert (
                "mask" in self.load_data_types
            ), "'obj_mask' rely on 'mask',  \
                plz add 'mask' in load_data_types"
            data_dict["obj_mask"] = None
        if "ground_mask" in self.load_data_types:
            # ground mask for homographynet and end2end
            assert (
                "mask" in self.load_data_types
            ), "'ground_mask' rely on 'mask', \
                plz add 'mask' in load_data_types"
            data_dict["ground_mask"] = None
        if "camera_high" in self.load_data_types:
            data_dict["camera_high"] = [camH_c]
        if "timestamp" in self.load_data_types:
            data_dict["timestamp"] = timestamp
        return data_dict

    def __getitem__(self, index):
        try:
            imgs, gts, params, timestamp = self.load_data(index)
            data_dict = self.get_sample_data(imgs, gts, params, timestamp)

            if self.transforms is not None:
                data_dict = self.transforms(data_dict)
            return data_dict
        except KeyError:
            print("Parameters of index '{}' does exist".format(index))
            if index >= len(self.file_list):
                index = 0
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.file_list)

    def __repr__(self):
        return "ElevationRec"


@OBJECT_REGISTRY.register
class ElevationFromImage(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.

    Args:
        data_path: The path where the image is stored.
        camera_view_names: which view to load.
        img_load_size: size of image to load.
        att_json_path: path of attribute.json.
        transforms: List of transform.

    """

    def __init__(
        self,
        data_path: str,
        camera_view_names: Optional[Sequence] = None,
        img_load_size: Optional[Sequence] = None,
        att_json_path: Optional[Sequence] = None,
        transforms: Optional[Sequence] = None,
    ):
        self.data_path = os.path.split(data_path)[
            0
        ]  # strip the `camera_view` level of the path hierarchy  # noqa
        self.transforms = transforms
        self.att_json_path = att_json_path
        self.img_load_size = img_load_size
        self.camera_view_names = camera_view_names
        self.front_img_name = camera_view_names[0]
        assert self.front_img_name in ["fisheye_front", "camera_front"]
        self.collect_samples()

    def collect_samples(self):
        """Create data samples from data_path."""

        postfix = ".jpg"
        att_json_path = os.path.join(self.data_path, "attribute.json")
        assert os.path.exists(att_json_path)
        att_file = json.load(open(att_json_path))

        self.sample_lines = []
        sync_list = att_file["sync"]
        for i in range(len(sync_list[self.front_img_name])):
            file_list_before = []
            file_list_current = []
            for camera_view_name in self.camera_view_names:
                # get index in imgs_list according sync_list
                _time_stamp = sync_list[camera_view_name][i]
                unsync_view_index = sync_list[camera_view_name].index(
                    _time_stamp
                )
                if unsync_view_index < len(sync_list[camera_view_name]):
                    file_list_before.append(
                        str(sync_list[camera_view_name][unsync_view_index - 1])
                        + postfix
                    )
                    file_list_current.append(
                        str(sync_list[camera_view_name][unsync_view_index])
                        + postfix
                    )

            self.front_idx = self.camera_view_names.index(self.front_img_name)
            self.sample_lines.append(
                [
                    file_list_before[self.front_idx],
                    file_list_current[self.front_idx],
                ]
            )

    def get_image_path(self, img_dir, name, postfix=".jpg"):
        image_path = os.path.join(self.data_path, img_dir, name + postfix)
        assert os.path.exists(image_path), image_path
        return image_path

    def get_image(self, img_dir, name, postfix, mode="RGB", size=None):
        image = pil_loader(
            self.get_image_path(img_dir, name, postfix), mode, size=size
        )
        return image

    def __len__(self):
        return len(self.sample_lines)

    def __getitem__(self, index):
        data_dict = {}
        img_names = self.sample_lines[index]

        data_dict["pil_imgs"] = []
        for img_names_i in img_names:
            imgs = [
                self.get_image(
                    self.front_img_name,
                    img_names_i,
                    "",
                    "RGB",
                    size=self.img_load_size,
                )
            ]
            data_dict["pil_imgs"].append(imgs)

        data_dict["img_name"] = _as_list(img_names[1])[0]

        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __repr__(self):
        return "ElevationFromImage"
