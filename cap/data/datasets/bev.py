# Copyright (c) Changan Auto. All rights reserved.
import json
import os
from typing import Mapping, Optional, Sequence

import cv2
import fsspec
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from cap.version import check_version

__all__ = ["HomoGenerator"]

VALID_Z = 0.05


class HomoGenerator(object):  # noqa: D205,D400
    """Generate homography by online compute or offline load.

    Note that vcs_range (bottom, right, top, left) in order, ipm use
    (front, back, left, right) in order. So change vcs_range order to
    (front, back, left, right).

    Args:
        calib_path: path of lidar and camera file, include calibration
            params files like,

            camera_front_left.yaml,
            camera_front_right.yaml,
            camera_front.yaml,
            camera_rear_left.yaml,
            camera_rear_right.yaml,
            camera_rear.yaml,
            log_trans,
            calibration.json,

            note that homography can be computed by xxx.ymal + log_trans
            files or just calibration.json file. If both calibration.json
            and xxx.ymal + log_trans provided, use calibration.json to
            compute homography. Also, calibration.json includes all camera
            intrinsic/extrinsic params, lidar2vcs and lidar2cam for
            compute homography. Refer to sda data system for details of
            calibration.json file of different vehicle, e.g.,


        homo_path: path of homograph matrix npy file.
        spatial_resolution: bev spatial resolution.(unit is meters)
        vcs_range: visbile range of bev, (bottom, right, top, left)
            in order.
        camera_view_names: sub directory name of each view,
        per_view_shape: original img shape correspond to each camera view.  # noqa
        per_view_shape_for_offset: img shape correspond to each camera
            view when calculate homography offset.
        use_distorted_offset: return distorted homography offset or undistorted.  # noqa
        norm_homo: whether to normalize homography matrix, default true.
        homo_transforms: dict of homo transforms,
        Describing the changes from origin img to the input of Bevfusion.  # noqa
            each veiw in Homo tansform is like this
        .. code-block:: JSON

        {
            Resize: (H,W)
            Crop: (top,left)
            Pad: (left, top, right and bottom)
            ResizeHomo: (scale)
        }
    """

    def __init__(
        self,
        calib_path: str,
        homo_path: str,
        spatial_resolution: Sequence[float],
        vcs_range: Sequence[float],
        camera_view_names: Sequence[str],
        per_view_shape: Mapping,
        homo_transforms: Mapping,
        norm_homo: bool = True,
        per_view_shape_for_offset: Optional[Mapping] = None,
        use_distorted_offset: bool = False,
    ):
        self.calib_path = calib_path
        self.homo_path = homo_path
        self.spatial_resolution = spatial_resolution
        self.camera_view_names = camera_view_names
        self.per_view_shape = per_view_shape
        self.per_view_shape_for_offset = per_view_shape_for_offset
        self.use_distorted_offset = use_distorted_offset
        self.homo_transforms = homo_transforms
        # (bottom, right, top, left) --> (front, back, left, right)
        self.vcs_range = (
            vcs_range[2],
            vcs_range[0],
            vcs_range[3],
            vcs_range[1],
        )
        self.norm_homo = norm_homo
        self.height = int(
            abs(vcs_range[2] - vcs_range[0]) / spatial_resolution[0]
        )  # bev image height
        self.width = int(
            abs(vcs_range[3] - vcs_range[1]) / spatial_resolution[1]
        )  # bev image width

        self.dist_homo_offset = None

    @staticmethod
    def _build_transform_ipm2egognd(
        camera_view_name, spatial_resolution, vcs_range
    ):
        """Get the ipm matrix."""
        view_vcs_roi = np.array(
            [
                vcs_range[3],
                vcs_range[0],
                vcs_range[2],
                vcs_range[1],
            ]
        )
        if "front" in camera_view_name:
            view_vcs_roi[3] = spatial_resolution[0] / 2
        if "rear" in camera_view_name:
            view_vcs_roi[1] = -spatial_resolution[0] / 2
        if "left" in camera_view_name:
            view_vcs_roi[0] = spatial_resolution[1] / 2
        if "right" in camera_view_name:
            view_vcs_roi[2] = -spatial_resolution[1] / 2

        view_ipm_roi_0 = (vcs_range[2] - view_vcs_roi[0]) / spatial_resolution[
            1
        ]  # y->u right
        view_ipm_roi_1 = (vcs_range[0] - view_vcs_roi[1]) / spatial_resolution[
            0
        ]  # x->v top
        view_ipm_roi_2 = (vcs_range[2] - view_vcs_roi[2]) / spatial_resolution[
            1
        ]  # y->u left
        view_ipm_roi_3 = (vcs_range[0] - view_vcs_roi[3]) / spatial_resolution[
            0
        ]  # x->v down

        gnd_pt1 = [view_vcs_roi[1], view_vcs_roi[2]]
        gnd_pt2 = [view_vcs_roi[1], view_vcs_roi[0]]
        gnd_pt3 = [view_vcs_roi[3], view_vcs_roi[2]]
        gnd_pt4 = [view_vcs_roi[3], view_vcs_roi[0]]

        ipm_pt1 = [view_ipm_roi_2, view_ipm_roi_1]
        ipm_pt2 = [view_ipm_roi_0, view_ipm_roi_1]
        ipm_pt3 = [view_ipm_roi_2, view_ipm_roi_3]
        ipm_pt4 = [view_ipm_roi_0, view_ipm_roi_3]

        gnd_region = [gnd_pt1, gnd_pt2, gnd_pt3, gnd_pt4]
        ipm_region = [ipm_pt1, ipm_pt2, ipm_pt3, ipm_pt4]

        T_ipm2vcsgnd = cv2.getPerspectiveTransform(
            np.array(ipm_region).astype(np.float32),
            np.array(gnd_region).astype(np.float32),
        ).astype("float32")
        return T_ipm2vcsgnd

    @staticmethod
    def _get_T_lidar2vcs(
        lidar_param_file: str,
        camera_param_file: str,
    ):
        """Get lidar to vcs transform.

        Lidar param file is log_tran file and camera param file is yaml file.
        If both param files contain lidar2vcs, prefer to use lidar2vcs in
        camera param file.
        """
        with fsspec.open(camera_param_file, "r") as fid:
            content = fid.read()
        camera_params = yaml.safe_load(content)
        if "lidar2vcs" in camera_params.keys():
            lidar2vcs = np.array(camera_params["lidar2vcs"]).reshape((2, 3))
        else:
            with open(lidar_param_file, "r") as f:
                lidar2vcs = f.readlines()
                for id, v in enumerate(lidar2vcs):
                    if "lidar2vcs" in v:
                        rpy_file = lidar2vcs[id + 1]
                        xyz_file = lidar2vcs[id + 2]
                        break

            rpy = rpy_file.split()
            xyz = xyz_file.split()
            lidar2vcs = np.zeros((2, 3))
            for idx, e in enumerate(rpy[1:]):
                lidar2vcs[0, idx] = float(e.split(",")[0])
            for idx, e in enumerate(xyz[1:]):
                lidar2vcs[1, idx] = float(e.split(",")[0])

        return lidar2vcs

    def _get_calib_parameters(
        self,
        camera_view_name: str,
        lidar_param_file: str,
        camera_param_file: str,
        calib_param_file: str = None,
    ):
        """Get calib params such as lidar extrinsics and camera intrinsics.

        Note that lidar extrinsics include lidar2vcs/lidar2cam transforms,
        camera intrinsics include K and distort coefficients.

        Args:
            camera_view_name str: camera view name, e.g., "camera_front".
            lidar_param_file str: params file for lidar, e.g.,
                "path/to/log_trans".
            camera_param_file str: params file for camera, e.g.,
                "path/to/camera_front.yaml".
            calib_param_file str: a calibration params json file, e.g.,
                "xxx/calibration.json", contains calibration parameters
                for lidar and cameras, refer to calibration params version
                in sda data system.

        Returns:
            list array: lidar2vcs, lidar2cam, camera K and distort coeffs.
        """
        if calib_param_file is not None and os.path.exists(calib_param_file):
            # unify camera name in calib_param_file
            camera_view_rename = {
                "camera_front_left": "camera_frontleft",
                "camera_front": "camera_front",
                "camera_front_right": "camera_frontright",
                "camera_rear_left": "camera_rearleft",
                "camera_rear": "camera_rear",
                "camera_rear_right": "camera_rearright",
                "fisheye_front": "camera_fisheye_front",
                "fisheye_rear": "camera_fisheye_rear",
                "fisheye_left": "camera_fisheye_left",
                "fisheye_right": "camera_fisheye_right",
            }
            with open(calib_param_file, "r") as f:
                calib_param = json.load(f)

            # The calibration.json file may contains only "lidar_top_2_vcs" or
            # "lidar_2_vcs" or both.
            # (1) if "lidar_2_vcs" only, only use "lidar_2_vcs"
            # (2) if "lidar_2_vcs" and "lidar_top_2_vcs" both exists, the xyz
            #     in both is different, we use "lidar_top_2_vcs" to keep same
            #     to attribute.json file.
            lidar2vcs = calib_param.get("lidar_top_2_vcs", "lidar_2_vcs")
            R_lidar2vcs = (
                Rotation.from_euler(
                    "xyz",
                    lidar2vcs["rpy"],
                )
            ).as_matrix()
            t_lidar2vcs = np.array(
                lidar2vcs["xyz"],
            )
            T_lidar2cam = np.array(
                calib_param[
                    f"lidar_top_2_{camera_view_rename[camera_view_name]}"
                ]
            )
            K = np.array(
                calib_param[camera_view_rename[camera_view_name]]["K"]
            )
            d_coef = np.array(
                calib_param[camera_view_rename[camera_view_name]]["d"]
            )
        else:
            assert os.path.exists(lidar_param_file), lidar_param_file
            assert os.path.exists(camera_param_file), camera_param_file

            T_lidar2vcs_file = self._get_T_lidar2vcs(
                lidar_param_file, camera_param_file
            )
            with open(camera_param_file, "r") as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            R_lidar2vcs = (
                Rotation.from_euler(
                    "xyz",
                    [
                        T_lidar2vcs_file[0, 0],
                        T_lidar2vcs_file[0, 1],
                        T_lidar2vcs_file[0, 2],
                    ],
                )
            ).as_matrix()
            t_lidar2vcs = np.array(
                [
                    T_lidar2vcs_file[1, 0],
                    T_lidar2vcs_file[1, 1],
                    T_lidar2vcs_file[1, 2],
                ]
            )

            # lidar to camera
            R0_lidar2cam = params["extrinsic"]["rotation0"]
            t0_lidar2cam = params["extrinsic"]["translation0"]

            R_lidar2cam = Rotation.from_quat(
                [
                    R0_lidar2cam["q.x"],
                    R0_lidar2cam["q.y"],
                    R0_lidar2cam["q.z"],
                    R0_lidar2cam["q.w"],
                ]
            ).as_matrix()
            t_lidar2cam = np.array(
                [t0_lidar2cam["x"], t0_lidar2cam["y"], t0_lidar2cam["z"]]
            )
            T_lidar2cam = np.zeros((4, 4))
            T_lidar2cam[:3, :3] = R_lidar2cam
            T_lidar2cam[3, 3] = 1
            T_lidar2cam[:3, 3] = t_lidar2cam

            K0 = params["intrinsic"]["cameraMatrix0"]
            d_coef = np.array(K0["distCoeffs"])
            K = np.array(
                [[K0["ax"], 0, K0["ux"]], [0, K0["ay"], K0["uy"]], [0, 0, 1]]
            )

        T_lidar2vcs = np.zeros((4, 4))
        T_lidar2vcs[:3, :3] = R_lidar2vcs
        T_lidar2vcs[:3, 3] = t_lidar2vcs
        T_lidar2vcs[3, 3] = 1

        return T_lidar2vcs, T_lidar2cam, K, d_coef

    def compute_homography(
        self,
        camera_view_name,
        K,
        T_lidar2vcs,
        T_lidar2cam,
    ):
        """Compute homography by calibration params for one camera view.

        Args:
            camera_view_name str: camera view name, e.g., "camera_front".
            K array: camera intrinsic K matrix.
            T_lidar2vcs array: transform of lidar to vcs.
            T_lidar2cam array: transform of lidar to camera.

        Returns:
            numpy array: homography.
        """
        T_vcs2cam = np.matmul(T_lidar2cam, np.linalg.inv(T_lidar2vcs))
        T_vcs2img = K.dot(T_vcs2cam[:3, :])
        T_vcsgnd2img = T_vcs2img[:, (0, 1, 3)]

        T_ipm2vcsgnd = self._build_transform_ipm2egognd(
            camera_view_name, self.spatial_resolution, self.vcs_range
        )
        H_ipm2img = np.dot(T_vcsgnd2img, T_ipm2vcsgnd)  # original image

        return H_ipm2img

    @staticmethod
    def _view_ipm_mask(
        camera_name,
        output_size,
        vis_range,
        pixel_per_meter,
        camera_yaws,
    ):
        yaw = camera_yaws[camera_name] + 90  # yaw relative to arctan2 theta
        mask = np.zeros((output_size[0], output_size[1], 3), dtype=np.bool)
        for i in range(output_size[1]):
            for j in range(output_size[0]):
                theta = np.rad2deg(
                    np.arctan2(
                        -j + vis_range[0] * pixel_per_meter,
                        i - vis_range[2] * pixel_per_meter,
                    )
                )

                if abs(theta - yaw) > 90 and abs(theta - yaw) < 270:
                    mask[j, i, :] = True
        return mask

    @staticmethod
    def _remap_image(img, x, y, target_img, u, v, use_opencv=False):
        if use_opencv:
            map1 = np.reshape(x, (target_img.shape[0], target_img.shape[1]))
            map2 = np.reshape(y, (target_img.shape[0], target_img.shape[1]))
            target_img = cv2.remap(
                img,
                map1.astype(np.float32),
                map2.astype(np.float32),
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            valid_index = (
                (x >= 0) * (x < img.shape[1]) * (y >= 0) * (y < img.shape[0])
            )
            x = x[valid_index]
            y = y[valid_index]
            u = u[valid_index]
            v = v[valid_index]
            target_img[v, u, :] = img[y, x, :]

        return target_img

    def _coords_before_after_ipm(self, height, width, ipm):
        """Two coordinates corresponding to before and after ipm.

        Args:
            height (int): height of image after ipm.
            width (int): width of image after ipm.
            ipm (np.array): homography mat, shape (3, 3).
        """
        x, y = np.meshgrid(range(width), range(height), indexing="xy")
        id_coords = np.stack([x, y], axis=0).astype("float32")
        ones = np.ones((1, height, width), dtype="float32")
        pix_coords = np.concatenate([id_coords, ones], axis=0).reshape((3, -1))

        cam_points = np.matmul(ipm, pix_coords)

        cam_points[2:3, :] = np.clip(
            cam_points[2:3, :], a_min=VALID_Z, a_max=None
        )
        new_pix_coords = cam_points / (cam_points[2:3, :])
        return pix_coords, new_pix_coords

    def _get_ipm_image(self, img_ud, ipm_size, ipm):
        """Get the ipm fusion image from 6v image."""
        height = ipm_size[0]
        width = ipm_size[1]
        x, y = np.meshgrid(range(width), range(height), indexing="xy")
        _, XYW = self._coords_before_after_ipm(height, width, ipm)

        u = XYW[0, :]
        v = XYW[1, :]
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        img_ipm = self._remap_image(
            img_ud, u, v, np.zeros((height, width, 3)), x, y, use_opencv=True
        )
        return img_ipm

    def _get_transform_mat(self, sub_dir):
        """Generate a transformation matrix to corrent camera intrinsics.

        The order of img changes:
        resize->crop->padding->scale_resize

        """
        transform_list = self.homo_transforms[sub_dir]
        resize_mat = np.eye(3, dtype="float32")
        crop_mat = np.eye(3, dtype="float32")
        padding_mat = np.eye(3, dtype="float32")
        scale_mat = np.eye(3, dtype="float32")

        if "Resize" in transform_list.keys():
            resize_mat[0, 0] = transform_list["Resize"][1]
            resize_mat[1, 1] = transform_list["Resize"][0]
        if "Crop" in transform_list.keys():
            crop_mat[1, 2] = -transform_list["Crop"][0]  # top
            crop_mat[0, 2] = -transform_list["Crop"][1]  # left
        if "Pad" in transform_list.keys():
            # left, top, right and bottom
            padding_mat[1, 2] = transform_list["Pad"][1]  # top
            padding_mat[0, 2] = transform_list["Pad"][0]  # left
        if "ResizeHomo" in transform_list.keys():
            scale_mat[0, 0] = transform_list["ResizeHomo"]
            scale_mat[1, 1] = transform_list["ResizeHomo"]
        return scale_mat @ padding_mat @ crop_mat @ resize_mat

    def visualize_bev_homography(self, sync_imgs, apply_view_mask=True):
        """Visualize ipm fusion image under bev.

        If provide calib_path, visualize ipm result for undistored 6v image.
        Otherwise, only visualize ipm result for distored 6v image, since
        homo_path only proive calculated homography without undistortion
        params K and d. Also, ipm fusion order must be
        camera_front->camera_rear_left->camera_rear_right->
        camera_front_left->camera_front_right-> camera_rear.

        Args:
            sync_imgs (dict): map image names to 6 camera views, e.g.,

                camera_front: 'path/to/xxx.jpg',
                camera_front_left: 'path/to/xxx.jpg',
                camera_front_right: 'path/to/xxx.jpg',
                camera_rear: 'path/to/xxx.jpg',
                camera_rear_left: 'path/to/xxx.jpg',
                camera_rear_right: 'path/to/xxx.jpg',
            apply_view_mask (bool): whether apply view mask on ipm result.

        Returns:
            numpy array: image after ipm.
        """
        pixel_per_meter = 1 / self.spatial_resolution[0]
        ipm_size = (self.height, self.width, 3)
        birdsEyeView = np.zeros(ipm_size)

        # correspond to 6v cameras
        camera_yaws = {
            "camera_front": 0,
            "camera_rear_left": 130,
            "camera_rear_right": -130,
            "camera_front_left": 50,
            "camera_front_right": -50,
            "camera_rear": 180,
        }
        raw_imgs = {}
        for camera_view in camera_yaws:
            if self.calib_path is not None:
                lidar_param_file = os.path.join(self.calib_path, "log_trans")
                camera_param_file = os.path.join(
                    self.calib_path, camera_view + ".yaml"
                )
                calib_param_file = os.path.join(
                    self.calib_path, "calibration.json"
                )

                (
                    T_lidar2vcs,
                    T_lidar2cam,
                    K,
                    d_coef,
                ) = self._get_calib_parameters(
                    camera_view,
                    lidar_param_file,
                    camera_param_file,
                    calib_param_file,
                )

                homo = self.compute_homography(
                    camera_view,
                    K,
                    T_lidar2vcs,
                    T_lidar2cam,
                )
            else:
                hom_file = os.path.join(self.homo_path, camera_view + ".npy")
                assert os.path.exists(hom_file), hom_file
                homo = np.load(hom_file)
                K, d_coef = None, None

            image = cv2.imread(sync_imgs[camera_view])
            if K is not None and d_coef is not None:
                undist_img = cv2.undistort(image[:, :, ::-1], K, d_coef)
            else:
                undist_img = image[:, :, ::-1]

            img_ipm = self._get_ipm_image(undist_img, ipm_size, homo)

            if apply_view_mask:
                mask = self._view_ipm_mask(
                    camera_view,
                    ipm_size,
                    self.vcs_range,
                    pixel_per_meter,
                    camera_yaws,
                )
            else:
                mask = np.zeros((ipm_size[0], ipm_size[1], 3), dtype=np.bool)
            img_ipm_masked = (img_ipm * (~mask)).astype("uint8")
            mask = np.any(img_ipm_masked != (0, 0, 0), axis=-1)
            birdsEyeView[mask] = img_ipm_masked[mask]

            raw_imgs[camera_view] = cv2.resize(image, (512, 320))
        top_img = np.hstack(
            [
                raw_imgs["camera_front_left"],
                raw_imgs["camera_front"],
                raw_imgs["camera_front_right"],
            ]
        )
        middle_img = np.hstack(
            [
                raw_imgs["camera_rear_left"],
                raw_imgs["camera_rear"],
                raw_imgs["camera_rear_right"],
            ]
        )
        pad_left = (512 * 3 - ipm_size[1]) // 2
        pad_right = 512 * 3 - ipm_size[1] - pad_left
        bottom_img = np.hstack(
            [
                np.zeros((ipm_size[0], pad_left, 3)),
                birdsEyeView,
                np.zeros((ipm_size[0], pad_right, 3)),
            ]
        )
        img_all = np.vstack([top_img, middle_img, bottom_img])

        return img_all

    def compute_homo_offset(self, homography):
        """Compute homography offset by homography."""
        pix_coords, new_pix_coords = self._coords_before_after_ipm(
            self.height, self.width, homography
        )
        homo_offset = new_pix_coords[:2] - pix_coords[:2]  # (2, -1)
        homo_offset = homo_offset.reshape(2, self.height, self.width)
        homo_offset = np.transpose(homo_offset, (1, 2, 0))
        return homo_offset

    def get_homography(self):  # noqa: D205,D400
        """Get homography by load homography in homo_path or compute
        homography according calibration params in calib_path for each
        camera view.

        Note that compute homography by calibration params if calib_path
        is provided, otherwise load homography saved in homo_path. Also,
        if both calibration.json and log_trans files in calib_path,
        use calibration.json to compute homography.

        Returns:
            numpy array: homography matrix.
        """
        assert not (
            self.calib_path is None and self.homo_path is None
        ), "please provide calib path or homo path"

        if self.norm_homo:
            assert len(self.camera_view_names) == len(self.per_view_shape), (
                "please provide ori image shape of each view"
                "to normalize homography matrix."
            )

        H = []
        for sub_dir in self.camera_view_names:
            if self.calib_path is not None:
                assert self.spatial_resolution is not None
                assert self.vcs_range is not None
                lidar_param_file = os.path.join(self.calib_path, "log_trans")
                camera_param_file = os.path.join(
                    self.calib_path, sub_dir + ".yaml"
                )
                calib_param_file = os.path.join(
                    self.calib_path, "calibration.json"
                )

                (
                    T_lidar2vcs,
                    T_lidar2cam,
                    K,
                    d_coef,
                ) = self._get_calib_parameters(
                    sub_dir,
                    lidar_param_file,
                    camera_param_file,
                    calib_param_file,
                )

                ori_homo = self.compute_homography(
                    sub_dir,
                    K,
                    T_lidar2vcs,
                    T_lidar2cam,
                )
            else:
                hom_file = os.path.join(self.homo_path, sub_dir + ".npy")
                assert os.path.exists(hom_file), hom_file
                ori_homo = np.load(hom_file)

            if self.norm_homo:
                norm_mat = np.array(
                    [
                        [1 / self.per_view_shape[sub_dir][1], 0, 0],
                        [0, 1 / self.per_view_shape[sub_dir][0], 0],
                        [0, 0, 1],
                    ],
                    dtype="float32",
                ).reshape(3, 3)
                H.append(norm_mat @ ori_homo.astype("float32"))
            else:
                H.append(ori_homo.astype("float32"))
        return np.stack(H, axis=0)

    def get_homo_offset(
        self,
    ):

        if self.use_distorted_offset:
            return self.get_dist_homo_offset()
        else:
            return self.get_undist_homo_offset()

    def get_undist_homo_offset(self):
        """Get homography offset for each camera view."""
        ori_norm_homo = self.norm_homo
        self.norm_homo = True

        H = self.get_homography()  # (6, 3, 3)
        self.norm_homo = ori_norm_homo
        homo_offset = []

        for i, camera_name in enumerate(self.camera_view_names):
            transfomed_mat = self._get_transform_mat(camera_name)
            H[i] = transfomed_mat @ H[i]
        for i in range(H.shape[0]):
            homo_offset.append(self.compute_homo_offset(H[i, :, :]))

        return np.stack(homo_offset, axis=0)

    def get_dist_homo_offset(self):
        assert self.calib_path is not None
        if self.norm_homo:
            assert len(self.camera_view_names) == len(self.per_view_shape), (
                "please provide ori image shape of each view"
                "to normalize homography matrix."
            )
        dist_homo_offset = []
        for _, sub_dir in enumerate(self.camera_view_names):
            assert self.spatial_resolution is not None
            assert self.vcs_range is not None
            lidar_param_file = os.path.join(self.calib_path, "log_trans")
            camera_param_file = os.path.join(
                self.calib_path, sub_dir + ".yaml"
            )
            calib_param_file = os.path.join(
                self.calib_path, "calibration.json"
            )

            (
                T_lidar2vcs,
                T_lidar2cam,
                K,
                d_coef,
            ) = self._get_calib_parameters(
                sub_dir,
                lidar_param_file,
                camera_param_file,
                calib_param_file,
            )

            ori_homo = self.compute_homography(
                sub_dir,
                K,
                T_lidar2vcs,
                T_lidar2cam,
            )
            norm_mat = np.array(
                [
                    [1 / self.per_view_shape[sub_dir][1], 0, 0],
                    [0, 1 / self.per_view_shape[sub_dir][0], 0],
                    [0, 0, 1],
                ],
                dtype="float32",
            ).reshape(3, 3)
            norm_homo = norm_mat @ ori_homo
            norm_K = norm_mat @ K
            transfomed_mat = self._get_transform_mat(sub_dir)
            cur_K = transfomed_mat @ norm_K
            cur_homo = transfomed_mat @ norm_homo

            cur_bev2cam = np.linalg.inv(cur_K).astype("float32") @ cur_homo

            pix_coords, camera_coords = self._coords_before_after_ipm(
                self.height, self.width, cur_bev2cam
            )

            rvec = np.zeros(shape=(3, 1), dtype=np.float32)
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)

            if "fisheye" in sub_dir:
                camera_coords = camera_coords[:2, :].T
                camera_coords = camera_coords[np.newaxis, :, :]
                pts_distorted = cv2.fisheye.distortPoints(
                    camera_coords, cur_K, D=d_coef
                )
            else:
                assert check_version(
                    cv2, "4.5"
                ), "Maybe you need to update opencv-python to 4.5"
                pts_distorted, _ = cv2.projectPoints(
                    camera_coords, rvec, tvec, cur_K, d_coef
                )
            pts_distorted = pts_distorted.reshape((-1, 2))
            d_homo_offset = pts_distorted - pix_coords.transpose(1, 0)[:, 0:2]
            d_homo_offset = d_homo_offset.reshape((self.height, self.width, 2))
            dist_homo_offset.append(d_homo_offset)
        if self.dist_homo_offset is None:
            dist_homo_offset = np.stack(dist_homo_offset, axis=0).astype(
                "float32"
            )

        return dist_homo_offset
