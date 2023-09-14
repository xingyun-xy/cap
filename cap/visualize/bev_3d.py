import copy
import json
import logging
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import cv2
import numpy as np

from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info
from cap.utils.logger import MSGColor, format_msg

logger = logging.getLogger(__name__)

VALID_Z = 0.1


@OBJECT_REGISTRY.register
class Bev3DVisualize(object):
    """BEV3D Visualize tools.

    Args:
        save_path (str): absolute path to save the visualize results.
        bev_size (Sequence): bev image map size. Common to (512, 512).
        vcs_range (Sequence): vcs visible range. Common to
            (-30.0, -51.2, 72.4, 51.2).
        score_threshold (float): score_threshold to draw bboxes. Common to 0.2.
        anno_show (Optional[bool]): whether to show the GT boxes.
            Defaults to True.
        project_image (Optional[bool]): whether project the bev3d to images.
            Defaults to True.
        concat_imgs (Optional[bool]): whether to concat the bev box img with
            cameras images. NOTE: only valid if project_image==True,
            Defaults to True.
        draw_lidar (Optional[bool]): whether to draw the bev boxes on lidar
            point cloud. Defaults to False. NOTE: if draw_lidar, the visualize
            time will be long.
        fisheye (Optional[bool]): whether to visualize for fisheye bev3d.
            Defaults to False.
        inference_mode (Optional[bool]): visualize and save pred pkl
            in inference mode. default=False.
        inference_pkl_dir (str): path to save the inference bev3d results.
        NOTE: the default color setting, GT: Green Boxes (ignore: white),
            Pred: Red Boxes
    """

    def __init__(
        self,
        save_path,
        bev_size,
        vcs_range,
        score_threshold,
        anno_show,
        project_image,
        concat_imgs,
        draw_lidar,
        fisheye,
        inference_mode=False,
        inference_pkl_dir=None,
    ):

        self.save_path = save_path
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.score_threshold = score_threshold
        self.anno_show = anno_show
        self.project_image = project_image
        self.draw_lidar = draw_lidar
        self.fisheye = fisheye
        self.inference_mode = inference_mode
        if not self.project_image:
            self.concat_imgs = False
            logger.warning(
                format_msg(
                    "`concat_imgs must False if " "`project_image == False",
                    MSGColor.RED,
                )
            )
        else:
            self.concat_imgs = concat_imgs
        if self.project_image or self.draw_lidar:
            self.project_calib = {}
            self.sync_info = {}
        if self.inference_mode:
            self.inference_pkl_dir = inference_pkl_dir
            self.inference_reset()

        self.cameras = (
            [
                "camera_front",
                "camera_front_left",
                "camera_front_right",
                "camera_rear_left",
                "camera_rear_right",
                "camera_rear",
            ]
            if not fisheye
            else [
                "fisheye_front",
                "fisheye_rear",
                "fisheye_left",
                "fisheye_right",
            ]
        )

    def inference_reset(self):
        self.save_res = defaultdict(list)
        rank, word_size = get_dist_info()
        self.pkl_save_path = os.path.join(
            self.inference_pkl_dir,
            f"inference_rank_{rank}.pkl",
        )
        pkl_save_root = os.path.dirname(self.pkl_save_path)
        if not os.path.exists(pkl_save_root):
            os.makedirs(pkl_save_root, exist_ok=True)

    def save_inference_result(self, output: Dict, batch: Dict):
        """Save bev3d inference results.

        Args:
            output: output from model.
            batch: items contains input's data and GT annos.
        """
        if os.path.exists(self.pkl_save_path):
            with open(self.pkl_save_path, "rb") as f:
                exist_pred = pickle.load(f)
            self.save_res.update(exist_pred)

        batch_size, num_objs = output[list(output.keys())[0]].shape[:2]

        if not isinstance(batch["timestamp"], np.ndarray):
            batch_timestamps = np.array(batch["timestamp"].cpu())
        else:
            batch_timestamps = batch["timestamp"]
        batch_timestamps = [
            str(int(_bs_time * 1000)) for _bs_time in batch_timestamps
        ]

        location = np.concatenate(
            (output["bev3d_ct"], np.expand_dims(output["bev3d_loc_z"], -1)),
            axis=-1,
        )
        for i in range(batch_size):
            front_img_timestamp = batch_timestamps[i]
            pack_dir = batch["pack_dir"][i]
            # the default timestamp in auto3dv is camera_front
            key = os.path.join(pack_dir, front_img_timestamp)
            for j in range(num_objs):
                # filter the padded objs in data transform
                if output["bev3d_score"][i][j] > 0.0:
                    self.save_res[key].append(
                        # all the key_names are used to adap to adas_eval
                        {
                            "dimensions": output["bev3d_dim"][i][j].tolist(),
                            "class_id": output["bev3d_cls_id"][i][j],
                            "score": output["bev3d_score"][i][j],
                            "yaw": output["bev3d_rot"][i][j],
                            "location": location[i][j].tolist(),
                            "timestamp": str(front_img_timestamp),
                        }
                    )
            if len(self.save_res[key]) < 1:
                self.save_res[key] = []

        with open(self.pkl_save_path, "wb") as f:
            pickle.dump(self.save_res, f)

    def save_imgs(self, output: Dict, batch: Dict):
        """Save bev 3d vis results.

        Args:
            output: output from model.
            batch: items contains input's data and GT annos.
        """
        if not isinstance(batch["timestamp"], np.ndarray):
            batch_timestamps = np.array(batch["timestamp"].cpu())
        else:
            batch_timestamps = batch["timestamp"]
        batch_timestamps = [
            str(int(_bs_time * 1000)) for _bs_time in batch_timestamps
        ]
        output_np = {}
        for k, v in output.items():
            if not isinstance(batch["timestamp"], np.ndarray):
                output_np[k] = v.cpu().numpy()
            else:
                output_np[k] = v

        if self.anno_show:
            batch_np = {}
            annos_bev_3d = batch["annos_bev_3d"]
            for k, v in annos_bev_3d.items():
                if not isinstance(batch["timestamp"], np.ndarray):
                    batch_np[k] = v.cpu().numpy()
                else:
                    batch_np[k] = v
        for bs, timestamp in enumerate(batch_timestamps):
            # get the lidar calib and 6V imgs
            pack_dir = batch["pack_dir"][bs]
            img_paths = [
                os.path.join(pack_dir, cam_path[bs])
                for cam_path in batch["img_paths"]
            ]
            attribute_file = os.path.join(pack_dir, "attribute.json")
            assert os.path.exists(
                attribute_file
            ), f"Please check the pack: {pack_dir}'s attribute file"

            # if project the bev3d results to camera image or
            # lidar point-cloud, should read the calibration.
            if self.project_image or self.draw_lidar:
                if attribute_file not in self.project_calib:
                    self.project_calib[attribute_file] = {}
                    lidar_calibs, lidar2chassis = load_calib(
                        attribute_file, self.cameras
                    )
                    self.project_calib[attribute_file][
                        "lidar_calibs"
                    ] = lidar_calibs
                    self.project_calib[attribute_file][
                        "lidar2chassis"
                    ] = lidar2chassis

            # initial the bev map, if draw_lidar=True, bev map is,
            # lidar point-cloud map
            if not self.draw_lidar:
                bev_img = init_bev(
                    world_width=102,
                    init_bev_size=(1024, 1024),
                    center_loc=(512, 362 * 2),
                )
                bev_size = bev_img.shape[:2]
            else:
                if attribute_file not in self.sync_info:
                    with open(attribute_file, "r") as f:
                        attr = json.load(f)
                    self.sync_info[attribute_file] = attr["sync"]

                sync = self.sync_info[attribute_file]
                sync_index = sync["camera_front"].index(int(timestamp))
                lidar_top_timestamp = sync["lidar_top"][sync_index]
                lidar_file_root = os.path.dirname(attribute_file)
                lidar_file = os.path.join(
                    lidar_file_root,
                    "lidar_top",
                    str(lidar_top_timestamp) + ".bin",
                )

                points = read_lidar(lidar_file)

                # NOTE: _voxel_size, _coors_range and center_bev
                # is hard code here for 6V images.
                _voxel_size = [0.08, 0.08, 0.2]
                _coors_range = [-128.0, -72.0, -5.0, 128.0, 72.0, 3.0]
                bev_img = point_to_vis_bev(points, _voxel_size, _coors_range)

                # draw ego car on bev_img, NOTE: hard code here
                center_bev = (1600, 900)
                bev_pixel_meter = 10
                self_lt = (
                    int((center_bev[0] - 4 / 2 * bev_pixel_meter)),
                    int((center_bev[1] - 1.8 / 2 * bev_pixel_meter)),
                )  # noqa
                self_rb = (
                    int((center_bev[0] + 4 / 2 * bev_pixel_meter)),
                    int((center_bev[1] + 1.8 / 2 * bev_pixel_meter)),
                )  # noqa
                front_arrow = (
                    int(center_bev[0] + 4 * bev_pixel_meter),
                    center_bev[1],
                )
                cv2.rectangle(bev_img, self_lt, self_rb, (255, 255, 255), 2)
                cv2.line(bev_img, center_bev, front_arrow, (255, 255, 255), 2)

            # Draw the GT boxes.
            ignore_exist = False
            if self.anno_show:
                anno_cls_id = batch_np["vcs_cls_"][bs]
                anno_cls_valid = anno_cls_id != -99

                anno_ignore = batch_np["vcs_ignore_"][bs]
                if anno_ignore.sum() > 0:
                    ignore_exist = True

                anno_valid = np.logical_and(
                    anno_cls_valid, np.logical_not(anno_ignore)
                )
                anno_ignore = np.logical_and(anno_cls_valid, anno_ignore)

                anno_cls_id_valid = anno_cls_id[anno_valid][:, None]
                anno_cls_id_ignore = anno_cls_id[anno_ignore][:, None]

                anno_score_valid = np.ones_like(anno_cls_id_valid)
                gt_bboxes_valid = np.concatenate(
                    [
                        batch_np["vcs_loc_"][bs][:, :2][anno_valid],
                        batch_np["vcs_loc_"][bs][:, 2][anno_valid][:, None],
                        batch_np["vcs_dim_"][bs][anno_valid],
                        batch_np["vcs_rot_z_"][bs][anno_valid][:, None],
                    ],
                    axis=1,
                )
                if self.draw_lidar:
                    bev_img = self.draw_lidar_boxes(
                        bev_map=bev_img,
                        bev3d_ct=gt_bboxes_valid[:, :2],
                        bev3d_dim=gt_bboxes_valid[:, 3:6],
                        bev3d_loc_z=gt_bboxes_valid[:, 2],
                        bev3d_rot=gt_bboxes_valid[:, -1],
                        bev3d_score=anno_score_valid[:, 0],
                        project_calib=self.project_calib[attribute_file],
                        coors_range=_coors_range,
                        color=(0, 255, 0),
                        score_threshold=self.score_threshold,
                    )
                else:
                    bev_img = self.draw_bev_boxes(
                        bev_img=bev_img,
                        pred_bboxes=gt_bboxes_valid,
                        class_id=anno_cls_id_valid,
                        score=anno_score_valid,
                        bev_size=bev_size,
                        bev_range=self.vcs_range,
                        score_threshold=self.score_threshold,
                        thickness=2,
                        color=(0, 255, 0),  # green
                    )
                # draw ignore by white color
                if ignore_exist:
                    anno_score_ignore = np.ones_like(anno_cls_id_ignore)
                    gt_bboxes_ignore = np.concatenate(
                        [
                            batch_np["vcs_loc_"][bs][:, :2][anno_ignore],
                            batch_np["vcs_loc_"][bs][:, 2][anno_ignore][
                                :, None
                            ],
                            batch_np["vcs_dim_"][bs][anno_ignore],
                            batch_np["vcs_rot_z_"][bs][anno_ignore][:, None],
                        ],
                        axis=1,
                    )
                    if self.draw_lidar:
                        bev_img = self.draw_lidar_boxes(
                            bev_map=bev_img,
                            bev3d_ct=gt_bboxes_ignore[:, :2],
                            bev3d_dim=gt_bboxes_ignore[:, 3:6],
                            bev3d_loc_z=gt_bboxes_ignore[:, 2],
                            bev3d_rot=gt_bboxes_ignore[:, -1],
                            bev3d_score=anno_score_ignore[:, 0],
                            project_calib=self.project_calib[attribute_file],
                            coors_range=_coors_range,
                            color=(255, 255, 255),
                            score_threshold=self.score_threshold,
                        )
                    else:
                        bev_img = self.draw_bev_boxes(
                            bev_img=bev_img,
                            pred_bboxes=gt_bboxes_ignore,
                            class_id=anno_cls_id_ignore,
                            score=anno_score_ignore,
                            bev_size=bev_size,
                            bev_range=self.vcs_range,
                            score_threshold=self.score_threshold,
                            thickness=2,
                            color=(255, 255, 255),
                        )
            pred_bboxes = np.concatenate(
                [
                    output_np["bev3d_ct"][bs],
                    output_np["bev3d_loc_z"][bs][:, None],
                    output_np["bev3d_dim"][bs],
                    output_np["bev3d_rot"][bs][:, None],
                ],
                axis=1,
            )
            class_id = output_np["bev3d_cls_id"][bs][:, None]
            score = output_np["bev3d_score"][bs][:, None]

            if self.draw_lidar:
                bev_boxes_img = self.draw_lidar_boxes(
                    bev_map=bev_img,
                    bev3d_ct=pred_bboxes[:, :2],
                    bev3d_dim=pred_bboxes[:, 3:6],
                    bev3d_loc_z=pred_bboxes[:, 2],
                    bev3d_rot=pred_bboxes[:, -1],
                    bev3d_score=score[:, 0],
                    project_calib=self.project_calib[attribute_file],
                    coors_range=_coors_range,
                    color=(0, 0, 255),
                    score_threshold=self.score_threshold,
                )
                bev_boxes_img = cv2.flip(bev_boxes_img, 0)
                bev_boxes_img = cv2.resize(bev_boxes_img, (1920, 1024))
                bev_boxes_img = cv2.flip(bev_boxes_img, 1)
                bev_boxes_img = cv2.transpose(bev_boxes_img)
                bev_boxes_img = bev_boxes_img[256:1664, :, :]
            else:
                bev_boxes_img = self.draw_bev_boxes(
                    bev_img=bev_img,
                    pred_bboxes=pred_bboxes,
                    class_id=class_id,
                    score=score,
                    bev_size=bev_size,
                    bev_range=self.vcs_range,
                    score_threshold=self.score_threshold,
                    thickness=2,
                    color=(0, 0, 255),  # red
                )
            if not self.concat_imgs:
                bev_box_path = os.path.join(
                    self.save_path, timestamp + "_bev.jpg"
                )
                if not self.inference_mode:
                    cv2.imwrite(bev_box_path, bev_boxes_img)

            # Project the bev3d pred boxes to 6V cam images.
            color_imgs = [cv2.imread(im_path) for im_path in img_paths]
            if self.project_image:
                if self.anno_show:
                    camera_boxes_imgs = self.draw_camera_boxes(
                        bev3d_ct=gt_bboxes_valid[:, :2],
                        bev3d_dim=gt_bboxes_valid[:, 3:6],
                        bev3d_loc_z=gt_bboxes_valid[:, 2],
                        bev3d_rot=gt_bboxes_valid[:, -1],
                        bev3d_score=anno_score_valid[:, 0],
                        bev3d_cls_id=anno_cls_id_valid[:, 0],
                        project_calib=self.project_calib[attribute_file],
                        cameras=self.cameras,
                        color_imgs=color_imgs,
                        fisheye=self.fisheye,
                        score_threshold=self.score_threshold,
                        color=(0, 255, 0),
                    )

                    # draw ignore by white
                    if ignore_exist:
                        camera_boxes_imgs = self.draw_camera_boxes(
                            bev3d_ct=gt_bboxes_ignore[:, :2],
                            bev3d_dim=gt_bboxes_ignore[:, 3:6],
                            bev3d_loc_z=gt_bboxes_ignore[:, 2],
                            bev3d_rot=gt_bboxes_ignore[:, -1],
                            bev3d_score=anno_score_ignore[:, 0],
                            bev3d_cls_id=anno_cls_id_ignore[:, 0],
                            project_calib=self.project_calib[attribute_file],
                            cameras=self.cameras,
                            color_imgs=camera_boxes_imgs,
                            fisheye=self.fisheye,
                            score_threshold=self.score_threshold,
                            color=(255, 255, 255),
                        )
                else:
                    camera_boxes_imgs = color_imgs
                camera_boxes_imgs = self.draw_camera_boxes(
                    bev3d_ct=output_np["bev3d_ct"][bs],
                    bev3d_dim=output_np["bev3d_dim"][bs],
                    bev3d_loc_z=output_np["bev3d_loc_z"][bs],
                    bev3d_rot=output_np["bev3d_rot"][bs],
                    bev3d_score=output_np["bev3d_score"][bs],
                    bev3d_cls_id=output_np["bev3d_cls_id"][bs],
                    project_calib=self.project_calib[attribute_file],
                    cameras=self.cameras,
                    color_imgs=camera_boxes_imgs,
                    fisheye=self.fisheye,
                    score_threshold=self.score_threshold,
                    color=(0, 0, 255),
                )
                camera_imgs = [
                    cv2.resize(img, (2048 // 2, 1280 // 2))
                    for img in camera_boxes_imgs
                ]
                if self.fisheye:
                    camera_imgs.append(np.zeros_like(camera_imgs[0]))
                    camera_imgs.append(np.zeros_like(camera_imgs[0]))
                    upper_imgs = np.hstack(
                        (camera_imgs[2], camera_imgs[0], camera_imgs[3])
                    )
                    bottom_imgs = np.hstack(
                        (camera_imgs[4], camera_imgs[1], camera_imgs[5])
                    )
                else:
                    upper_imgs = np.hstack(
                        (camera_imgs[1], camera_imgs[0], camera_imgs[2])
                    )
                    bottom_imgs = np.hstack(
                        (camera_imgs[3], camera_imgs[5], camera_imgs[4])
                    )
                if not self.concat_imgs:
                    camera_bboxes_img = np.vstack((upper_imgs, bottom_imgs))

                    camera_box_path = os.path.join(
                        self.save_path, batch_timestamps[bs] + "_6v_cam.jpg"
                    )
                    if not self.inference_mode:
                        cv2.imwrite(camera_box_path, camera_bboxes_img)
                    else:
                        return camera_bboxes_img
                elif not self.draw_lidar:
                    bev_boxes_img = cv2.resize(bev_boxes_img, (768, 768))
                    image_pad = np.zeros((768, 1152, 3))
                    middle_imgs = np.hstack(
                        (image_pad, bev_boxes_img, image_pad)
                    )
                    img_all = np.vstack((upper_imgs, middle_imgs, bottom_imgs))
                    savefile = os.path.join(
                        self.save_path, batch_timestamps[bs] + ".jpg"
                    )
                    if not self.inference_mode:
                        cv2.imwrite(savefile, img_all)
                    else:
                        return img_all
                else:
                    front = cv2.resize(camera_boxes_imgs[0], (1024, 640))
                    front_left = cv2.resize(camera_boxes_imgs[1], (683, 384))
                    front_right = cv2.resize(camera_boxes_imgs[2], (683, 384))
                    front_fake = np.zeros((384, 682, 3))

                    padding_image = np.zeros((640, 512, 3))

                    rear_left = cv2.resize(camera_boxes_imgs[3], (683, 384))
                    rear_right = cv2.resize(camera_boxes_imgs[4], (683, 384))
                    rear = cv2.resize(camera_boxes_imgs[5], (682, 384))

                    upper_img = np.hstack(
                        (front_left, front_fake, front_right)
                    )
                    middle_img = np.hstack(
                        (padding_image, front, padding_image)
                    )
                    bottom_img = np.hstack((rear_left, rear, rear_right))
                    img_left = np.vstack((upper_img, middle_img, bottom_img))
                    img_all = np.hstack((img_left, bev_boxes_img))
                    savefile = os.path.join(
                        self.save_path, batch_timestamps[bs] + ".jpg"
                    )
                    if not self.inference_mode:
                        cv2.imwrite(savefile, img_all)
                    else:
                        return img_all

    @staticmethod
    def draw_bev_boxes(
        bev_img,
        pred_bboxes,
        class_id,
        score,
        bev_size,
        bev_range,
        score_threshold,
        thickness,
        color,
    ):
        """Draw the boxes on bev image.

        Args:
            bev_img (np.ndarray): bev image used for draw boxes.
            pred_bboxes: the predict boxes,(N, 7): N means: num_of_objs,
                7 means [x,y,z,h,w,l,yaw].
            class_id (np.ndarray): the predict class_id.
            score (np.ndarray): the predict score.
            bev_size (Sequence): bev image map size. Common to (512, 512).
            bev_range (Sequence): vcs visible range. Common to
                (-30.0, -51.2, 72.4, 51.2).
            score_threshold (float, optional): score_threshold,
                Defaults to 0.2.
            thickness (int): thickness of lines.
            color (tuple): color of boxes.
        """
        vcs_corner = get_3dboxcorner_in_vcs_numpy(pred_bboxes)
        bev_vis_image = draw_color_bbox(
            vcs_corner,
            bev_img,
            class_id,
            score,
            bev_size,
            bev_range,
            scor_thr=score_threshold,
            thickness=thickness,
            color=color,
        )
        bev_img = cv2.resize(bev_img, bev_size)
        return bev_vis_image

    @staticmethod
    def draw_camera_boxes(
        bev3d_dim,
        bev3d_ct,
        bev3d_loc_z,
        bev3d_rot,
        bev3d_score,
        bev3d_cls_id,
        project_calib,
        cameras,
        color,
        color_imgs=None,
        fisheye=False,
        score_threshold: float = 0.2,
    ):
        """Draw box on camera images."""
        lidar_calibs, lidar2chassis = (
            project_calib["lidar_calibs"],
            project_calib["lidar2chassis"],
        )

        camera2index = (
            {
                "fisheye_front": 0,
                "fisheye_rear": 1,
                "fisheye_left": 2,
                "fisheye_right": 3,
            }
            if fisheye
            else {
                "camera_front": 0,
                "camera_front_left": 1,
                "camera_front_right": 2,
                "camera_rear_left": 3,
                "camera_rear_right": 4,
                "camera_rear": 5,
            }
        )

        # (1) Getting the attr based on score_thr
        score_mask = bev3d_score > score_threshold
        vcs_loc_xy = bev3d_ct[score_mask]
        box_dimension = bev3d_dim[score_mask]  # h w l
        vcs_loc_z = bev3d_loc_z[score_mask]
        # cls_id = bev3d_cls_id[score_mask]
        vcs_loc_xy = np.concatenate(
            (vcs_loc_xy, vcs_loc_z[:, np.newaxis]), axis=1
        )
        # (2) Change the attr to support vis setting
        box_dimension = box_dimension[:, (1, 2, 0)]
        rot_z = bev3d_rot[score_mask]
        rot_z = -(rot_z + np.pi / 2)
        if len(vcs_loc_xy) > 0:
            # vcs->lidar
            homo_ones = np.ones([len(vcs_loc_xy), 1])
            vcs_loc_xy = np.concatenate((vcs_loc_xy, homo_ones), axis=1)
            lidar_loc = [
                np.linalg.inv(lidar2chassis) @ _vcs_loc
                for _vcs_loc in vcs_loc_xy
            ]
            lidar_loc = np.array(lidar_loc)[:, :3]
            # vis results
            ddd_corners = center_to_corner_box3d(
                lidar_loc,
                box_dimension,
                rot_z,
                origin=(0.5, 0.5, 0.5),
            )
            img_list = []
            for cam in cameras:
                index = camera2index[cam]
                img = color_imgs[index]
                if not fisheye:
                    # img = undistort(img, lidar_calibs[cam])
                    image = draw_detection_box_on_image(
                        img,
                        lidar_calibs[cam],
                        ddd_corners,
                        color=color,
                        distort_label=True,
                    )
                else:
                    image = draw_fisheye_detection_box_on_image(
                        img,
                        lidar_calibs[cam],
                        ddd_corners,
                        color=color,
                        distort_label=True,
                        draw_lidar=True,
                    )
                img_list.append(image)
        else:
            img_list = []
            for cam in cameras:
                index = camera2index[cam]
                img = color_imgs[index]
                if not fisheye:
                    pass
                img_list.append(img)
        return img_list

    @staticmethod
    def draw_lidar_boxes(
        bev_map,
        bev3d_dim,
        bev3d_ct,
        bev3d_loc_z,
        bev3d_rot,
        bev3d_score,
        project_calib,
        coors_range,
        color,
        score_threshold: float = 0.2,
    ):
        """Draw box on lidar point-cloud image."""
        _, lidar2chassis = (
            project_calib["lidar_calibs"],
            project_calib["lidar2chassis"],
        )

        # (1) Getting the attr based on score_thr
        score_mask = bev3d_score > score_threshold
        vcs_loc_xy = bev3d_ct[score_mask]
        box_dimension = bev3d_dim[score_mask]  # h w l
        vcs_loc_z = bev3d_loc_z[score_mask]
        vcs_loc_xy = np.concatenate(
            (vcs_loc_xy, vcs_loc_z[:, np.newaxis]), axis=1
        )
        # (2) Change the attr to support vis setting
        box_dimension = box_dimension[:, (1, 2, 0)]
        rot_z = bev3d_rot[score_mask]
        rot_z = -(rot_z + np.pi / 2)
        if len(vcs_loc_xy) > 0:
            # vcs->lidar
            homo_ones = np.ones([len(vcs_loc_xy), 1])
            vcs_loc_xy = np.concatenate((vcs_loc_xy, homo_ones), axis=1)
            lidar_loc = [
                np.linalg.inv(lidar2chassis) @ _vcs_loc
                for _vcs_loc in vcs_loc_xy
            ]
            lidar_loc = np.array(lidar_loc)[:, :3]
            # vis results
            pred_bbox = np.concatenate(
                [lidar_loc, box_dimension, rot_z[:, None]], axis=1
            )

            bev_3d_vis_img = draw_kitti_pred_in_bev(
                pred_bbox, bev_map, coors_range, None, color=color
            )
        else:
            bev_3d_vis_img = bev_map
        return bev_3d_vis_img


def init_bev(
    world_width=102, init_bev_size=(1024, 1024), center_loc=(512, 362 * 2)
):
    """Intialize a bev map.

    Args:
        world_width (int, optional): The real world size. Defaults to 102m.
        init_bev_size (tuple, optional): Init bev map size. Defaults
            to (1024,1024).
        center_loc (tuple, optional): The pixel location of ego car.
            Defaults to (512, 362 * 2).

    """
    bev = np.zeros((*init_bev_size, 3), dtype=np.uint8)
    bev_pixel_meter = bev.shape[1] / world_width  # 10 pixel/meter
    # draw ego car
    self_lt = (
        int((center_loc[0] - 1.8 / 2 * bev_pixel_meter)),
        int((center_loc[1] - 4 / 2 * bev_pixel_meter)),
    )  # noqa
    self_rb = (
        int((center_loc[0] + 1.8 / 2 * bev_pixel_meter)),
        int((center_loc[1] + 4 / 2 * bev_pixel_meter)),
    )  # noqa
    front_arrow = (center_loc[0], int(center_loc[1] - 4 * bev_pixel_meter))
    cv2.rectangle(bev, self_lt, self_rb, (255, 255, 255), 2)
    cv2.line(bev, center_loc, front_arrow, (255, 255, 255), 2)
    # draw circle
    for meter in range(0, world_width, 10):
        color = (200, 200, 200)
        bev = cv2.circle(
            bev,
            center_loc,
            int(bev_pixel_meter * meter),
            color,
            1,
        )  # noqa
    # draw stright line
    cv2.line(
        bev,
        (init_bev_size[0] // 2, 0),
        (init_bev_size[0] // 2, init_bev_size[0]),
        color,
        1,
    )
    cv2.line(
        bev, (0, center_loc[1]), (init_bev_size[0], center_loc[1]), color, 1
    )
    return bev


def center_to_corner_box3d(
    centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2
):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim]
    )
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def draw_detection_box_on_image(
    _image,
    calib,
    points,
    distort_label=False,
    color=(0, 255, 0),
):
    image = _image.copy()
    P2, distCoeffs_default = calib["P2"], calib["disCoeffs"]
    Tr_vel2cam = calib["Tr_vel2cam"]
    if not distort_label:
        distCoeffs = 0.0 * np.copy(distCoeffs_default)
    else:
        distCoeffs = np.copy(distCoeffs_default)
    within_fov = []
    for idx, box in enumerate(points):  # noqa
        points_aug = deepcopy(box)
        points_aug = np.concatenate(
            (points_aug, np.ones((points_aug.shape[0], 1))), axis=1
        )
        points_aug[:, 3] = 1
        points_cam = project_velo_to_camera(points_aug, Tr_vel2cam)

        within_fov = points_cam[:, 2] > 0
        points_cam = points_cam[within_fov, :3].astype(np.float32)

        if points_cam.shape[0] == 0:
            continue

        image_pts = camera2image_pinhole(
            points_cam[:, :3], P2, distCoeffs, image.shape[1], image.shape[0]
        )

        if image_pts is not None:
            image_pts = image_pts.astype(np.int32)
            _draw_box_3d(image, image_pts, color, thickness=4)
    return image


def draw_fisheye_detection_box_on_image(
    _image,
    calib,
    points,
    distort_label=True,
    color=(0, 255, 0),
    draw_lidar=False,
):
    """Draw bev3d results on fisheye camera images.

    Args:
        _image (np.ndarray): fisheye image.
        calib (dict): calibration parameters.
        points (np.ndarray): lidar box points.
        distort_label (bool, optional): whether to distort images.
            Defaults to True.
        color (tuple, optional): box colors. Defaults to (0, 255, 0).
        draw_lidar (bool, optional): whether draw lidar box. Defaults to False.

    """
    image = _image.copy()
    image_height, image_width, _ = image.shape
    rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
    tvec = np.zeros(shape=(3, 1), dtype=np.float32)

    P2, distCoeffs_default, Tr_vel2cam = (
        calib["P2"],
        calib["disCoeffs"],
        calib["Tr_vel2cam"],
    )
    if not distort_label:
        distCoeffs = 0.0 * np.copy(distCoeffs_default)
    else:
        distCoeffs = np.copy(distCoeffs_default)

    within_fov = []
    for _, box in enumerate(points):
        points_aug = deepcopy(box)
        points_aug = np.concatenate(
            (points_aug, np.ones((points_aug.shape[0], 1))), axis=1
        )
        points_aug[:, 3] = 1
        points_cam = project_velo_to_camera(points_aug, Tr_vel2cam)

        within_fov = points_cam[:, 2] > 0
        points_cam = points_cam[within_fov, :3].astype(np.float32)

        if points_cam.shape[0] == 0:
            continue
        points_cam = np.expand_dims(points_cam[:, :3], 0)
        fx, fy = P2[0, 0], P2[1, 1]
        u, v = P2[0, 2], P2[1, 2]
        k_ = np.mat([[fx, 0.0, u], [0.0, fy, v], [0.0, 0.0, 1.0]])
        d_ = np.mat(distCoeffs[:4].T)
        image_pts = cv2.fisheye.projectPoints(
            points_cam, np.array(rvec), tvec, k_, d_
        )[0]
        image_pts = np.array(image_pts).squeeze(axis=0)

        if image_pts.shape[0] == 0:
            continue
        within_img0 = np.logical_and(
            image_pts[:, 0] < image_width, image_pts[:, 0] > 1
        )
        within_img1 = np.logical_and(
            image_pts[:, 1] < image_height, image_pts[:, 1] > 1
        )
        old_image_pts = image_pts
        within_img = np.logical_and(within_img1, within_img0)
        within_fov[within_fov] = within_img
        image_pts = image_pts[within_img]
        if image_pts.shape[0] == 0:
            continue
        if draw_lidar:
            if image_pts.shape[0] == 8:
                # idx1, idx2 means the bottom and top idx of a
                # 3D box's 8 points, and the (idx1, idx2) can determine
                # a line of boxes, total 12 lines in a box.
                for idx1, idx2 in [
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (0, 3),
                    (1, 2),
                    (4, 7),
                    (5, 6),
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                ]:
                    x1 = int(image_pts[idx1, 0])
                    x2 = int(image_pts[idx2, 0])
                    y1 = int(image_pts[idx1, 1])
                    y2 = int(image_pts[idx2, 1])
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=3)
            else:
                x1 = int(max(min(old_image_pts[:, 0]), 0))
                y1 = int(max(min(old_image_pts[:, 1]), 0))
                x2 = int(min(max(old_image_pts[:, 0]), image_width - 1))
                y2 = int(min(max(old_image_pts[:, 1]), image_height - 1))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=3)
    return image


def project_velo_to_camera(vel_data, Tr):
    # vel_data_c: col 0: back -> front
    #             col 1: down -> up
    #             col 2: left -> right
    homo_vel_data = np.hstack(
        (vel_data[:, :3], np.ones((vel_data.shape[0], 1), dtype="float32"))
    )
    vel_data_c = np.dot(homo_vel_data, Tr.T)
    vel_data_c /= vel_data_c[:, -1].reshape((-1, 1))
    vel_data_c = np.hstack(
        (vel_data_c[:, :3], vel_data[:, -1].reshape((-1, 1)))
    )
    return vel_data_c


def camera2image_pinhole(
    points_cam_input, P2, distCoeffs, img_w=1920, img_h=1080, ratio=0.5
):
    """
    Copy from jianglei.huang's HDflow repo.

    project points from camera coordinate to image coordinate,
    using pinhole model
    """

    # Get valid point index
    # print("img_w: %d, img_h: %d"%(img_w, img_h))
    # print(points_cam.shape)
    wl = -ratio * img_w
    hl = -ratio * img_h
    wh = (1 + ratio) * img_w
    hh = (1 + ratio) * img_h
    points_cam = copy.deepcopy(points_cam_input)
    box_h = np.max(points_cam[:, 1]) - np.min(points_cam[:, 1])
    z_valid_index = np.squeeze(np.argwhere(points_cam[:, 2] > VALID_Z))
    if z_valid_index.size <= 2:
        return None
    z_valid_points = points_cam[z_valid_index, :]
    P2 = np.array(P2)
    z_valid_points_image = cam2img_pinhole_linear_kernel(z_valid_points, P2)
    valid_point_index = np.squeeze(
        np.argwhere(
            np.logical_and(
                np.logical_and(
                    z_valid_points_image[:, 0] > wl,
                    z_valid_points_image[:, 0] < wh,
                ),
                np.logical_and(
                    z_valid_points_image[:, 1] > hl,
                    z_valid_points_image[:, 1] < hh,
                ),
            )
        )
    )
    if valid_point_index.size <= 2:
        return None
    valid_point_original = z_valid_index[valid_point_index]
    valid_point_cam = points_cam[valid_point_original, :]

    valid_points_indx_set = set(valid_point_original)
    for i in range(points_cam.shape[0]):
        pp = points_cam[i, :]
        if i not in valid_points_indx_set:
            candi_anchors = valid_point_cam[
                np.abs(pp[1] - valid_point_cam[:, 1]) < box_h * 0.4, :
            ]
            if candi_anchors.shape[0] == 0:
                print("Error, there should be at least one valid point")
                return None
            elif candi_anchors.shape[0] == 1:
                anchor = candi_anchors[0, :]
            else:
                dist = np.sum(
                    np.square(candi_anchors[:, [0, 2]] - pp[[0, 2]]), axis=-1
                )
                anchor = candi_anchors[np.argsort(dist)[-2]]
            l = anchor[0]  # noqa
            r = pp[0]  # noqa
            edge_p = np.zeros((1, 3))
            edge_p[0, 0] = pp[0]
            edge_p[0, 1] = pp[1]
            edge_p[0, 2] = pp[2]
            while np.abs(l - r) > 1e-6:
                mid = (l + r) / 2.0
                z = (pp[2] - anchor[2]) / (pp[0] - anchor[0]) * (
                    mid - pp[0]
                ) + pp[2]
                if z < VALID_Z:
                    r = mid
                    continue
                edge_p[0, 0] = mid
                edge_p[0, 2] = z
                edge_p_image = cam2img_pinhole_linear_kernel(edge_p, P2)
                if (
                    edge_p_image[0, 0] < wl
                    or edge_p_image[0, 0] > wh
                    or edge_p_image[0, 1] < hl
                    or edge_p_image[0, 1] > hh
                ):
                    r = mid
                else:
                    l = mid  # noqa
            points_cam[i, :] = edge_p[0, :]

    image_pts = project_to_image(points_cam, P2, distCoeffs)
    return image_pts


def cam2img_pinhole_linear_kernel(points_cam, P2):
    """Convert points from camera coordinate to image."""
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]

    fx, fy = P2[0, 0], P2[1, 1]
    cx, cy = P2[0, 2], P2[1, 2]
    xp = x * fx + cx
    yp = y * fy + cy
    image_pts = np.hstack((xp.reshape(-1, 1), yp.reshape(-1, 1)))
    return image_pts


def project_to_image(pts_3d, P, dist_coeff=None, fisheye=False):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    P = np.array(P)
    if dist_coeff is not None:
        if not fisheye:
            rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)
            dist_coeff = np.array(dist_coeff)
            image_pts = cv2.projectPoints(
                pts_3d[:, :3], np.array(rvec), tvec, P[:, :3], dist_coeff
            )[0]
            pts_2d = np.squeeze(image_pts)
        else:
            pts_3d = pts_3d[:, :3]
            pts_3d = copy.deepcopy(pts_3d)
            pts_3d[pts_3d[:, 2] < 0, 2] = 0.001
            pts_3d = np.expand_dims(pts_3d, 0)
            rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)
            dist_coeff = np.array(dist_coeff)
            fx, fy = P[0, 0], P[1, 1]
            u, v = P[0, 2], P[1, 2]
            k_ = np.mat([[fx, 0.0, u], [0.0, fy, v], [0.0, 0.0, 1.0]])
            d_ = np.mat(dist_coeff[:4].T)
            image_pts = cv2.fisheye.projectPoints(
                pts_3d, np.array(rvec), tvec, k_, d_
            )[0]
            pts_2d = np.squeeze(image_pts)
    else:
        pts_3d_homo = np.concatenate(
            [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1
        )
        pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d


def _draw_box_3d(image, corners, c=(0, 0, 255), show_arrow=False, thickness=1):
    face_idx = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            try:
                cv2.line(
                    image,
                    (corners[f[j], 0], corners[f[j], 1]),
                    (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )  # noqa
            except:  # noqa
                continue  # noqa
        if not show_arrow:
            if ind_f == 0:
                try:
                    cv2.line(
                        image,
                        (corners[f[0], 0], corners[f[0], 1]),
                        (corners[f[2], 0], corners[f[2], 1]),
                        c,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )  # noqa
                except:  # noqa
                    continue  # noqa
                try:
                    cv2.line(
                        image,
                        (corners[f[1], 0], corners[f[1], 1]),
                        (corners[f[3], 0], corners[f[3], 1]),
                        c,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )  # noqa
                except:  # noqa
                    continue

        # # show an arrow to indicate 3D orientation of the object
        # if show_arrow:
        #     # 4,5,6,7
        #     p1 = (corners[0, :] + corners[1, :] +
        #           corners[2, :] + corners[3, :]) / 4
        #     p2 = (corners[0, :] + corners[1, :]) / 2
        #     p3 = p2 + (p2 - p1) * 0.5

        #     p1 = p1.astype(np.int32)
        #     p2 = p2.astype(np.int32)
        #     p3 = p3.astype(np.int32)

        #     cv2.line(image, (p1[0], p1[1]), (p3[0], p3[1]),
        #              c, thickness, lineType=cv2.LINE_AA)
    return image


def load_calib(attri_file, cameras):
    """Load calibration from attribute.json.

    Args:
        attri_file ([str]): [the attribute json path]
        cameras ([list]): [camera view names]

    Returns:
        [dict]: [dict contains each cam's calibration]
    """
    lidar_calibs = defaultdict(dict)
    calibration = json.load(open(attri_file, "r"))["calibration"]
    for cam in cameras:
        lidar_calib = get_calib(calibration, cam)
        lidar_calibs[cam] = lidar_calib
    lidar2chassis = np.array(
        calibration["lidar_top_2_chassis"], dtype=np.float
    )
    return lidar_calibs, lidar2chassis


def get_calib(calibs, cam):
    """Get each view's calibration from all calibs].

    Args:
        calibs ([dict]): [calibs contains all cameras]
        cam ([str]): [cam view]

    Returns:
        [dict]: [dict contains cam's calibration]
    """
    res_calib = defaultdict()
    # get intrinics and disCoeffs
    res_calib["P2"] = np.array(calibs[cam]["K"], dtype=np.float)
    res_calib["disCoeffs"] = np.array(calibs[cam]["d"], dtype=np.float)
    # get Transformation (lidar->cam)
    Tr_key = "lidar_top_2_" + cam
    res_calib["Tr_vel2cam"] = np.array(calibs[Tr_key], dtype=np.float)
    res_calib["rotMat"] = res_calib["Tr_vel2cam"][:3, :3]
    res_calib["transMat"] = res_calib["Tr_vel2cam"][:3, 3]
    return res_calib


def get_3dboxcorner_in_vcs_numpy(box3d):
    """Convert from bev3d label to 3dbox corner in vcs coordinate.

    Args:
        box3d (np.ndarray): input boxes, shape=[N,7]
    Returns:
        np.ndarray: the 8 corners on vcs coordinate.
    """

    box3d = box3d[:, [0, 1, 2, 5, 4, 3, 6]]
    # get normalized corners and multiply with dimension
    # (box height, width, length)
    corner = np.array(
        [
            [-1, -1, 1, 1, -1, -1, 1, 1],  # x
            [1, -1, -1, 1, 1, -1, -1, 1],  # y
            [-1, -1, -1, -1, 1, 1, 1, 1],
        ]
    )  # z
    dim = (
        np.expand_dims(box3d[:, 3:6], axis=1) / 2
    )  # modify dim->1/2 dim, shape:[N,1,3]
    corner_nd = (
        dim * corner.transpose()
    )  # generate the corner based on (h,w,l), shape:[N,8,3]
    corner_nd = np.transpose(corner_nd, (0, 2, 1))  # N,3,8

    # Rotation matrix, shape=[N, 3, 3]
    yaw = box3d[:, -1]
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    rot_mat = np.stack(
        [
            [rot_cos, -rot_sin, zeros],
            [rot_sin, rot_cos, zeros],
            [zeros, zeros, ones],
        ]
    )  # [3, 3, N]
    rot_mat = np.transpose(rot_mat, (2, 0, 1))  # [N, 3, 3]

    bbox_corner = np.matmul(rot_mat, corner_nd)  # [N,3,8]
    bbox_corner = np.transpose(bbox_corner, (0, 2, 1))  # [N,8,3]

    # NOTE: each box: rot @ corner -> rot corner
    # bbox_corner = np.einsum("aij,jka->aik", corner_nd, rot_mat)

    center = np.stack((box3d[:, 0], box3d[:, 1], box3d[:, 2]), 1)
    center = np.expand_dims(center, 1)

    # shift the normalized box to center
    bbox_corner += center
    return bbox_corner


def draw_color_bbox(
    bboxes,
    bev_image,
    cls_id,
    score,
    bev_size,
    bev_range,
    scor_thr=0.2,
    thickness=1,
    color=(0, 255, 0),
):
    """Draw colored image based on vcs corner.

    Args:
        bboxes (np.ndarray): the box info, shape=[N, 8, 3], (8, 3) means:
            each box's 8 corner coordinate (x,y,z) in vcs coord
        bev_image (np.ndarray): image to draw image.
        cls_id (np.ndarray): objs class id.
        score (np.ndarray): objs score.
        bev_size (tuple, optional): bev image map size. Defaults to (512, 512).
        bev_range (tuple, optional): vcs visible range. Defaults to
            (-30.0, -51.2, 72.4, 51.2).
        scor_thr (float, optional): score_threshold. Defaults to 0.2.
        thickness (int, optional): the thickness og line.
        color (tuple, optional): color of box line.

    Returns:
        np.ndarray: drawed image.
    """
    for bbox, scor, _id in zip(bboxes, score, cls_id):
        if scor > scor_thr:
            p1, p2, p3, p4 = bbox[:4, :2]
            (
                p1_index,
                p2_index,
                p3_index,
                p4_index,
            ) = convert_vcs_coord_to_bev_index_numpy(
                np.stack((p1, p2, p3, p4), axis=0),
                bev_size=bev_size,
                bev_range=bev_range,
            )
            cv2.line(
                bev_image,
                tuple(p1_index[::-1]),
                tuple(p4_index[::-1]),
                color,
                thickness,
            )
            cv2.line(
                bev_image,
                tuple(p4_index[::-1]),
                tuple(p3_index[::-1]),
                color,
                thickness,
            )
            # cv2.line(bev_image, tuple(p3_index[
            #     ::-1]), tuple(p4_index[::-1]), color, 1)

            cv2.line(
                bev_image,
                tuple(p3_index[::-1]),
                tuple(p2_index[::-1]),
                color,
                thickness,
            )
    return bev_image


def convert_vcs_coord_to_bev_index_numpy(
    vcs_coord,
    bev_size=(512, 512),  # bev coord y, x
    bev_range=(-30.0, -51.2, 72.4, 51.2),
):
    """Convert vcs coordinate into bev image coordinate.

    Args:
        vcs_coord (np.ndarray, shape=[N,2]): the vcs coordinate, (x,y)
        bev_size (tuple, optional): bev image map size. Defaults to (512, 512).
        bev_range (tuple, optional): vcs visible range. Defaults to
            (-30.0, -51.2, 72.4, 51.2).

    Returns:
        np.ndarray: (N, 2)  (y, x) index of (512, 512) bev image map.
    """

    voxel_size = (
        abs(bev_range[2] - bev_range[0]) / bev_size[0],
        abs(bev_range[3] - bev_range[1]) / bev_size[1],
    )

    bev_index_y = np.floor(
        (bev_range[2] - vcs_coord[:, 0]) / voxel_size[0]
    ).astype(int)
    bev_index_x = np.floor(
        (bev_range[3] - vcs_coord[:, 1]) / voxel_size[1]
    ).astype(int)

    bev_index = np.stack([bev_index_y, bev_index_x], axis=1)
    return bev_index


def cv2_draw_lines(img, lines, colors, thickness, line_type=cv2.LINE_8):
    lines = lines.astype(np.int32)
    for line, color in zip(lines, colors):
        color = list(int(c) for c in color)  # noqa [C400]
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return img


def rotation_2d(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def corner_to_standup_nd(boxes_corner):
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.

    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def draw_box_in_bev(
    img, coors_range, boxes, color, thickness=1, labels=None, label_color=None
):
    coors_range = np.array(coors_range)
    bev_corners = center_to_corner_box2d(
        boxes[:, [0, 1]], boxes[:, [3, 4]], boxes[:, 6]
    )
    bev_corners -= coors_range[:2]
    bev_corners *= np.array(img.shape[:2])[::-1] / (
        coors_range[3:5] - coors_range[:2]
    )
    standup = corner_to_standup_nd(bev_corners)
    text_center = standup[:, 2:]
    text_center[:, 1] -= (standup[:, 3] - standup[:, 1]) / 2

    bev_lines = np.concatenate(
        [bev_corners[:, [0, 2, 3]], bev_corners[:, [1, 3, 0]]], axis=2
    )
    bev_lines = bev_lines.reshape(-1, 4)
    colors = np.tile(np.array(color).reshape(1, 3), [bev_lines.shape[0], 1])
    colors = colors.astype(np.int32)
    img = cv2_draw_lines(img, bev_lines, colors, thickness)
    if labels is not None:
        if label_color is None:
            label_color = colors
        else:
            label_color = np.tile(
                np.array(label_color).reshape(1, 3), [bev_lines.shape[0], 1]
            )
            label_color = label_color.astype(np.int32)

        img = cv2_draw_text(  # noqa [F821]
            img, text_center, labels, label_color, thickness * 2
        )
    return img


def draw_kitti_pred_in_bev(
    bbox_pred,
    bev_map,
    point_cloud_range,
    label_pred,
    color=[0, 0, 255],  # noqa [B006]
):
    """Draw kitti predictions in bev.

    The reason why we have draw_kitti_gt_in_bev and
    draw_kitti_pred_in_bev is that we want to add features to
    draw preds and gt for all 3 different categories.

    args
    ----
    bbox_gt: numpy.array of shape (num_objs, 8)
        Ground truth of bbox.
    label_gt: list of labels in natural language
        The labels of the bbox_gt in same order.
    bev_map: numpy.array
        The image which has been drew in previous steps
    point_cloud_range: list
        Show the range of the detection range.

    returns
    -------
    bev_map: numpy.array
    """
    # TODO (runzhou.ge) Use the label_gt to draw different colors for
    # different classes
    if bbox_pred.shape[0] > 0:
        return draw_box_in_bev(bev_map, point_cloud_range, bbox_pred, color, 2)
    else:
        return bev_map


def read_lidar(path, dim=6):
    return (
        np.fromfile(path, dtype=np.double).reshape(-1, dim).astype(np.float32)
    )


# @numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    coor_to_voxelidx,
    # coors_2d,
    bev_map,
    height_lowers,
    # density_norm_num=16,
    with_reflectivity=False,
    max_voxels=40000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    coor = np.zeros(shape=(3,), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]
        ) / height_slice_size
        if incomimg_height_norm > height_norm:
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(
    points,
    voxel_size,
    coors_range,
    with_reflectivity=False,
    density_norm_num=16,
    max_voxels=40000,
):
    """Convert kitti points(N, 4) to a bev map. return [C, H, W] map.

    this function based on algorithm in points_to_voxel.
    takes 5ms in a reduced pointcloud with voxel_size=[0.1, 0.1, 0.8]

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3] contain reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        with_reflectivity: bool. if True, will add a intensity map to bev map.
    Returns:
        bev_map: [num_height_maps + 1(2), H, W] float tensor.
            `WARNING`: bev_map[-1] is num_points map, NOT density map,
            because calculate density map need more time in cpu rather
            than gpu. if with_reflectivity is True, bev_map[-2] is
            intensity map.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] += 1
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False
    )
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        coor_to_voxelidx,
        bev_map,
        height_lowers,
        with_reflectivity,
        max_voxels,
    )
    # print(voxel_num)
    return bev_map


def point_to_vis_bev(
    points, voxel_size=None, coors_range=None, max_voxels=80000
):
    if voxel_size is None:
        voxel_size = [0.1, 0.1, 0.1]
    if coors_range is None:
        coors_range = [-50, -50, -3, 50, 50, 1]
    voxel_size[2] = coors_range[5] - coors_range[2]
    bev_map = points_to_bev(
        points, voxel_size, coors_range, max_voxels=max_voxels
    )
    height_map = (bev_map[0] * 255).astype(np.uint8)
    return cv2.cvtColor(height_map, cv2.COLOR_GRAY2RGB)
