# Copyright (c) Changan Auto. All rights reserved.
import copy
import glob
import json
import logging
import os
import os.path as osp

import cv2
import numpy as np
import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY

__all__ = ["PilotTestDataset"]


@OBJECT_REGISTRY.register
class PilotTestDataset(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.

    Args:
        data_path (str): The path where the image is stored.
        transforms (list): List of transform.
        infer_model_type (str): Inference model type, currently only the
            crop model is supported, optional values include
            [`crop_with_resize_quarter`, `crop_wo_resize`].
        to_rgb (bool): Whether to convert to `rgb` color_space.
        return_orig_img (bool): Whether to return an extra original img,
            orig_img can usually be used on visualization.
        image_types (list[str]): The format list of images that needs to
            read.
        skip_first_frame: Skip first frame to get compatible with resflow task.

    """

    def __init__(
        self,
        data_path,
        im_hw,
        transforms=None,
        infer_model_type=None,
        buf_only=False,
        to_rgb=False,
        return_orig_img=False,
        image_types=None,
        skip_first_frame: bool = False,
    ):
        self.data_path = data_path
        self.im_hw = im_hw  # img size before net
        self.buf_only = buf_only
        self.transforms = transforms
        if image_types is None:
            image_types = [".jpeg", ".png", ".jpg"]
        self.image_types = copy.deepcopy(image_types)
        # sometimes image_name like this
        # '594975_2/data/ADAS_20210306-112355_482_0__135656_1615001259481_0.jpg',
        # we can't just get the base name of this image or we'll get an error
        # in the eval process.
        (
            self.image_path_list,
            self.image_name_list,
            self.image_annos,
        ) = get_image_info(self.data_path, self.image_types)
        if skip_first_frame:
            self.image_name_list = self.image_name_list[1:]
        self.num_samples = len(self.image_name_list)
        self.to_rgb = to_rgb
        self.return_orig_img = return_orig_img
        self.infer_model_type = infer_model_type

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, item):
        data = {}
        image_path = self.image_path_list[item]
        image = cv2.imread(image_path)
        data["ori_img_shape"] = np.array(image.shape)  # hwc    add by zmj
        image = cv2.resize(image, (self.im_hw[1], self.im_hw[0]))
        # return ori_img, and then resize            change by zmj
        if self.return_orig_img:
            data["ori_img"] = image
            # data["ori_img_shape"] = image.shape     # hwc    add by zmj

        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"
        data["img_name"] = self.image_name_list[item]

        data["img_height"] = image.shape[0]
        data["img_width"] = image.shape[1]
        data["img_id"] = item
        image = image.transpose(2, 0, 1)
        if self.buf_only:
            with open(image_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = image

        if image_path in self.image_annos:
            img_anno = self.image_annos[image_path]
            if isinstance(img_anno, dict):
                if "calib" in img_anno:
                    data["calib"] = np.array(img_anno["calib"],
                                             dtype=np.float32)
                if "distCoeffs" in img_anno:
                    data["distCoeffs"] = np.array(img_anno["distCoeffs"],
                                                  dtype=np.float32)

        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["pad_shape"] = image.shape
        if self.infer_model_type in [
                "crop_with_resize_quarter",
                "crop_wo_resize",
        ]:
            data["camera_info"] = (0 if len(self.image_annos) == 0 else
                                   self.image_annos[image_path])
            data["infer_model_type"] = self.infer_model_type
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        repr_str += f"return_orig_img={self.return_orig_img}"
        return repr_str


def get_image_info(data_path, image_types):
    """Get the path, name and annotation list of all images under the \
    data path."""

    image_path_list = []
    image_name_list = []
    image_annos = {}
    jsonfiles = glob.glob(osp.join(data_path, "*/*.json"))
    if not jsonfiles:
        images = glob.glob(osp.join(data_path, "*/images"))
        jsonfile = None
        if len(images) == 1:
            images = images[0]
    else:
        jsonfile = jsonfiles[0]
        # images = osp.splitext(jsonfile)[0]
        images = data_path
    # When the dir don't have jsonfile or images dir
    # We try to get images from 'self.data_path'
    if jsonfile is None and len(images) == 0:
        for file in sorted(os.listdir(data_path)):
            if osp.splitext(file)[1] in image_types:
                image_path_list.append(osp.join(data_path, file))
                image_name_list.append(file)
    else:
        if jsonfile is not None:
            with open(jsonfile, "r") as fread:
                # annos = fread.readlines()
                # img_anno_list = [json.loads(anno_i) for anno_i in annos]
                annos_json = json.load(fread)  # 3d  zmj
            img_anno_list = annos_json["images"]

            img_url_list = []
            for anno_i in img_anno_list:
                try:
                    annos = {
                        "calib": anno_i["calib"],
                        "distCoeffs": anno_i["distCoeffs"],
                    }
                except KeyError:
                    annos = 0
                image_annos[osp.join(images, anno_i["image_source"])] = annos
                img_url_list.append(osp.join(images, anno_i["image_source"]))
        else:
            img_url_list = []
            img_key_list = []
            for curdir, _dirnames, filenames in os.walk(images):
                for filename in filenames:
                    if osp.splitext(filename)[-1] in image_types:
                        img_url = osp.join(curdir, filename)
                        img_url_list.append(img_url)
                        img_key_list.append(img_url[len(images) + 1:])  # noqa
        for img in img_url_list:
            assert (osp.splitext(img)[1] in image_types
                    ), "%s type must in %s" % (img.split(".")[-1], image_types)
            image_path_list.append(img)
            image_name_list.append(osp.basename(img))

    return image_path_list, image_name_list, image_annos


@OBJECT_REGISTRY.register
class PilotTestDatasetSimple(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.
    Folder structure:
    -- CAM_FRONT
    ---- xxx.json (inclue calib/distCoeffs/shape)
    ---- 1.jpg
    ---- 2.jpg
         ...
    -- CAM_BACK
    ---- xxx.json (inclue calib/distCoeffs/shape)
    ---- 1.jpg
    ---- 2.jpg
         ...
    ...


    Args:
        data_path (str): The path where the image is stored.
        transforms (list): List of transform.
        infer_model_type (str): Inference model type, currently only the
            crop model is supported, optional values include
            [`crop_with_resize_quarter`, `crop_wo_resize`].
        to_rgb (bool): Whether to convert to `rgb` color_space.
        return_orig_img (bool): Whether to return an extra original img,
            orig_img can usually be used on visualization.
        image_types (list[str]): The format list of images that needs to
            read.
        skip_first_frame: Skip first frame to get compatible with resflow task.

    """

    def __init__(
        self,
        data_path,
        im_hw,
        transforms=None,
        infer_model_type=None,
        buf_only=False,
        to_rgb=False,
        return_orig_img=False,
        image_types=None,
        skip_first_frame: bool = False,
    ):
        self.data_path = data_path
        self.im_hw = im_hw  # img size before net
        self.buf_only = buf_only
        self.transforms = transforms
        if image_types is None:
            image_types = [".jpeg", ".png", ".jpg"]
        self.image_types = copy.deepcopy(image_types)
        # sometimes image_name like this
        # '594975_2/data/ADAS_20210306-112355_482_0__135656_1615001259481_0.jpg',
        # we can't just get the base name of this image or we'll get an error
        # in the eval process.
        (
            self.image_path_list,
            self.camera_info_list,
        ) = get_image_info_simple(self.data_path, self.image_types)
        if skip_first_frame:
            self.image_path_list = self.image_path_list[1:]
        self.num_samples = len(self.image_path_list)
        self.to_rgb = to_rgb
        self.return_orig_img = return_orig_img
        self.infer_model_type = infer_model_type

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, item):
        data = {}
        image_path = self.image_path_list[item]
        camera_info = self.camera_info_list[item]

        image = cv2.imread(image_path)
        # ori_img_shape 优先用json文件中的ori_img_shape字段，没有的话用读取图像的原始shape
        data["ori_img_shape"] = (np.array(camera_info["ori_img_shape"])
                                 if "ori_img_shape" in camera_info else
                                 np.array(image.shape))
        image = cv2.resize(image, (self.im_hw[1], self.im_hw[0]))

        if self.return_orig_img:
            data["ori_img"] = image

        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"
        data["img_name"] = osp.basename(image_path)

        data["img_height"] = image.shape[0]
        data["img_width"] = image.shape[1]
        data["img_id"] = 1
        # data["img_id"] = item
        data["img_index"] = item
        image = image.transpose(2, 0, 1)
        if self.buf_only:
            with open(image_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = image

        # load camera info
        data["calib"] = np.array(camera_info["calib"], dtype=np.float32)
        data["distCoeffs"] = np.array(camera_info["distCoeffs"],
                                      dtype=np.float32)

        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["pad_shape"] = image.shape

        if self.transforms is not None:
            data = self.transforms(data)
        """
        配合BEV输入的临时方案
        """
        import torch

        data["sensor2ego_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["intrin_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["ida_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["sensor2sensor_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["bda_mat"] = (torch.rand((1, 4, 4)), )  # ),
        data["gt_boxes_batch"] = (torch.rand((17, 9)), )
        data["gt_labels_batch"] = (torch.rand((17)), )
        data["img_metas_batch"] = []

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        repr_str += f"return_orig_img={self.return_orig_img}"
        return repr_str


def get_image_info_simple(data_path, image_types):
    """Get the path, name and annotation list of all images under the \
    data path."""

    image_path_list, camera_info_list = [], []
    folders = os.listdir(data_path)
    for folder in folders:
        if os.path.isfile(os.path.join(data_path, folder)):
            folders.remove(folder)
        if "gt" in folders:
            folders.remove("gt")
    logging.info(f"finding {len(folders)} folders")
    jsonfiles = glob.glob(osp.join(data_path, "*/*.json"))

    if jsonfiles:
        assert len(folders) == len(jsonfiles)

    for folder in sorted(folders):
        cur = osp.join(data_path, folder)
        cur_images = []
        for image_type in image_types:
            cur_images.extend(glob.glob(osp.join(cur, "*" + image_type)))
        image_path_list.extend(cur_images)
        jsonfile = glob.glob(osp.join(cur, "*.json"))
        # with camera json file
        if jsonfile:
            with open(jsonfile[0], "r") as fread:
                caminfo_ = json.load(fread)
                try:
                    caminfo = {
                        "calib": caminfo_["calib"],
                        "distCoeffs": caminfo_["distCoeffs"],
                        "ori_img_shape": caminfo_["shape"],
                    }
                except KeyError as ke:
                    raise KeyError(
                        f"Missing key parameters in {jsonfile}: {ke}")
                camera_info_list.extend([caminfo] * len(cur_images))
        else:
            # 如果没有提供json文件，给定一个默认内参以保证3d推理不报错，但是推理结果是不正确的
            logging.warning(
                "No camera info file, please check it,"
                "otherwise default camera parameters will be used.")
            default_calib = [
                [1252.8131021185304, 0.0, 826.588114781398, 0.0],
                [0.0, 1252.8131021185304, 469.9846626224581, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
            default_distcoeffs = [[0, 0, 0, 0, 0, 0, 0, 0]]

            caminfo = {
                "calib": default_calib,
                "distCoeffs": default_distcoeffs,
            }

            camera_info_list.extend([dict(caminfo)] * len(cur_images))

    return image_path_list, camera_info_list


# TODO @xuefangwang
def get_image_info_eval(data_path, image_types):
    """Get the path, name and annotation list of all images under the \
    data path."""
    folders = os.listdir(data_path)
    for folder in folders:
        if os.path.isfile(os.path.join(data_path, folder)):
            folders.remove(folder)
        if "gt" in folders:
            folders.remove("gt")
    logging.info(f"finding {len(folders)} folders")
    jsonfiles = glob.glob(osp.join(data_path, "*.json"))

    image_path_list = []
    image_id_path = {}
    imgid2anno = {}

    if jsonfiles:
        assert len(folders) == len(jsonfiles)

    # with camera json file
    if jsonfiles is not None:
        with open(jsonfiles[0], "r") as fread:
            annos = fread.readlines()
        img_anno_list = [json.loads(anno_i) for anno_i in annos]

        images = img_anno_list[0]["images"]
        annotations = img_anno_list[0]["annotations"]

        for im_info in images:
            filename = im_info["image_source"]
            img_url = os.path.join(data_path, filename)

            image_path_list.append(img_url)
            image_id_path[img_url] = im_info["id"]
            imgid2anno[im_info["id"]] = {
                "calib": im_info["calib"],
                "distCoeffs": im_info["distCoeffs"],
                "ori_img_shape": im_info["shape"],
                "annotations": [],
            }

        # objs = []

        for anno in annotations:
            img_id = anno["image_id"]
            if img_id in imgid2anno:
                # objs.append(anno)
                imgid2anno[img_id]["annotations"].append(anno)
                # imgid2anno[img_id]["annotations"] += [anno]

        # if not objs == []:
        #     imgid2anno[img_id]["annotations"] = objs

        # imgid2anno[img_id]["annotations"] = objs

    return image_path_list, image_id_path, imgid2anno


@OBJECT_REGISTRY.register
class PilotEvalDataset(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.
    Folder structure:
    -- CAM_FRONT
    ---- xxx.json (inclue calib/distCoeffs/shape)
    ---- 1.jpg
    ---- 2.jpg
         ...
    -- CAM_BACK
    ---- xxx.json (inclue calib/distCoeffs/shape)
    ---- 1.jpg
    ---- 2.jpg
         ...
    ...


    Args:
        data_path (str): The path where the image is stored.
        transforms (list): List of transform.
        infer_model_type (str): Inference model type, currently only the
            crop model is supported, optional values include
            [`crop_with_resize_quarter`, `crop_wo_resize`].
        to_rgb (bool): Whether to convert to `rgb` color_space.
        return_orig_img (bool): Whether to return an extra original img,
            orig_img can usually be used on visualization.
        image_types (list[str]): The format list of images that needs to
            read.
        skip_first_frame: Skip first frame to get compatible with resflow task.

    """

    def __init__(
        self,
        data_path,
        im_hw,
        transforms=None,
        infer_model_type=None,
        buf_only=False,
        to_rgb=False,
        return_orig_img=False,
        image_types=None,
        skip_first_frame: bool = False,
    ):
        self.data_path = data_path
        self.im_hw = im_hw  # img size before net
        self.buf_only = buf_only
        self.transforms = transforms
        if image_types is None:
            image_types = [".jpeg", ".png", ".jpg"]
        self.image_types = copy.deepcopy(image_types)
        # sometimes image_name like this
        # '594975_2/data/ADAS_20210306-112355_482_0__135656_1615001259481_0.jpg',
        # we can't just get the base name of this image or we'll get an error
        # in the eval process.
        (
            self.image_path_list,
            self.image_id_path,
            self.image_annos,
        ) = get_image_info_eval(self.data_path, self.image_types)
        if skip_first_frame:
            self.image_path_list = self.image_path_list[1:]
        self.num_samples = len(self.image_path_list)
        self.to_rgb = to_rgb
        self.return_orig_img = return_orig_img
        self.infer_model_type = infer_model_type

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, item):
        data = {}
        image_path = self.image_path_list[item]
        image_id = self.image_id_path[image_path]
        img_anno = self.image_annos[image_id]

        image = cv2.imread(image_path)
        # ori_img_shape 优先用json文件中的ori_img_shape字段，没有的话用读取图像的原始shape
        data["ori_img_shape"] = (np.array(img_anno["ori_img_shape"])
                                 if "ori_img_shape" in img_anno else np.array(
                                     image.shape))
        image = cv2.resize(image, (self.im_hw[1], self.im_hw[0]))

        if self.return_orig_img:
            data["ori_img"] = image

        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"
        data["img_name"] = osp.basename(image_path)

        data["img_height"] = image.shape[0]
        data["img_width"] = image.shape[1]
        data["img_id"] = image_id
        data["img_index"] = item
        image = image.transpose(2, 0, 1)
        if self.buf_only:
            with open(image_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = image

        # load image info
        if image_id in self.image_annos:
            img_anno = self.image_annos[image_id]
            if isinstance(img_anno, dict):
                if "calib" in img_anno:
                    data["calib"] = np.array(img_anno["calib"],
                                             dtype=np.float32)
                if "distCoeffs" in img_anno:
                    data["distCoeffs"] = np.array(img_anno["distCoeffs"],
                                                  dtype=np.float32)
                if "annotations" in img_anno:
                    data["annotations"] = img_anno["annotations"]

        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["pad_shape"] = image.shape

        if self.transforms is not None:
            data = self.transforms(data)
        """
        配合BEV输入的临时方案
        """
        import torch

        data["sensor2ego_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["intrin_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["ida_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["sensor2sensor_mats"] = (torch.rand((1, 1, 6, 4, 4)), )
        data["bda_mat"] = (torch.rand((1, 4, 4)), )  # ),
        data["gt_boxes_batch"] = (torch.rand((17, 9)), )
        data["gt_labels_batch"] = (torch.rand((17)), )
        data["img_metas_batch"] = []
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        repr_str += f"return_orig_img={self.return_orig_img}"
        return repr_str
