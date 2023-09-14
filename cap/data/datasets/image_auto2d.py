# Copyright (c) Changan Auto. All rights reserved.

import copy
import glob
import json
import os

import cv2
import numpy as np
import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY

__all__ = ["Auto2dFromImage"]


@OBJECT_REGISTRY.register
class Auto2dFromImage(data.Dataset):
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
        transforms=None,
        infer_model_type=None,
        buf_only=False,
        to_rgb=False,
        return_orig_img=False,
        image_types=None,
        skip_first_frame: bool = False,
    ):
        self.data_path = data_path
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
        if self.return_orig_img:
            data["ori_img"] = image
        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"
        data["img_name"] = self.image_name_list[item]
        data["img_height"] = image.shape[0]
        data["img_width"] = image.shape[1]
        data["img_id"] = item
        if self.buf_only:
            with open(image_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = image

        if image_path in self.image_annos:
            img_anno = self.image_annos[image_path]
            if isinstance(img_anno, dict):
                if "calib" in img_anno:
                    data["calib"] = np.array(img_anno["calib"])
                if "distCoeffs" in img_anno:
                    data["distCoeffs"] = np.array(img_anno["distCoeffs"])

        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["pad_shape"] = image.shape
        if self.infer_model_type in [
            "crop_with_resize_quarter",
            "crop_wo_resize",
        ]:
            data["camera_info"] = (
                0
                if len(self.image_annos) == 0
                else self.image_annos[image_path]
            )
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
    jsonfiles = glob.glob(os.path.join(data_path, "*/*.json"))
    if not jsonfiles:
        images = glob.glob(os.path.join(data_path, "*/images"))
        jsonfile = None
        if len(images) == 1:
            images = images[0]
    else:
        jsonfile = jsonfiles[0]
        images = os.path.splitext(jsonfile)[0]
    # When the dir don't have jsonfile or images dir
    # We try to get images from 'self.data_path'
    if jsonfile is None and len(images) == 0:
        for file in sorted(os.listdir(data_path)):
            if os.path.splitext(file)[1] in image_types:
                image_path_list.append(os.path.join(data_path, file))
                image_name_list.append(file)
    else:
        if jsonfile is not None:
            with open(jsonfile, "r") as fread:
                annos = fread.readlines()
            img_anno_list = [json.loads(anno_i) for anno_i in annos]

            img_url_list = []
            for anno_i in img_anno_list:
                try:
                    annos = {
                        "calib": anno_i["calib"],
                        "distCoeffs": anno_i["distCoeffs"],
                    }
                except KeyError:
                    annos = 0
                image_annos[os.path.join(images, anno_i["image_key"])] = annos
                img_url_list.append(os.path.join(images, anno_i["image_key"]))
        else:
            img_url_list = []
            img_key_list = []
            for curdir, _dirnames, filenames in os.walk(images):
                for filename in filenames:
                    if os.path.splitext(filename)[-1] in image_types:
                        img_url = os.path.join(curdir, filename)
                        img_url_list.append(img_url)
                        img_key_list.append(img_url[len(images) + 1 :])
        for img in img_url_list:
            assert (
                os.path.splitext(img)[1] in image_types
            ), "%s type must in %s" % (img.split(".")[-1], image_types)
            image_path_list.append(img)
            image_name_list.append(img.replace(images + "/", ""))

    return image_path_list, image_name_list, image_annos
