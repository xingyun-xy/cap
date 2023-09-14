# Copyright (c) Changan Auto. All rights reserved.
import json
import os
from typing import Callable, Dict, List, Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data
from PIL import Image

from cap.registry import OBJECT_REGISTRY
from .data_packer import Packer
from .mscoco import transforms as kitti_trans
from .pack_type import PackTypeMapper
from .pack_type.utils import get_packtype_from_path

# The goal in the 2D object detection task for kitti is
# to train object detectors for the classes
# 'Car', 'Pedestrian', and 'Cyclist'.
KITTI_DICT = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

__all__ = ["Kitti2DDetection", "Kitti2DDetectionPacker", "Kitti2D"]


class Kitti2DDetection(data.Dataset):
    """Kitti 2D Detection Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file,
            kitti_train.json or kitti_eval.json. (
            For ground truth, we do not use the official txt file format data,
            but use the json file marked by the Changan Auto.
            )
        transforms (callable, optional): A function transform that takes input
            sample and its target as entry and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Optional[Callable] = None,
    ):
        super(Kitti2DDetection, self).__init__()
        self.root = root
        self.annFile = annFile
        self.transforms = transforms
        self.imgs_info_list = self.get_imgs()

    def get_imgs(self):
        imgs_info_list = []
        with open(self.annFile, "r") as js:
            for line in js.readlines():
                img_info = json.loads(line)
                imgs_info_list.append(img_info)
        print(f"{self.annFile} reading completed!")
        return imgs_info_list

    def _load_image(self, img_name: str):
        path = os.path.join(self.root, img_name)
        return Image.open(path).convert("RGB")

    def _load_target(self, targets: List[Dict]):
        label = np.zeros((0, 5))
        for _, target in enumerate(targets):
            if target["attrs"]["type"] not in KITTI_DICT.keys():
                continue
            bbox = [float(coord) for coord in target["data"]]
            if (bbox[2] - bbox[0]) < 1 or (bbox[3] - bbox[1]) < 1:
                continue
            anno = np.zeros((1, 5))
            anno[0, :4] = bbox
            anno[0, 4] = float(KITTI_DICT[target["attrs"]["type"]])
            label = np.append(label, anno, axis=0)
        return label

    def __len__(self):
        return len(self.imgs_info_list)

    def __getitem__(self, index):
        img_info = self.imgs_info_list[index]
        img_name = img_info["image_key"]
        targets = img_info["kitti_vehicle"]
        image = self._load_image(img_name)
        labels = self._load_target(targets)
        if self.transforms is not None:
            image, labels = self.transforms(image, labels)
        height, width, _ = image.shape
        assert height == int(img_info["height"]) or width == int(
            img_info["width"]
        ), (
            f"The height or width of the image {img_name} is "
            f"inconsistent with the given in {self.annFile}"
        )
        if len(labels) == 0:
            print(f"image {img_name} has no bounding boxes")
        return image, labels, img_name


class Kitti2DDetectionPacker(Packer):  # noqa: D205,D400
    """
    Kitti2DDetectionPacker is used for converting
    kitti2D dataset to target DataType format.

    Args:
        src_data_dir (str): The dir of original kitti2D data.
        target_data_dir (str): Path for LMDB file.
        annFile (string): Path to json annotation file,
            kitti_train.json or kitti_eval.json.
        num_workers (int): The num workers for reading data
            using multiprocessing.
        pack_type (str): The file type for packing.
        num_samples (int): the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        annFile: str,
        num_workers: int,
        pack_type: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.dataset = Kitti2DDetection(
            root=src_data_dir,
            annFile=annFile,
            transforms=kitti_trans,
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(Kitti2DDetectionPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        datas = self.dataset[idx]
        image, labels, img_name = datas
        img_name = img_name.encode()
        num_bboxes = labels.shape[0]
        num_bboxes_data = np.asarray(num_bboxes, dtype=np.uint8).tobytes()
        image = cv2.imencode(".png", image)[1].tobytes()
        labels = np.asarray(labels, dtype=np.float).tobytes()
        return msgpack.packb(img_name + num_bboxes_data + labels + image)


@OBJECT_REGISTRY.register
class Kitti2D(data.Dataset):  # noqa: D205,D400
    """Kitti2D provides the method of reading kitti2d data
    from target pack type.

    Args:
        data_path (str): The path of LMDB file.
        transforms (list): Transforms of voc before using.
        pack_type (str): The pack type.
        pack_kwargs (dict): Kwargs for pack type.
    """

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.root = data_path
        self.transforms = transforms
        self.kwargs = {} if pack_kwargs is None else pack_kwargs

        try:
            self.pack_type = get_packtype_from_path(data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raw_data = self.pack_file.read(self.samples[item])
        raw_data = msgpack.unpackb(raw_data, raw=True)
        img_name = raw_data[:10]
        img_name = img_name.decode()
        num_bboxes_data = raw_data[10:11]
        num_bboxes = np.frombuffer(num_bboxes_data, dtype=np.uint8)
        num_bboxes = num_bboxes[0]
        label_data = raw_data[11 : 11 + num_bboxes * 5 * 8]
        label = np.frombuffer(label_data, dtype=np.float)
        label = label.reshape((num_bboxes, 5))
        image_data = raw_data[11 + num_bboxes * 5 * 8 :]
        img = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = img.copy()
        label = label.copy()
        data = {
            "ori_img": img,
            "img_name": img_name,
            "img": img,
            "gt_bboxes": label[:, 0:4],
            "gt_classes": label[:, 4],
            "layout": "hwc",
        }
        if self.transforms is not None:
            data = self.transforms(data)
        return data
