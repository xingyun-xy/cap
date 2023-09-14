# Copyright (c) Changan Auto. All rights reserved.
import copy
from typing import List, Optional

import msgpack
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image

from cap.core.data_struct.img_structures import ImgObjDet
from cap.registry import OBJECT_REGISTRY
from .data_packer import Packer
from .mscoco import transforms as coco_transforms
from .pack_type import PackTypeMapper
from .pack_type.utils import get_packtype_from_path

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

__all__ = ["PascalVOC", "VOCDetectionPacker", "VOCFromImage"]

_PASCAL_VOC_LABELS = {
    "aeroplane": (0, "Vehicle"),
    "bicycle": (1, "Vehicle"),
    "bird": (2, "Animal"),
    "boat": (3, "Vehicle"),
    "bottle": (4, "Indoor"),
    "bus": (5, "Vehicle"),
    "car": (6, "Vehicle"),
    "cap": (7, "Animal"),
    "chair": (8, "Indoor"),
    "cow": (9, "Animal"),
    "diningtable": (10, "Indoor"),
    "dog": (11, "Animal"),
    "horse": (12, "Animal"),
    "motorbike": (13, "Vehicle"),
    "person": (14, "Person"),
    "pottedplant": (15, "Indoor"),
    "sheep": (16, "Animal"),
    "sofa": (17, "Indoor"),
    "train": (18, "Vehicle"),
    "tvmonitor": (19, "Indoor"),
}


@OBJECT_REGISTRY.register
class PascalVOC(data.Dataset):  # noqa: D205,D400
    """
    PascalVOC provides the method of reading voc data
    from target pack type.

    Args:
        data_path (str): The path of packed file.
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

    def __getitem__(self, index):
        raw_data = self.pack_file.read(self.samples[index])
        raw_data = msgpack.unpackb(
            raw_data, object_hook=msgpack_numpy.decode, raw=True
        )
        sample = raw_data[b"image"].astype(np.uint8)

        labels = copy.deepcopy(raw_data[b"label"])
        data = {}
        data["img"] = sample
        data["ori_img"] = sample
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        h, w, _ = sample.shape
        data["img_height"], data["img_width"] = h, w
        data["img_shape"] = sample.shape
        data["gt_bboxes"] = labels[:, :4]
        data["gt_classes"] = labels[:, 4]
        data["gt_difficult"] = labels[:, 5]
        data["img_id"] = np.array([index])

        if self.transforms is not None:
            data = self.transforms(data)

            data["gt_labels"] = torch.cat(
                (data["gt_bboxes"], data["gt_classes"].unsqueeze(-1)), -1
            )

        data["structure"] = ImgObjDet(
            img=data["resized_ori_img"]
            if self.transforms is not None
            else data["ori_img"],
            img_id=index,
            layout="hwc",
            color_space=data["color_space"],
            img_height=data["img_height"],
            img_width=data["img_width"],
        )
        return data

    def __len__(self):
        return len(self.samples)


class VOCDetectionPacker(Packer):
    """
    VOCDetectionPacker is used for packing voc dataset to target format.

    Args:
        src_data_dir (str): Dir of original voc data.
        target_data_dir (str): Path for packed file.
        split_name (str): Split name of data, such as trainval and test.
        num_workers (int): Num workers for reading data using multiprocessing.
        pack_type (str): The file type for packing.
        num_samples (int): the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int,
        pack_type: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        if split_name == "trainval":
            ds_2007 = torchvision.datasets.VOCDetection(
                root=src_data_dir,
                year="2007",
                image_set=split_name,
                transforms=coco_transforms,
            )
            ds_2012 = torchvision.datasets.VOCDetection(
                root=src_data_dir,
                year="2012",
                image_set=split_name,
                transforms=coco_transforms,
            )
            self.dataset = data.dataset.ConcatDataset([ds_2007, ds_2012])
        elif split_name == "test":
            self.dataset = torchvision.datasets.VOCDetection(
                root=src_data_dir,
                year="2007",
                image_set="test",
                transforms=coco_transforms,
            )
        else:
            raise NameError(
                "split name must be trainval or test, but get %s"
                % (split_name)
            )

        if num_samples is None:
            num_samples = len(self.dataset)
        super(VOCDetectionPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        image, label = self.dataset[idx]
        image = image

        h, w, c = image.shape
        label = label["annotation"]
        labels = []
        for obj in label["object"]:
            labels.append(
                [
                    (float(obj["bndbox"]["xmin"]) - 1),
                    (float(obj["bndbox"]["ymin"]) - 1),
                    (float(obj["bndbox"]["xmax"]) - 1),
                    (float(obj["bndbox"]["ymax"]) - 1),
                    float(_PASCAL_VOC_LABELS[obj["name"]][0]),
                    float(obj["difficult"]),
                ]
            )
        labels = np.array(labels)
        return msgpack.packb(
            {"image": image, "label": labels}, default=msgpack_numpy.encode
        )


@OBJECT_REGISTRY.register
class VOCFromImage(torchvision.datasets.VOCDetection):
    """VOC from image by torchvision.

    The params of VOCFromImage is same as params of
    torchvision.dataset.VOCDetection.
    """

    def __init__(self, *args, **kwargs):
        super(VOCFromImage, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        label = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        w, h = img.size
        label = label["annotation"]
        labels = []
        for obj in label["object"]:
            labels.append(
                [
                    (float(obj["bndbox"]["xmin"]) - 1),
                    (float(obj["bndbox"]["ymin"]) - 1),
                    (float(obj["bndbox"]["xmax"]) - 1),
                    (float(obj["bndbox"]["ymax"]) - 1),
                    float(_PASCAL_VOC_LABELS[obj["name"]][0]),
                    float(obj["difficult"]),
                ]
            )
        labels = np.array(labels)

        gt_bboxes = labels[:, :4]
        gt_labels = labels[:, 4]
        gt_difficults = labels[:, 5]

        data = {
            "img": np.array(img),
            "gt_bboxes": gt_bboxes,
            "gt_classes": gt_labels,
            "gt_difficult": gt_difficults,
            "layout": "hwc",
            "color_space": "rgb",
        }

        if self.transforms is not None:
            data = self.transforms(data)

        return data
