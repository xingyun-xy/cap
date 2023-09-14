# Copyright (c) Changan Auto. All rights reserved.
from typing import List, Optional

import cv2
import msgpack
import numpy as np
import torch
import torch.utils.data as data
import torchvision

from cap.registry import OBJECT_REGISTRY
from .data_packer import Packer
from .pack_type import PackTypeMapper
from .pack_type.utils import get_packtype_from_path

__all__ = ["ImageNet", "ImageNetPacker", "ImageNetFromImage"]


@OBJECT_REGISTRY.register
class ImageNet(data.Dataset):  # noqa: D205,D400
    """
    ImageNet provides the method of reading imagenet data
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
        raw_data = msgpack.unpackb(raw_data, raw=False)
        img_data = raw_data[:-2]
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = img[:, :, ::-1].copy()  # bgr -> rgb
        sample = torch.as_tensor(img).permute(2, 0, 1)

        label_data = raw_data[-2:]
        target = np.frombuffer(label_data, dtype=np.int16).astype(np.int64)
        target = target.squeeze()

        data = {"img": sample, "labels": target}
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return "ImageNet"


class ImageNetPacker(Packer):  # noqa: D205,D400
    """
    ImageNetPacker is used for converting ImageNet dataset
    in torchvision to DataType format.

    Args:
        src_data_dir (str): The dir of original imagenet data.
        target_data_dir (str): Path for LMDB file.
        split_name (str): Split name of data, such as train, val and so on.
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
        **kwargs
    ):
        self.dataset = torchvision.datasets.ImageNet(
            root=src_data_dir, split=split_name, loader=loader
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(ImageNetPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        image, label = self.dataset[idx]
        label = np.uint16(label).tobytes()
        pack_block = msgpack.packb(image + label, use_bin_type=True)
        return pack_block


def loader(data_path):
    img = cv2.imread(data_path)
    img = cv2.imencode(".JPEG", img)[1]
    img = np.asarray(img).astype(np.uint8).tobytes()
    return img


@OBJECT_REGISTRY.register
class ImageNetFromImage(torchvision.datasets.ImageNet):
    """ImageNet from image by torchvison.

    The params of ImageNetFromImage are same as params of
    torchvision.datasets.ImageNet.
    """

    def __init__(self, transforms=None, *args, **kwargs):
        super(ImageNetFromImage, self).__init__(
            transform=transforms, *args, **kwargs
        )

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        data = {"img": sample, "labels": target}
        if self.transform is not None:
            data = self.transform(data)
        return data
