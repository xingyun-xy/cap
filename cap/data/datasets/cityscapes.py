# Copyright (c) Changan Auto. All rights reserved.
import warnings
from typing import Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data
import torchvision
from PIL import Image

from cap.registry import OBJECT_REGISTRY
from .data_packer import Packer
from .pack_type import PackTypeMapper
from .pack_type.utils import get_packtype_from_path

__all__ = ["Cityscapes", "CityscapesPacker", "CITYSCAPES_LABLE_MAPPINGS"]

CITYSCAPES_LABLE_MAPPINGS = (
    (-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1)
    + (2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10)
    + (11, 12, 13, 14, 15, -1, -1, 16, 17, 18)
)


@OBJECT_REGISTRY.register
class Cityscapes(data.Dataset):  # noqa: D205,D400
    """
    Cityscapes provides the method of reading cityscapes data
    from target pack type.

    Args:
        data_path (str): The path of packed file.
        pack_type (str): The pack type.
        transfroms (list): Transfroms of cityscapes before using.
        pack_kwargs (dict): Kwargs for pack type.
    """

    def __init__(
        self,
        data_path: str,
        transforms: list = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.root = data_path
        self.transforms = transforms
        self.samples = None
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
        raw_data = msgpack.unpackb(raw_data, raw=False)

        shape = np.frombuffer(raw_data[: 2 * 2], dtype=np.uint16).astype(
            np.int64
        )
        total_data = np.frombuffer(raw_data[2 * 2 :], dtype=np.uint8)
        label_size = shape[0] * shape[1]
        image = cv2.imdecode(total_data[:-label_size], cv2.IMREAD_COLOR)
        label = total_data[-label_size:].reshape((shape[0], shape[1]))

        image, label = (
            Image.fromarray(image),
            Image.fromarray(label),
        )
        data = {
            "img": image,
            "gt_seg": label,
            "layout": "chw",
            "color_space": "bgr",
        }
        if self.transforms is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data = self.transforms(data)

        return data


class CityscapesPacker(Packer):  # noqa: D205,D400
    """
    CityscapesPacker is used for converting Cityscapes dataset
    in torchvision to target DataType format.

    Args:
        src_data_dir (str): The dir of original cityscapes data.
        target_data_dir (str): Path for packed file.
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
        self.dataset = torchvision.datasets.Cityscapes(
            root=src_data_dir,
            split=split_name,
            mode="fine",
            target_type="semantic",
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(CityscapesPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        label = np.array(label)
        shape_data = np.asarray(label.shape, dtype=np.uint16).tobytes()
        image_data = cv2.imencode(".png", image)[1].tobytes()
        label_data = np.asarray(label, dtype=np.uint8).tobytes()

        return msgpack.packb(
            shape_data + image_data + label_data, use_bin_type=True
        )
