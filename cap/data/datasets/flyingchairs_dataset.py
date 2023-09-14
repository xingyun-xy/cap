# Copyright (c) Changan Auto. All rights reserved.
import os
from typing import Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data
from skimage.io import imread

from cap.registry import OBJECT_REGISTRY
from .data_packer import Packer
from .pack_type import PackTypeMapper
from .pack_type.utils import get_packtype_from_path

__all__ = ["FlyingChairs", "FlyingChairsFromImage", "FlyingChairsPacker"]


@OBJECT_REGISTRY.register
class FlyingChairs(data.Dataset):  # noqa: D205,D400
    """
    FlyingChairs provides the method of reading flyingChairs data
    from target pack type.

    Args:
        data_path: The path of packed file.
        transforms: Transfroms of data before using.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
        to_rgb: Whether to convert to `rgb` color_space.
    """

    def __init__(
        self,
        data_path: str,
        transforms: list = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        to_rgb: bool = True,
    ):

        self.data_path = data_path
        self.transforms = transforms
        self.samples = None
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        try:
            self.pack_type = get_packtype_from_path(self.data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())
        self.pack_file = None
        self.to_rgb = to_rgb

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raw_data = self.pack_file.read(self.samples[item])
        raw_data = msgpack.unpackb(raw_data, raw=True)
        data = {}
        data["img_name"] = (
            "%05d" % np.frombuffer(raw_data[:8], dtype=np.int64)[0]
        )
        data_shape = np.frombuffer(
            raw_data[8 : 8 + 2 * 3], dtype=np.uint16
        ).astype(np.int64)

        image_len = np.frombuffer(
            raw_data[8 + 2 * 3 : 8 + 2 * 3 + 2 * 8], dtype=np.int64
        ).astype(np.int64)

        total_image_data = np.frombuffer(
            raw_data[
                8
                + 2 * 3
                + 2 * 8 : 8
                + 2 * 3
                + 2 * 8
                + image_len[0]
                + image_len[1]
            ],
            dtype=np.uint8,
        )
        image1 = cv2.imdecode(
            total_image_data[: image_len[0]], cv2.IMREAD_COLOR
        )
        image2 = cv2.imdecode(
            total_image_data[image_len[0] :], cv2.IMREAD_COLOR
        )

        gt_flow_data = np.frombuffer(
            raw_data[8 + 2 * 3 + 2 * 8 + image_len[0] + image_len[1] :],
            dtype=np.float,
        )
        gt_flow = gt_flow_data.reshape((data_shape[0], data_shape[1], 2))

        color_space = "rgb"
        if not self.to_rgb:
            cv2.cvtColor(image1, cv2.COLOR_RGB2BGR, image1)
            cv2.cvtColor(image2, cv2.COLOR_RGB2BGR, image2)
            color_space = "bgr"

        data["img_height"] = data_shape[0]
        data["img_width"] = data_shape[1]
        data["img_id"] = item
        data["img"] = np.concatenate((image1, image2), axis=2)
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image1.shape
        data["gt_flow"] = gt_flow

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        return repr_str


@OBJECT_REGISTRY.register
class FlyingChairsFromImage(data.Dataset):
    """Dataset which gets img data from the data_path.

    Args:
        data_path: The path where the image and gt_flow is stored.
        transforms: List of transform.
        to_rgb: Whether to convert to `rgb` color_space.
        train_flag: Whether the data use to train or test.
        image1_name: The name suffix of image1.
        image2_name: The name suffix of image2.
        image_type: The image type of image1 and image2.
        flow_name: The name suffix of flow.
        flow_type: The flow type of flow.
    """

    def __init__(
        self,
        data_path: str,
        transforms: list = None,
        to_rgb: bool = True,
        train_flag: bool = False,
        image1_name: str = "_img1",
        image2_name: str = "_img2",
        image_type: str = ".ppm",
        flow_name: str = "_flow",
        flow_type: str = ".flo",
    ):

        assert flow_type in [
            ".flo",
            ".pfm",
            ".png",
        ], "flow_type must is .flo, .pfm, or .png."
        self.data_path = data_path
        self.transforms = transforms
        with open(self.data_path + "/FlyingChairs_train_val.txt", "r") as f:
            train_val_IDs = f.readlines()
        train_val_IDs = list(map(int, train_val_IDs))
        label = 2
        if train_flag:
            label = 1
        self.image_name_list = [
            os.path.join(
                self.data_path, "FlyingChairs_release", "data", "%05d"
            )
            % (idx + 1)
            for idx, value in enumerate(train_val_IDs)
            if value == label
        ]
        self.num_samples = len(self.image_name_list)
        self.to_rgb = to_rgb
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.image_type = image_type
        self.flow_name = flow_name
        self.flow_type = flow_type

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        data = {}
        image1 = imread(
            self.image_name_list[item] + self.image1_name + self.image_type
        )
        image2 = imread(
            self.image_name_list[item] + self.image2_name + self.image_type
        )
        gt_flow = flow_read(
            self.image_name_list[item] + self.flow_name + self.flow_type
        )

        color_space = "rgb"
        if not self.to_rgb:
            cv2.cvtColor(image1, cv2.COLOR_RGB2BGR, image1)
            cv2.cvtColor(image2, cv2.COLOR_RGB2BGR, image2)
            color_space = "bgr"

        data["img_name"] = self.image_name_list[item].split("/")[-1]
        data["img_height"] = image1.shape[0]
        data["img_width"] = image1.shape[1]
        data["img_id"] = item
        data["img"] = np.concatenate((image1, image2), axis=2)
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image1.shape
        data["gt_flow"] = gt_flow

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        return repr_str


@OBJECT_REGISTRY.register
class FlyingChairsPacker(Packer):  # noqa: D205,D400
    """
    FlyingChairsPacker is used for converting FlyingChairs dataset
    to target DataType format.

    Args:
        src_data_dir: The dir of original cityscapes data.
        target_data_dir: Path for packed file.
        split_name: Split name of data, such as train, val and so on.
        num_workers: Num workers for reading data using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
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
        assert split_name in [
            "train",
            "val",
        ], "split_name must be one of train and val."
        if split_name == "train":
            train_flag = True
        if split_name == "val":
            train_flag = False
        self.dataset = FlyingChairsFromImage(
            src_data_dir, train_flag=train_flag
        )

        if num_samples is None:
            num_samples = len(self.dataset)
        super(FlyingChairsPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        data = self.dataset[idx]
        image1, image2, label = (
            data["img"][..., :3],
            data["img"][..., 3:],
            data["gt_flow"],
        )
        shape_data = np.asarray(label.shape, dtype=np.uint16).tobytes()
        image_name = np.asarray(data["img_name"], dtype=np.int64).tobytes()
        image1 = cv2.imencode(".png", image1)[1].tobytes()
        image2 = cv2.imencode(".png", image2)[1].tobytes()
        image1_len = len(image1)
        image2_len = len(image2)
        image_len = (image1_len, image2_len)
        image_len = np.asarray(image_len, dtype=np.int64).tobytes()
        label = np.asarray(label, dtype=np.float).tobytes()

        return msgpack.packb(
            image_name + shape_data + image_len + image1 + image2 + label
        )


def flow_read(src_file):
    """
    Read optical flow stored in a .flo, .pfm, or .png file.

    - Interpret bytes as packed binary data.
    Per https://docs.python.org/3/library/struct.html#format-characters:
    format: f -> C Type: float, Python type: float, Standard size: 4.
    format: d -> C Type: double, Python type: float, Standard size: 8.

    - To read optical flow data from 16-bit PNG file:
    https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py.
    Written by Clément Pinard, Copyright (c) 2017 Clément Pinard.
    MIT License.
    - To read optical flow data from PFM file:.
    https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py.
    Written by Ruoteng Li, Copyright (c) 2017 Ruoteng Li.
    License Unknown.
    - To read optical flow data from FLO file:
    https://github.com/daigo0927/PWC-Net_tf/blob/master/flow_utils.py.
    Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka.
    MIT License.

    Args:
        src_file: Path to flow file.
    Returns:
        flow: optical flow in [h, w, 2] format.

    """
    # Read in the entire file, if it exists
    assert os.path.exists(src_file)
    TAG_FLOAT = 202021.25
    if src_file.lower().endswith(".flo"):

        with open(src_file, "rb") as f:

            # Parse .flo file header
            tag = float(np.fromfile(f, np.float32, count=1)[0])
            assert tag == TAG_FLOAT
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]

            # Read in flow data and reshape it
            flow = np.fromfile(f, np.float32, count=h * w * 2)
            flow.resize((h, w, 2))

    elif src_file.lower().endswith(".png"):

        # Read in .png file
        flow_raw = cv2.imread(src_file, -1)

        # Convert from [H,W,1] 16bit to [H,W,2] float formet
        flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
        flow = flow - 32768
        flow = flow / 64

        # Clip flow values
        flow[np.abs(flow) < 1e-10] = 1e-10

        # Remove invalid flow values
        invalid = flow_raw[:, :, 0] == 0
        flow[invalid, :] = 0

    elif src_file.lower().endswith(".pfm"):

        with open(src_file, "rb") as f:

            # Parse .pfm file header
            tag = f.readline().rstrip().decode("utf-8")
            assert tag == "PF"
            dims = f.readline().rstrip().decode("utf-8")
            w, h = map(int, dims.split(" "))
            scale = float(f.readline().rstrip().decode("utf-8"))

            # Read in flow data and reshape it
            flow = np.fromfile(f, "<f") if scale < 0 else np.fromfile(f, ">f")
            flow = np.reshape(flow, (h, w, 3))[:, :, 0:2]
            flow = np.flipud(flow)
    else:
        raise NotImplementedError(
            "Only files in .flo or .pfm or .png format are supported"
        )

    return flow
