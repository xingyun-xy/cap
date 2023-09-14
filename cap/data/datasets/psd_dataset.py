# Copyright (c) Changan Auto. All rights reserved.

import json

import cv2
import numpy as np
import torch
import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY
from .pack_type import PackTypeMapper
from .pack_type.utils import get_packtype_from_path

__all__ = ["PSDSlotDataset", "PSDTestSlotDataset"]


@OBJECT_REGISTRY.register
class PSDTestSlotDataset(data.Dataset):
    """Parking slot detection test slot dataset.

    Args:
        path (str): The path of dataset file.
        input_size (list): The size of input images.
        pack_type (str): The file type for packing.
        transforms (list): Transforms of psd dataset before using.
    """

    def __init__(
        self, path, input_size=(896, 896), pack_type=None, transforms=None
    ):
        self.path = path
        self.input_size = input_size
        self.is_rgb = True
        self.transforms = transforms

        try:
            self.pack_type = get_packtype_from_path(path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())
        self.pack_file = None

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        kwargs = {}
        self.pack_file = self.pack_type(self.path, writable=False, **kwargs)
        self.pack_file.open()
        self.num_samples = len(self.pack_file.get_keys()) // 2

    def __getitem__(self, index):
        # data_dict
        # {"img_name":xxx, "img": xxx, "ori_img": xxx}
        data_dict = {}
        raw_data = self.pack_file.txn.get(b"img-%09d" % (index + 1))
        name = b"name-%09d" % (index + 1)
        img_name = name.decode()
        data_dict["img_name"] = img_name
        data_buff = np.asarray(bytearray(raw_data), dtype=np.uint8)
        sample = cv2.imdecode(data_buff, cv2.IMREAD_COLOR)
        sample = cv2.resize(sample, self.input_size)
        data_dict["ori_img"] = sample.copy()
        if self.is_rgb:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        data_dict["img"] = sample
        data_dict["layout"] = "hwc"
        data_dict["color_space"] = "rgb" if self.is_rgb else "bgr"
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __len__(self):
        return self.num_samples


@OBJECT_REGISTRY.register
class PSDSlotDataset(data.Dataset):
    """Parking slot detection slot dataset.

    Args:
        path (str): The path of dataset file.
        input_size (list): The size of input images.
        pack_type (str): The file type for packing.
        transforms (list): Transforms of psd dataset before using.
    """

    def __init__(
        self, path, input_size=(448, 448), pack_type=None, transforms=None
    ):
        self.path = path
        self.input_size = input_size
        self.occupancy_dict = {"Y": 0, "N": 1}
        self.slot_type_dict = {
            "Vertical": 0,
            "Parallel": 1,
            "Oblique": 2,
            "Stereo": 0,
            "Ignore": -1,
        }
        self.point_type_dict = {"OJ": 1, "OT": 1, "SJ": 0, "ST": 0}
        self.scene_dict = {"Indoors": 0, "Outdoors": 1}
        self.transforms = transforms
        self.is_rgb = True  # need to be improved

        try:
            self.pack_type = get_packtype_from_path(path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())
        self.pack_file = None

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        kwargs = {}
        self.pack_file = self.pack_type(self.path, writable=False, **kwargs)
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()
        self.num_samples = len(self.samples) // 2

    def gen_slot_angle(self, point_data_list):

        point_data_list = torch.tensor(point_data_list)
        # torch.tensor, (4 x 2)
        diff1 = point_data_list[3] - point_data_list[0]
        diff2 = point_data_list[2] - point_data_list[1]
        diff3 = (point_data_list[3] + point_data_list[2]) - (
            point_data_list[1] + point_data_list[0]
        )
        vec0 = torch.nn.functional.normalize(diff1, dim=0).tolist()
        vec1 = torch.nn.functional.normalize(diff2, dim=0).tolist()
        vec2 = torch.nn.functional.normalize(-diff2, dim=0).tolist()
        vec3 = torch.nn.functional.normalize(-diff1, dim=0).tolist()
        slot_vec = torch.nn.functional.normalize(diff3, dim=0).tolist()
        return vec0, vec1, vec2, vec3, slot_vec

    def gen_slot_center(self, point_data_list):

        point_data_list = np.array(point_data_list)
        center = point_data_list.mean(0)
        return list(center)

    def label_format(self, label):

        total_global_label, total_local_label = [], []
        img_info = json.loads(label.decode())
        img_name = img_info["image_key"]
        scene = img_info["Image"][0]["attrs"]["Domain"]
        img_scene = self.scene_dict[scene]
        if "Slot" not in img_info.keys():
            return [total_global_label, total_local_label, img_name, img_scene]
        slot_info_list = img_info["Slot"]
        for slot_info in slot_info_list:
            slot_type = slot_info["attrs"]["Slot_type"]
            global_label, local_label = [], []
            point_data_list = slot_info["data"]
            point_attrs_list = slot_info["point_attrs"]
            occupancy = slot_info["attrs"]["Occupancy"]
            vec0, vec1, vec2, vec3, slot_vec = self.gen_slot_angle(
                point_data_list
            )
            center = self.gen_slot_center(point_data_list)
            center = [float(i) for i in center]
            global_label.extend(center)
            global_label.append(self.slot_type_dict[slot_type])
            global_label.extend(slot_vec)
            global_label.append(self.occupancy_dict[occupancy])
            for point_data in point_data_list:
                global_label.extend([float(i) for i in point_data])
                local_label.extend([float(i) for i in point_data])
            local_label.extend(vec0)
            local_label.extend(vec1)
            local_label.extend(vec2)
            local_label.extend(vec3)
            local_label.extend(
                [
                    self.point_type_dict[i["point_label"]["Junction"]]
                    for i in point_attrs_list
                ]
            )
            local_label.append(self.slot_type_dict[slot_type])

            total_global_label.append(global_label)
            total_local_label.append(local_label)

        return [total_global_label, total_local_label, img_name, img_scene]

    def __getitem__(self, index):
        # data_dict
        # {"img_name": xxx, "img": xxx,
        #  "ori_img": xxx, "label: xxx"}
        data_dict = {}
        raw_data = self.pack_file.txn.get(b"img-%09d" % (index + 1))
        data_buff = np.asarray(bytearray(raw_data), dtype=np.uint8)
        sample = cv2.imdecode(data_buff, cv2.IMREAD_COLOR)
        sample = cv2.resize(sample, self.input_size)
        data_dict["ori_img"] = sample.copy()
        if self.is_rgb:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        label = self.pack_file.txn.get(b"label-%09d" % (index + 1))
        global_label, local_label, img_name, img_scene = self.label_format(
            label
        )
        data_dict["img"] = sample
        data_dict["label"] = [global_label, local_label]
        data_dict["img_name"] = img_name
        data_dict["layout"] = "hwc"
        data_dict["color_space"] = "rgb" if self.is_rgb else "bgr"
        data_dict["img_scene"] = img_scene
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __len__(self):
        return self.num_samples
