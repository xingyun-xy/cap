# Copyright (c) Changan Auto. All rights reserved.

import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from cap.registry import OBJECT_REGISTRY

__all__ = ["HeartRateDataset"]


@OBJECT_REGISTRY.register
class HeartRateDataset(data.Dataset):
    """HeartRate dataset class.

    Args:
       st_maps_path (str): Paths for stmaps.
       is_training (bool): Whether in the training stage.
       transforms (Transform): Transforms that applies to the data.
       test_list (List): List of val ids.
       mode (str): "vipl" or "syn" or "cockpit".
       Correspond to VIPL stmaps/Synthetic stmaps/Cockpit stmaps.
    """

    def __init__(
        self,
        st_maps_path: str,
        is_training: bool = True,
        transform: Optional[List] = None,
        test_list: Optional[List] = None,
        mode: str = "vipl",
    ):
        self.training = is_training
        self.test_list = test_list
        self.st_maps_path = st_maps_path
        self.transform = transform
        self.mode = mode
        self.all_dir = os.listdir(self.st_maps_path)
        self.test_dir = []
        self.train_dir = []
        for dir_i in self.all_dir:
            if dir_i.split("_")[0] in self.test_list:
                self.test_dir.append(dir_i)
            else:
                self.train_dir.append(dir_i)

    def __len__(self):
        if self.training:
            return len(self.train_dir)
        else:
            return len(self.test_dir)

    def __getitem__(self, idx):
        """About the organization of STmaps.

        There is one spatio-temporal map under a single folder,
        named 'p1_v1_source1_1_0','p1_v1_source1_2_14','p1_v1_source1_3_29',
        which means 'id_scene_source_index_startframe',
            id from 'p1' to 'p107',
            scene from 'v1' to 'v9',
            source from 'source1' to 'source3'(i.e. web cam,phone,RGBD cam),
            index from 1 to num of stmaps,
            startframe represent the stmap calculate from the
            [startframe, startframe+300] frames (video clip) of whole video.
            (300 means clip length, represent the length of a stmap.)
        """
        if self.training:
            dir_idx = self.train_dir[idx]
        else:
            dir_idx = self.test_dir[idx]

        if self.mode == "vipl" or self.mode == "cockpit":
            img_name = str(dir_idx) + "/img_yuv.png"
        else:
            img_name = str(dir_idx) + "/img_syn.png"

        img_path = os.path.join(self.st_maps_path, img_name)
        feature_map = Image.open(img_path)

        if self.transform:
            feature_map = self.transform(feature_map)

        json_path = os.path.join(
            self.st_maps_path, str(dir_idx), "data_info.json"
        )
        with open(json_path, "r") as load_f:
            load_dict = json.load(load_f)

        fps = np.array(load_dict["fps"])
        gt_HR = np.array(load_dict["gt_HR"])
        heart_beats_num = np.array(load_dict["heart_beats_num"])

        return {
            "st_maps": torch.as_tensor(feature_map, dtype=torch.float),
            "fps": torch.as_tensor(torch.from_numpy(fps), dtype=torch.float),
            "gt_HR": torch.as_tensor(
                torch.from_numpy(gt_HR), dtype=torch.float
            ),
            "heart_beats_num": torch.as_tensor(
                torch.from_numpy(heart_beats_num), dtype=torch.float
            ),
        }
