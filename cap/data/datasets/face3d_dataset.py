# Copyright (c) Changan Auto. All rights reserved.
import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY

__all__ = ["Face3dDataset"]


@OBJECT_REGISTRY.register
class Face3dDataset(data.Dataset):
    """Dataset for 3D Face Reconstruction.

    Read raw image, bbox, 3D face landmark(68pts) and face mask. Return cropped
    face roi and landmark. Target image and mask are also required in
    `finetune` stage.

    Args:
        data_path (str): The path of image files.
        anno_path (str): The path of anno file.
        transforms (list): Transforms of data augmentation.
        stage (str): Data stage. Only `pretrain` and `finetune` are supported.
            In `pretrain` stage, model is only trained with landmark loss.
            In `finetune` stage, model is trained with both landmark loss and
            image-based reconstruction loss. Face mask and target image are
            necessary for the reconstruction loss.

    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        transforms: Optional[List] = None,
        stage: Optional[str] = "finetune",
    ):
        self.data_path = data_path
        with open(anno_path, "r") as f:
            self.annos = json.load(f)
        self.transforms = transforms
        self.stage = stage.lower()
        assert self.stage in [
            "finetune",
            "pretrain",
        ], "stage must be pretrain or finetune."
        self.sym_idxs = [
            [0, 16],
            [1, 15],
            [2, 14],
            [3, 13],
            [4, 12],
            [5, 11],
            [6, 10],
            [7, 9],
            [17, 26],
            [18, 25],
            [19, 24],
            [20, 23],
            [21, 22],
            [31, 35],
            [32, 34],
            [36, 45],
            [37, 44],
            [38, 43],
            [39, 42],
            [40, 47],
            [41, 46],
            [48, 54],
            [49, 53],
            [50, 52],
            [61, 63],
            [60, 64],
            [67, 65],
            [58, 56],
            [59, 55],
        ]

    def __len__(self):
        return len(self.annos)

    def __repr__(self):
        return "Face3dDataset"

    def __getitem__(self, index):
        data = {}
        anno = self.annos[index]
        img_path = os.path.join(self.data_path, anno["img_path"])
        data["img"] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        data["img_shape"] = data["img"].shape
        data["layout"] = "hwc"
        data["gt_bboxes"] = np.array(anno["bbox"], dtype=np.float32)
        data["gt_ldmk"] = np.array(anno["keypoints"], dtype=np.float32)
        data["ldmk_pairs"] = self.sym_idxs

        if self.stage == "finetune":
            mask_path = img_path[:-3] + "png"
            data["gt_mask"] = cv2.imread(mask_path)
            data["gt_img"] = data["img"].copy()
        if self.transforms is not None:
            data = self.transforms(data)
        return data
