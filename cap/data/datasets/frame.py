import json

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FrameDataset(Dataset):
    def __init__(
        self,
        img_path,
        calib_path,
        buf_only=False,
        transforms=None,
    ):
        data = {}
        data["ori_img"] = cv.imread(img_path)
        data["img_id"] = 0
        if buf_only:
            with open(img_path, "rb") as rf:
                data["img_buf"] = rf.read()
        else:
            data["img"] = data["ori_img"]

        assert calib_path.endswith(".json")

        with open(calib_path, "r") as rf:
            calib_dict = json.load(rf)

        data["calib"] = np.array(calib_dict["calib"])
        data["distCoeffs"] = np.array(calib_dict["distCoeffs"])

        data["layout"] = "hwc"
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if self.transforms is not None:
            data = self.transforms(self.data)
        return data
