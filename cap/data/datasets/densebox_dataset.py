# Copyright (c) Changan Auto. All rights reserved.
"""Dataset for densebox mx-record data, used in auto."""
import contextlib
import getpass
import logging
import os
import tempfile
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY

__all__ = ["DenseboxDataset"]


@OBJECT_REGISTRY.register
class DenseboxDataset(data.Dataset):
    """Dataset for densebox record data in auto, such as adas-mini.

    Args:
        data_path : Path of data relative to buket path.
        anno_path : Path of annotation.
        transforms : List of transform.
        to_rgb: Convert bgr(cv2 imread) to rgb.
        task_type : Consist of 'detection', 'segmentation'
        class_id : the rec's class id, 1base
        category : the used category, 0base
        rec_idx_file_path: index file related to data_path. Used only when
            there is already index file somewhere.
        disable_default_densebox_log: Disable default print output from
            `LegacyDenseBoxImageRecordDataset`. Default is True.
        ignore_hard : Ignore hard instances if `hard` tag in annotation.
            Default is False.
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        transforms: Optional[List] = None,
        to_rgb: Optional[bool] = False,
        task_type: Optional[str] = "detection",
        class_id: Optional[int] = -1,
        category: Optional[int] = -1,
        rec_idx_file_path: Optional[str] = None,
        disable_default_densebox_log: Optional[bool] = True,
        ignore_hard: Optional[bool] = False,
    ):
        assert LegacyDenseBoxImageRecordDataset, (
            "LegacyDenseBoxImageRecordDataset is discarded, "
        )
        self.data_path = data_path
        self.anno_path = anno_path
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.task_type = task_type
        self.class_id = class_id
        self.category = category
        self.ignore_hard = ignore_hard

        if rec_idx_file_path is None:
            rec_idx_file_path = get_idx_path(self.data_path)

        if self.task_type == "detection":
            kwargs = {}
        elif self.task_type == "segmentation":
            kwargs = {"with_seg_label": True, "seg_label_dtype": np.int8}
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(self.task_type)
            )

        try:
            if disable_default_densebox_log:
                temp_fid = tempfile.NamedTemporaryFile("w")
                with contextlib.redirect_stdout(temp_fid):
                    self.dataset = LegacyDenseBoxImageRecordDataset(
                        rec_path=self.data_path,
                        anno_path=self.anno_path,
                        rec_idx_file_path=rec_idx_file_path,
                        # we do to_rgb below
                        to_rgb=False,
                        **kwargs,
                    )
            else:
                self.dataset = LegacyDenseBoxImageRecordDataset(
                    rec_path=self.data_path,
                    anno_path=self.anno_path,
                    rec_idx_file_path=rec_idx_file_path,
                    # we do to_rgb below
                    to_rgb=False,
                    **kwargs,
                )
        except TypeError as e:
            print("DenseboxDataset is discarded!")
            raise e
        logging.info(f"dataset path: {self.data_path}, {self.anno_path}")
        logging.info(f"dataset length: {len(self.dataset)}")

    def __getitem__(self, index: int) -> Dict:
        image, anno = self.dataset[index]
        color_space = "bgr"
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        if self.task_type == "detection":
            anno = anno
        elif self.task_type == "segmentation":
            seg_label = anno[1]
            seg_label = seg_label.astype(np.uint8)
            anno = anno[0]
        anno = anno.to_dict()
        data = {}
        data["img_name"] = anno["img_url"].split("/")[-1]
        data["img_height"] = anno["img_h"]
        data["img_width"] = anno["img_w"]
        data["img_id"] = np.expand_dims(anno["idx"], 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        if self.task_type == "detection":
            gt_bboxes = []
            gt_classes = []
            for ins in anno["instances"]:
                is_hard = ins["is_hard"][0]
                points_data = ins["points_data"]
                class_id = int(ins["class_id"][0])
                bbox = []
                bbox.extend(points_data[0])
                bbox.extend(points_data[2])
                gt_bboxes.append(bbox)
                if class_id == self.class_id:
                    if is_hard and self.ignore_hard:
                        gt_classes.append(-1)
                    else:
                        gt_classes.append(self.category)
                else:
                    gt_classes.append(-1)
            data["gt_bboxes"] = np.array(gt_bboxes)
            data["gt_classes"] = np.array(gt_classes, dtype=np.int64)
        elif self.task_type == "segmentation":
            data["gt_seg"] = seg_label

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return "DenseboxDataset"


def get_idx_path(rec_path: str, root: Optional[str] = None) -> str:
    """
    Get index file path based on rec file and root.

    If root is not provided, idx file will be saved in /cluster_home or /tmp.
    """
    existed_idx_file = rec_path + ".idx"
    if os.path.exists(existed_idx_file):
        return existed_idx_file
    idx_file_name = rec_path.replace("/", "_") + ".idx"
    username = getpass.getuser()
    if root:
        idx_path = os.path.join(root, idx_file_name)
    else:
        if os.path.exists("/cluster_home"):
            root = "/cluster_home/idx_files_genereated_by_hat"
        else:
            root = f"/tmp/idx_files_genereated_by_cap_{username}"
        if not os.path.exists(root):
            os.makedirs(root)
        idx_path = os.path.join(root, idx_file_name)
    return idx_path

