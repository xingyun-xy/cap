# Copyright (c) Changan Auto. All rights reserved.
import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.utils.data as data

from cap.registry import OBJECT_REGISTRY
from .image_auto2d import Auto2dFromImage

__all__ = ["Real3DDataset", "Auto3dFromImage", "Real3DDatasetRec"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Real3DDataset(data.Dataset):
    """Real3D dataset class.

    Args:
        paths (dict): Paths for image roots and annotations.
        num_classes (int): Number of classes.
        transforms (Transform): Transforms that applies to the data.
        num_dist (int): Length of dist coeffs.
        view (str): Camera view.
    """

    def __init__(
        self,
        paths: List[str],
        num_classes: int,
        num_dist: int,
        transforms: Optional[Callable] = None,
        view: Optional[str] = None,
    ):
        super(Real3DDataset, self).__init__()
        self.paths = paths
        self.num_classes = num_classes
        self.transforms = transforms
        self.num_dist = num_dist
        self.view = view

        assert (
            self.num_classes == 3 or self.num_classes == 6
        ), "currently the number of classes must be 3 or 6"

        self._load_annotations(self.paths)
        self.num_samples = len(self._images)

    @staticmethod
    def get_category_id_dict(num_classes):
        if num_classes == 3:
            category_id_dict = {
                1: 0,  # Pedestrian -> Pedestrian
                2: 1,  # Car        -> Car
                3: 2,  # Cyclist    -> Cyclist
                4: 1,  # Bus        -> Car
                5: 1,  # Truck      -> Car
                6: 1,  # SpecialCar -> Car
                7: 1,  # Blur       -> ignore
                8: -99,  # Other    -> ignore
            }  # 'Dontcare' -> Ignore
        elif num_classes == 6:
            category_id_dict = {
                1: 0,  # Pedestrian -> Pedestrian
                2: 1,  # Car        -> Car
                3: 2,  # Cyclist    -> Cyclist
                4: 3,  # Bus        -> Bus
                5: 4,  # Truck      -> Truck
                6: 5,  # SpecialCar -> SpecialCar
                7: -99,  # Blur     -> ignore
                8: -99,  # Other    -> ignore
            }  # 'Dontcare' -> Ignore
        return category_id_dict

    def _load_annotations(self, paths):
        img_dir, anno_path = paths["img_dir"], paths["anno_path"]
        anno_list = []
        if isinstance(img_dir, (list, tuple)):
            assert len(img_dir) == len(
                anno_path
            ), "img_dir and anno_path are not matched, {} vs. {}".format(
                len(img_dir), len(anno_path)
            )

            for _img_dir, _anno_path in zip(img_dir, anno_path):
                anno = json.load(open(_anno_path))
                for im in anno["images"]:
                    im["img_path"] = os.path.join(_img_dir, im["file_name"])
                anno_list += [anno]
        elif isinstance(img_dir, str):
            for _anno_path in anno_path:
                logger.info(f"loading {_anno_path}")
                anno = json.load(open(_anno_path))
                for im in anno["images"]:
                    im["img_path"] = os.path.join(img_dir, im["file_name"])
                anno_list += [anno]
        else:
            raise NotImplementedError
        anno = anno_list[0]
        for _anno in anno_list[1:]:
            anno["images"] += _anno["images"]
            anno["annotations"] += _anno["annotations"]

        self._images = anno["images"]
        annos_by_image = {im["id"]: [] for im in self._images}
        for ann in anno["annotations"]:
            img_id = ann["image_id"]
            annos_by_image[img_id] += [ann]
        self._annos_by_image = annos_by_image

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_info = self._images[index]
        img_id = img_info["id"]
        img_path = img_info["img_path"]
        img = cv2.imread(img_path)
        calib = np.array(img_info["calib"], dtype=np.float32)
        dist_coeffs = img_info.get("distCoeffs", [0] * self.num_dist)
        dist_coeffs = dist_coeffs[: self.num_dist]
        anns = self._annos_by_image[img_id]

        data_dict = {
            "image_name": img_info["file_name"],
            "image_height": img.shape[0],
            "image_width": img.shape[1],
            "img": img,
            "imgs": [img],
            "color_space": "bgr",
            "annotations": anns,
            "calibration": calib,
            "dist_coeffs": dist_coeffs,
            "image_id": str(img_id),
            "ignore_mask": img_info["ignore_mask"],
            "view": self.view,
        }

        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict


@OBJECT_REGISTRY.register
class Auto3dFromImage(Auto2dFromImage):  # noqa: D205,D400
    """Auto3d from image.

    Args:
        calibration (np.ndarray): With shape (3,4), for example, \
            calibration = ndarray([[1548, 0,    963, 0], \
                                   [0,    1548, 577, 0], \
                                   [0,    0,    1,   0]])
        dist_coeffs (list[int]): For example, the length is 4, \
            dist_coeffs = [-0.3477686047554016,   0.13457843661308289, \
                           0.0004990333109162748, 8.008156873984262e-05]
    """

    def __init__(
        self,
        data_path,
        calibration,
        dist_coeffs,
        transforms=None,
        to_rgb=False,
        return_src_img=False,
        view=None,
    ):
        super(Auto3dFromImage, self).__init__(
            data_path, transforms, to_rgb, return_src_img
        )
        self.calibration = calibration
        self.dist_coeffs = dist_coeffs
        self.view = view

    def __getitem__(self, item):
        data = super(Auto3dFromImage, self).__getitem__(item)
        # add calibration and dist_coeffs to data
        data["calibration"] = self.calibration
        data["dist_coeffs"] = self.dist_coeffs
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        return repr_str


class Real3DRecReader:
    def __init__(self, rec_file, idx_file=None, decode_image=True):
        """Real3d Rec Reader.

        rec_file: the path of rec
        decode_image: whether to return the decoded image.
        """
        assert (
            RecordUnit is not None
        ), "Discard"
        assert rec_file.endswith(".rec")
        if idx_file is None:
            idx_file = rec_file + ".idx"
        self.rec = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, "r")
        self.img_decoder = ImageDecoder(use_turbo_jpeg=False)
        self._len = len(open(idx_file, "r").readlines())
        self.decode_image = decode_image

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        item = self.rec.read_idx(idx)
        _, s = mx.recordio.unpack(item)
        rec_data = RecordUnit()
        rec_data.ParseFromString(s)
        rec_data = rec_data.body
        assert len(rec_data.data) == 2
        assert len(rec_data.extra) == 0
        image_buf = rec_data.data[0].value
        label_buf = rec_data.data[1].value
        label = json.loads(label_buf)
        if self.decode_image:
            image = self.img_decoder(image_buf)
            return image, label
        return image_buf, label

    def __del__(self):
        self.rec.close()


@OBJECT_REGISTRY.register
class Real3DDatasetRec(data.Dataset):
    """Real3D dataset class in rec fashion.

    Args:
        paths (list): Paths for rec path.
        num_classes (int): Number of classes.
        transforms (Transform): Transforms that applies to the data.
        num_dist (int): Length of dist coeffs.
        view (str): Camera view.
        track_params List[int]: [Mode, pre_index, cur_index, range].
            if you select Mode is Front_back, you can set the range of the
            data fragments, and which index you want. for example, my data have
            3 continuous frame and I want to use first and third, you can set
            track_params=[2, 0, 2, 3]
        select_sample (bool): Whether reselect index when invalid data.
    """

    def __init__(
        self,
        paths: List[str],
        num_classes: int,
        num_dist: int,
        transforms: Optional[Callable] = None,
        view: Optional[str] = None,
        track_params: List[int] = None,
        select_sample: Optional[bool] = False,
    ):
        super(Real3DDatasetRec, self).__init__()
        self.paths = paths
        self.num_classes = num_classes
        self.transforms = transforms
        self.num_dist = num_dist
        self.view = view
        if track_params is None:
            self.track_params = [0, 0, 0, 1]
        else:
            self.track_params = track_params
        assert (
            self.num_classes == 3 or self.num_classes == 6
        ), "currently the number of classes must be 3 or 6"

        self._load_recs(self.paths)
        self.num_samples = self.acc_lengths[-1]
        self.indices = list(range(self.num_samples))
        self.select_sample = select_sample

    def _load_recs(self, paths):
        assert isinstance(paths, list)
        self.recs = []
        for rec_path in paths:
            logger.info(f"loading {rec_path}")
            rec_reader = Real3DRecReader(rec_path)
            self.recs.append(rec_reader)
        lengths = [len(rec) for rec in self.recs]
        self.acc_lengths = np.cumsum(lengths)

    def __len__(self):
        return self.num_samples

    def _get_rec_from_idx(self, idx):
        """Get recio from index."""
        for length_idx in range(len(self.acc_lengths)):
            length_i = self.acc_lengths[length_idx]
            if idx > length_i:
                continue
            rec = self.recs[length_idx]
            if length_idx == 0:
                previous_idx = 0
            else:
                previous_idx = self.acc_lengths[length_idx - 1]
            idx_in_rec = idx - previous_idx
            assert idx_in_rec >= 0
            if idx_in_rec >= len(rec):
                idx_in_rec = len(rec) - 1
            return rec, idx_in_rec

    def _prepare_data(self, index):
        rec, idx = self._get_rec_from_idx(index)
        img, label = rec[idx]
        img_info = label["meta"]
        img_id = img_info["id"]
        calib = np.array(img_info["calib"], dtype=np.float32)
        dist_coeffs = img_info.get("distCoeffs", [0] * self.num_dist)
        dist_coeffs = np.array(dist_coeffs[: self.num_dist], dtype=np.float32)
        Tr_vel2cam = np.array(
            img_info.get("Tr_vel2cam", np.zeros((4, 4))), dtype=np.float32
        )
        if "file_name" in img_info:
            image_name = img_info["file_name"]
        elif "image_key" in img_info:
            image_name = img_info["image_key"] + ".jpg"

        anns = label["objects"]
        data_dict = {
            "image_name": image_name,
            "image_height": img.shape[0],
            "image_width": img.shape[1],
            "img": img,
            "imgs": [img],
            "color_space": "bgr",
            "annotations": anns,
            "calibration": calib,
            "dist_coeffs": dist_coeffs,
            "Tr_vel2cam": Tr_vel2cam,
            "image_id": str(img_id),
            "ignore_mask": img_info["ignore_mask"],
            "view": self.view,
            "index": index,
        }
        return data_dict

    def _cell(self, index):
        if self.track_params[0]:
            index -= index % self.track_params[3]
            index_pre = index + self.track_params[1]
            index += self.track_params[2]

        data_dict = self._prepare_data(index)
        if self.track_params[0]:
            rec, idx_pre = self._get_rec_from_idx(index_pre)
            img_pre, label_pre = rec[idx_pre]
            data_dict.update(
                {
                    "imgs": [img_pre, data_dict["img"]],
                    "track_mode": self.track_params[0],
                }
            )

        if self.track_params[0] == 2:
            anns = [label_pre["objects"], data_dict["annotations"]]
            data_dict.update({"annotations": anns})

        data_dict.update({"valid": True})
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __getitem__(self, index):
        if self.select_sample:
            flag = False
            while not flag:
                data_dict = self._cell(index)
                flag = data_dict["valid"]
                if not flag:
                    index = self.indices[np.random.randint(self.num_samples)]
        else:
            data_dict = self._cell(index)

        return data_dict
