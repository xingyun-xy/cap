# Copyright (c) Changan Auto. All rights reserved.

import os
import pickle
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
except ImportError:
    label_pb2 = None
    metrics_pb2 = None

from cap.registry import OBJECT_REGISTRY

__all__ = ["WaymoDataset"]

# ignore sign class
LABEL_TO_TYPE = {0: 1, 1: 2, 2: 4}


@OBJECT_REGISTRY.register_module
class WaymoDataset(Dataset):
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
        self,
        info_path: str,
        root_path: str,
        transforms: Optional[Compose] = None,
        class_names: List[str] = None,
        test_mode: bool = False,
        nsweeps: int = 1,
        load_interval: int = 1,
    ) -> None:
        """Waymo Open Dataset.

        Evaluation of this dataset cannot be performed online. It has to be
        performed using a compiled evaluation tool.
        To do so, you need to predict over the validation set, dump the
        prediction results to disk, and use waymo-open-dataset package to
        generate .bin file which is interpretable by the package.

        More details at https://waymo.com/open/.

        NOTE: this class implementation is for Waymo lidar data only. Camera
        data can be integrated in the future.

        Args:
            info_path (str): path to annotation file.
            root_path (str): root path ot data.
            transforms (Compose, optional): A transform Compose object.
                Defaults to None.
            class_names (List[str], optional): A list of category names.
                Defaults to None.
            test_mode (bool, optional): whether in test. Defaults to False.
            nsweeps (int, optional): number of sweeps of lidar points.
                Defaults to 1.
            load_interval (int, optional): jump this many samples when
                loading a sample. For example, 5 means load 1 sample
                every 5 samples. Defaults to 1.
        """
        super(WaymoDataset, self).__init__()
        self._info_path = info_path
        self._root_path = root_path
        self._class_names = class_names
        self._test_mode = test_mode
        self._nsweeps = nsweeps
        self._sample_interval = load_interval
        self._waymo_infos = None

        self._num_point_features = self.NumPointFeatures
        if self._nsweeps != 1:
            self._num_point_features += 1  # add a timestamp dimension

        self.transforms = transforms

        self.load_infos()

    def load_infos(self) -> None:
        """Load entire Waymo Open Dataset information into memory.

        Note that this only loads info about each frame, not the actual frames.
        """

        with open(self._info_path, "rb") as f:
            waymo_infos_all = pickle.load(f)
        self._waymo_infos = waymo_infos_all[:: self.load_interval]

    @staticmethod
    def load_waymo_single_frame(frame_path: str) -> np.ndarray:
        """Load a single lidar frame into memory.

        Args:
            frame_path (str): path to the point cloud file.

        Returns:
            [np.ndarray]: an N x 5 array containing all points.
        """
        with open(frame_path, "rb") as f:
            frame_obj = pickle.load(f)
        points_xyz = frame_obj["lidars"]["points_xyz"]
        points_feature = frame_obj["lidars"]["points_feature"]
        points_feature[:, 0] = np.tanh(points_feature[:, 0])
        points = np.concatenate([points_xyz, points_feature], axis=-1)
        return points

    def load_waymo_single_sweep(self, sweep: Dict[str, Any]) -> np.ndarray:
        """Load a single lidar sweep into memory.

        To be used when nsweeps > 1.

        Args:
            sweep (Dict[str, Any]): a dictionary describing the sweep.

        Returns:
            [np.ndarray]: an N x 5 array containing all points of that sweep.
        """
        with open(os.path.join(self.root_path, sweep["path"]), "rb") as f:
            sweep_obj = pickle.load(f)
        points_xyz = sweep_obj["lidars"]["points_xyz"]
        points_feature = sweep_obj["lidars"]["points_feature"]
        points_feature[:, 0] = np.tanh(points_feature[:, 0])
        points_sweep = np.concatenate(
            [points_xyz, points_feature], axis=-1
        ).T  # shape 5 x N
        num_points = points_sweep.shape[1]
        if sweep["transform_matrix"] is not None:
            points_sweep[:3, :] = sweep["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points)))
            )[:3, :]
        curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, curr_times.T

    def __len__(self) -> int:
        if not hasattr(self, "_waymo_infos"):
            self.load_infos()

        return len(self._waymo_infos)

    def __getitem__(self, index) -> dict:
        sample_info = self._waymo_infos[index]
        # prepare a container to load everything
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps,
            },
            "metadata": {
                "image_prefix": self.root_path,
                "num_point_features": self.num_point_features,
                "token": sample_info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }
        # load point cloud from file
        path = sample_info["path"]
        points = self.load_waymo_single_frame(
            os.path.join(self.root_path, path)
        )
        if self.nsweeps > 1:
            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]
            assert (self.nsweeps - 1) == len(
                sample_info["sweeps"]
            ), "nsweeps {} should be equal to the list length {}".format(
                self.nsweeps, len(sample_info["sweeps"])
            )
            for i in range(self.nsweeps - 1):
                sweep = sample_info["sweeps"][i]
                points_sweep, times_sweep = self.load_waymo_single_sweep(sweep)
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(
                points.dtype
            )

            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        res["lidar"]["points"] = points

        # load point cloud annotations from file
        assert "gt_boxes" in sample_info, "'gt_boxes' key not found in info"
        res["lidar"]["annotations"] = {
            "boxes": sample_info["gt_boxes"].astype(np.float32),
            "names": sample_info["gt_names"],
        }

        if self.transforms is not None:
            data, _ = self.transforms(res, sample_info)
            return data
        else:
            return res

    def evaluation(
        self, detections, output_dir: str = None, write_file: bool = True
    ):
        """Produce offline evaluation results."""
        infos = self._waymo_infos
        infos = _reorganize_info(infos)
        _create_pd_detection(
            detections,
            infos,
            self.root_path,
            output_dir,
            write_file=write_file,
        )

    @property
    def root_path(self) -> str:
        return self._root_path

    @property
    def test_mode(self) -> bool:
        return self._test_mode

    @property
    def nsweeps(self) -> int:
        return self._nsweeps

    @property
    def load_interval(self) -> int:
        return self._sample_interval

    @property
    def num_point_features(self) -> int:
        return self._num_point_features


# ===================================================================
# The following functions and classes are used for waymo evaluation.
# ===================================================================


class UUIDGeneration(object):
    """Generate uuid for detected object."""

    def __init__(self):
        self.mapping = {}

    def get_uuid(self, seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex
        return self.mapping[seed]


def _get_obj(path: str):
    """Load a single object (frame).

    Args:
        path (str): path to the pickle file.

    Returns:
        A dictionary contain object info.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def _reorganize_info(infos: List[Dict[str, Any]]):
    """Represent frames using {token: info} pair.

    Args:
        infos (List[Dict[str, Any]]): A list of frame infos, such as objects,
            timestamps, etc.

    Returns:
        A dictionary of {token: info}.
    """
    new_info = {}

    for info in infos:
        token = info["token"]
        new_info[token] = info

    return new_info


def _create_pd_detection(
    detections: Dict[str, Any],
    infos: Dict[str, Any],
    root_path: str,
    result_path: str,
    simple_infer: bool = False,
    threshold: float = 0.15,
    tracking: bool = False,
    write_file: bool = True,
):
    """Create Waymo standard detection format results.

    Args:
        detections (Dict[str, Any]): a dictionary of detection results.
        infos (Dict[str, Any]): a dictionary of frame infos.
        root_path (str): root path where data info is stored.
        result_path (str): path to write file to.
        simple_infer (bool, optional): whether the results are in simple
            formats. Determines how detections will be interpreted.
            Defaults to False.
        threshold (float, optional): Detection threshold. Defaults to 0.15.
        tracking (bool, optional): Whether this is tracking results.
            Defaults to False.
        write_file (bool): whether to write final detections to disk. False is
            used only in unit tests environment so no files generated but the
            whole process runs at least once.
            Defaults to True.
    """
    if label_pb2 is None or metrics_pb2 is None:
        raise ModuleNotFoundError(
            "Please run 'pip3 install "
            "waymo-open-dataset-tf-2-4-0 --user' to "
            "install waymo open dataset."
        )

    uuid_gen = UUIDGeneration()
    objects = metrics_pb2.Objects()

    for token, detection in tqdm(detections.items()):
        info = infos[token]
        obj = _get_obj(os.path.join(root_path, info["anno_path"]))

        if simple_infer:
            box3d = detection["boxes"]
            scores = detection["scores"]
            labels = detection["classes"]
        else:
            box3d = detection["box3d_lidar"].detach().cpu().numpy()
            scores = detection["scores"].detach().cpu().numpy()
            labels = detection["label_preds"].detach().cpu().numpy()

        mask = scores > threshold

        scores = scores[mask]
        labels = labels[mask]
        box3d = box3d[mask]

        # transform back to Waymo coordinate
        # x,y,z,w,l,h,r2
        # x,y,z,l,w,h,r1
        # r2 = -pi/2 - r1
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]

        if tracking:
            tracking_ids = detection["tracking_ids"]

        for i in range(box3d.shape[0]):
            det = box3d[i]
            score = scores[i]

            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = obj["scene_name"]
            o.frame_timestamp_micros = int(obj["frame_name"].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = LABEL_TO_TYPE[label]

            if tracking:
                o.object.id = uuid_gen.get_uuid(int(tracking_ids[i]))

            objects.objects.append(o)

    # Write objects to a file.
    if tracking:
        path = os.path.join(result_path, "tracking_pred.bin")
    else:
        path = os.path.join(result_path, "detection_pred.bin")

    if write_file:
        print("results saved to {}".format(path))
        f = open(path, "wb")
        f.write(objects.SerializeToString())
        f.close()
