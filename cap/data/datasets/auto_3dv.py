# Copyright (c) Changan Auto. All rights reserved.

import json
import logging
import os
from typing import Mapping, Optional, Sequence, Union

import msgpack
import numpy as np
import timeout_decorator
import torch.utils.data as data
from PIL import Image

from cap.data.datasets.bev import HomoGenerator
from cap.data.utils import pil_loader
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from cap.utils.filesystem import join_path
from .data_packer import Packer
from .pack_type.lmdb import Lmdb
from .pack_type.mxrecord import MXRecord
from .real3d_dataset import Real3DRecReader

try:
    from changan_driving_dataset import PoseTransformer
except ImportError:
    PoseTransformer = None

__all__ = [
    "Auto3DV",
    "Bev3DDatasetRec",
]

logger = logging.getLogger(__name__)


# class NamedIndexDatasetV2(NamedIndexDataset):
#     """
#     Wrapping over NamedIndexDataset.

#     support read index from lmdb to avoid excessive memory consumption.

#     """

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.lst_lmdb = {}
#         self.init_lst_lmdb()

#     def init_lst_lmdb(self):
#         count_rec = 0
#         for rec_path in self.imgrec_path_list:
#             lst_lmdb_path = rec_path.replace(".rec", ".lst.lmdb")
#             if os.path.isdir(lst_lmdb_path):
#                 # lst_lmdb_path is a directory means loading lmdb.
#                 self.lst_lmdb[count_rec] = Lmdb(
#                     lst_lmdb_path,
#                     False,
#                     True,
#                     readonly=True,
#                     map_size=1024 ** 2 * 10,
#                 )
#                 self.lst_lmdb[count_rec].open()

#             count_rec += 1

#     @property
#     def img_dict(self):
#         """Generate image name to rec index dictionary."""
#         if self._img_dict is None:
#             count_rec = 0
#             self._img_dict = {}
#             for rec_path in self.imgrec_path_list:
#                 lst_lmdb_path = rec_path.replace(".rec", ".lst.lmdb")
#                 if not os.path.isdir(lst_lmdb_path):
#                     # use lst file
#                     lst_path = rec_path.replace(".rec", ".lst")
#                     cur_img_list = _read_lst(lst_path)
#                     for k in cur_img_list:
#                         cur_img_list[k] = (count_rec, cur_img_list[k])
#                     self._img_dict.update(cur_img_list)
#                 count_rec += 1

#         return self._img_dict

#     def __getstate__(self):
#         state = self.__dict__
#         for _, lst_lmdb in state["lst_lmdb"].items():
#             lst_lmdb.close()
#         state["lst_lmdb"] = {}
#         return state

#     def __setstate__(self, state):
#         self.__dict__ = state
#         self.init_lst_lmdb()

#     def __getitem__(self, key):
#         if self.read_by_name:
#             assert isinstance(key, str)
#             if key in self.img_dict:
#                 idxs = self.img_dict[key]
#             else:
#                 for rec_idx, lst_lmdb in self.lst_lmdb.items():
#                     try:
#                         idxs = (rec_idx, int(lst_lmdb.read(key).decode()))
#                         break
#                     except Exception:
#                         continue
#         else:
#             idxs = key
#         return super(NamedIndexDataset, self).__getitem__(idxs)


class SyncInfo(object):
    """
    A SyncInfo object desribes all sample`s sync info of a dataset.

    It can extract sync infomation from a text file or a lmdb file.
    The format of one sample`s sync infomation is explain in __getitem__ func.

    """

    def __init__(self, sync_file=None, sync_file_lmdb=None, num_samples=None):

        assert (
            sync_file is not None or sync_file_lmdb is not None
        ), "please provede sync_file or sync_file_lmdb!"
        self.sync_file_lmdb = sync_file_lmdb
        if sync_file_lmdb is not None:
            self.num_samples = num_samples
            assert num_samples is not None, "please provide num_samples!"
            self.lmdb = Lmdb(
                self.sync_file_lmdb,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )
        else:
            self.items = self.readlines(sync_file)
            self.num_samples = len(self.items)

    def __getstate__(self):
        state = self.__dict__
        state["lmdb"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.sync_file_lmdb is not None:
            self.lmdb = Lmdb(
                self.sync_file_lmdb,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )

    def readlines(self, filename):
        """Read all the lines in a text file and return as a list."""
        with open(filename, "r") as f:
            lines = f.read().splitlines()
        return lines

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> dict:  # noqa: D205,D400
        """
        Return one sample contains multiple frame`s sync info as a list of
        dict organized like this:
        [
            # first frame
            {
                pack_dir: 'pack_dir_relative_to_dataset_root',
                lidar_top: 'xxx',
                camera_front: 'xxx',
                camera_front_left: 'xxx',
                camera_front_right: 'xxx',
                camera_rear: 'xxx',
                camera_rear_left: 'xxx',
                camera_rear_right: 'xxx',
                bev_seg: 'bev_seg/xxx',
                bev_3d: 'bev_3d/xxx',
                bev_motion_flow: 'bev_motion_flow/xxx',
                ......
            },

            # second frame
            {
                pack_dir: 'pack_dir_relative_to_dataset_root',
                lidar_top: 'xxx',
                camera_front: 'xxx',
                camera_front_left: 'xxx',
                camera_front_right: 'xxx',
                camera_rear: 'xxx',
                camera_rear_left: 'xxx',
                camera_rear_right: 'xxx',
                bev_seg: 'bev_seg/xxx',
                bev_3d: 'bev_3d/xxx',
                bev_motion_flow: 'bev_motion_flow/xxx',
                ......
            },
            ......
        ]
        """
        if self.sync_file_lmdb is not None:
            one_item = self.lmdb.read(index).decode()
        else:
            one_item = self.items[index]

        cur_sample = json.loads(one_item)
        """ cur_sample`s format is like this:
        {
            pack_dir: 'xxx',
            camera_front: [xxx,xxx,...]
            camera_front_left: [xxx,xxx,...]
            ...
        }
        """

        sync_infos = []
        pack_dir = cur_sample.pop("pack_dir")
        if "root" in cur_sample:
            cur_sample.pop("root")
        key = list(cur_sample.keys())[0]
        for i in range(len(cur_sample[key])):
            sync_info = {"pack_dir": pack_dir}
            for key in cur_sample:
                sync_info[key] = cur_sample[key][i]
            sync_infos.append(sync_info)
        return sync_infos


class Reader3DV(object):
    """
    Read img/depth/seg/bev_seg... data from origin file or rec.

    Args:
        root (str): dataset root path.
    """

    def __init__(
        self,
        root: str,
        img_rec_path: Optional[Union[str, Sequence[str]]] = None,
        seg_rec_path: Optional[Union[str, Sequence[str]]] = None,
        depth_rec_path: Optional[Union[str, Sequence[str]]] = None,
        bev_seg_rec_path: Optional[Union[str, Sequence[str]]] = None,
        bev_occlusion_rec_path: Optional[Union[str, Sequence[str]]] = None,
        bev3d_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        bev_static_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        bev_motion_flow_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        bev_discrete_obj_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        om_rec_path: Optional[Union[str, Sequence[str]]] = None,
        om_target_categorys: Optional[Union[str, Sequence]] = None,
        om_rec_type: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        self.root = root
        self.img_rec = self._load_rec(img_rec_path, read_by_name=True)
        self.seg_rec = self._load_rec(
            seg_rec_path, read_by_name=True, cv_format=-1, rgb=False
        )
        self.bev_seg_rec = self._load_rec(
            bev_seg_rec_path, read_by_name=True, cv_format=-1, rgb=False
        )
        self.bev_occlusion_rec = self._load_rec(
            bev_occlusion_rec_path, read_by_name=True, cv_format=-1, rgb=False
        )
        self.depth_rec = self._load_rec(
            depth_rec_path, read_by_name=True, cv_format=-1, rgb=False
        )
        self.pack_front_pose = {}

        self.bev3d_lmdb_path = bev3d_lmdb_path
        if bev3d_lmdb_path is not None:
            self.bev_3d_lmdb = Lmdb(
                bev3d_lmdb_path,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )
        self.bev_static_lmdb_path = bev_static_lmdb_path
        if bev_static_lmdb_path is not None:
            self.bev_static_lmdb = Lmdb(
                bev_static_lmdb_path,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )
        self.bev_motion_flow_lmdb_path = bev_motion_flow_lmdb_path
        if bev_motion_flow_lmdb_path is not None:
            self.bev_motion_flow_lmdb = Lmdb(
                bev_motion_flow_lmdb_path, False, True, readonly=True
            )

        self.bev_discrete_obj_lmdb_path = bev_discrete_obj_lmdb_path
        if bev_discrete_obj_lmdb_path is not None:
            self.bev_discrete_obj_lmdb = Lmdb(
                bev_discrete_obj_lmdb_path, False, True, readonly=True
            )

        self.om_lmdb = None
        self.om_lmdb_path = bev_static_lmdb_path
        if self.om_lmdb_path is not None:
            self.om_lmdb = Lmdb(
                self.om_lmdb_path,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )

        # use rec to read om gt if without lmdb
        self.om_rec = None
        if om_rec_path and self.om_lmdb is None:
            self.om_rec = MXRecord(
                om_rec_path,
                writable=False,
                key_type=str,
            )
        self.om_target_categorys = om_target_categorys
        self.om_rec_type = om_rec_type

    def _load_rec(self, rec_path, **kwargs):
        if rec_path is not None:
            return NamedIndexDatasetV2(_as_list(rec_path), **kwargs)
        return None

    def __getstate__(self):
        state = self.__dict__
        state["bev_3d_lmdb"] = None
        state["bev_static_lmdb"] = None
        state["bev_motion_flow_lmdb"] = None
        state["bev_discrete_obj_lmdb"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.bev3d_lmdb_path is not None:
            self.bev_3d_lmdb = Lmdb(
                self.bev3d_lmdb_path,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )
        if self.bev_static_lmdb_path is not None:
            self.bev_static_lmdb = Lmdb(
                self.bev_static_lmdb_path,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )
        if self.bev_motion_flow_lmdb_path is not None:
            self.bev_motion_flow_lmdb = Lmdb(
                self.bev_motion_flow_lmdb_path, False, True, readonly=True
            )
        if self.bev_discrete_obj_lmdb_path is not None:
            self.bev_discrete_obj_lmdb = Lmdb(
                self.bev_discrete_obj_lmdb_path,
                False,
                True,
                readonly=True,
            )
        if self.om_lmdb_path is not None:
            self.om_lmdb = Lmdb(
                self.om_lmdb_path,
                False,
                True,
                readonly=True,
                map_size=1024 ** 2 * 10,
            )

    def get_img_data(self, data_path, mode="RGB", size=None, rec_dataset=None):
        if data_path is None:
            return None
        if rec_dataset:
            img_np = rec_dataset[data_path]
            img = Image.fromarray(img_np).convert(mode)
        else:
            img = pil_loader(
                os.path.join(self.root, data_path), mode, size=size
            )
        return img

    def read_rgb(self, data_path, img_load_size):
        return self.get_img_data(
            data_path, "RGB", size=img_load_size, rec_dataset=self.img_rec
        )

    def read_depth(self, data_path):
        return self.get_img_data(data_path, "I", rec_dataset=self.depth_rec)

    def read_seg(self, data_path):
        return self.get_img_data(data_path, "I", rec_dataset=self.seg_rec)

    def read_bev_seg(self, data_path):
        return self.get_img_data(data_path, "I", rec_dataset=self.bev_seg_rec)

    def read_bev_occlusion(self, data_path):
        return self.get_img_data(
            data_path, "I", rec_dataset=self.bev_occlusion_rec
        )

    def read_bev_motion_flow(self, data_key, data_path):
        if self.bev_motion_flow_lmdb is not None:
            data = self.bev_motion_flow_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            annos_json = os.path.join(self.root, data_path)
            annos = json.load(open(annos_json, "rb"))
            return annos[data_key]

    def read_bev_3d(self, data_key, data_path):
        if self.bev3d_lmdb_path is not None:
            data = self.bev_3d_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            annos_json = os.path.join(self.root, data_path)
            annos = json.load(open(annos_json, "rb"))
            return annos[data_key]

    def read_static_anno(self, data_key, data_path):
        if self.bev_static_lmdb_path is not None:
            data = self.bev_static_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            annos_json = os.path.join(self.root, data_path)
            annos = json.load(open(annos_json, "rb"))
            return annos[data_key]

    def read_bev_discrete_obj(self, data_key):
        data = self.bev_discrete_obj_lmdb.read(data_key).decode()
        return json.loads(data)

    # TODO(): move transform code below to OnlineMappingTargetGenerator  # noqa: E501
    def read_om(self, om_path):
        @timeout_decorator.timeout(600)
        def load_om_from_txt():
            data = get_navnet(
                om_path,
                target_categorys=self.om_target_categorys,
                region=(72.4, -30, 51.2, -51.2),
            )
            return data

        def load_om_from_rec():
            timestamp = os.path.split(om_path)[-1]
            raw_data = self.om_rec.read(timestamp)
            raw_data = msgpack.unpackb(raw_data, raw=True)
            target = {}
            rec_type = self.om_rec_type
            for k, v in raw_data.items():
                if isinstance(v, list):
                    rec_type = "origin"
                    k = k.decode("utf-8")
                    if k in self.om_target_categorys:
                        v = [
                            np.frombuffer(iii, dtype=np.float).reshape(-1, 6)
                            for iii in v
                        ]
                        target[k] = v
                else:
                    v = np.frombuffer(v, dtype=np.float)
                    target[k.decode("utf-8")] = v.reshape(64, 64)
            if rec_type == "target":
                return {"om_target": target}
            else:
                return target

        def load_om_from_lmdb():
            timestamp = os.path.split(om_path)[-1]
            target = {}
            try:
                label = self.om_lmdb.read(timestamp).decode()
                label_dict = json.loads(label)
                for k, v in label_dict.items():
                    if len(v[0]) == 0:
                        continue
                    res = []
                    for ins in label_dict[k]:
                        if not ins:
                            continue
                        ins = np.array(ins, np.float)
                        if ins.shape[1] == 6:
                            res.append(ins)
                        else:
                            trans_ins = np.zeros((ins.shape[0], 6), np.float)
                            trans_ins[:, :2] = ins[:, :2]
                            if k != "crosspoints":
                                trans_ins[:, 3:5] = ins[:, 2:]
                            res.append(trans_ins)
                    target[k] = res
            except Exception:
                raise Exception(f"Error timestamp: {timestamp}")
            return target

        if self.om_lmdb:
            return load_om_from_lmdb()
        elif self.om_rec:
            return load_om_from_rec()

        try:
            return load_om_from_txt()
        except (timeout_decorator.TimeoutError, FileNotFoundError) as e:
            if isinstance(e, timeout_decorator.TimeoutError):
                raise FileNotFoundError(f"read {om_path} timeout > 600 sec")
            elif isinstance(e, FileNotFoundError):
                raise FileNotFoundError(f"{om_path} FileNotFoundError")

    def load_front_pose(self, pack_dir, timestamp):
        if pack_dir not in self.pack_front_pose:
            front_pose_path = os.path.join(
                self.root, pack_dir, "odometry/front_camera_loam.txt"
            )
            front_pose = np.loadtxt(front_pose_path)

            pt = PoseTransformer()
            pt.loadarray(front_pose)
            timestamps = pt.get_timestamps()

            transform = pt.as_transform(absolute=True)

            self.pack_front_pose[pack_dir] = {
                "timestamp": timestamps.reshape(-1),
                "transform": transform,
            }

        idx = np.searchsorted(
            self.pack_front_pose[pack_dir]["timestamp"], timestamp, side="left"
        )

        return self.pack_front_pose[pack_dir]["transform"][idx]


class Frame(object):  # noqa: D205,D400
    """
    A Frame object could load many kinds type of img data using
    giving sync information, include img, depth, segmentation,
    bev_seg and so on.

    # TODO(): fix gt_online_mapping_dir in om. #
    """

    def __init__(
        self,
        reader,
        frame_sync_info: Mapping,
        camera_view_names: Sequence,
        img_load_size: Sequence,
        gt_online_mapping_dir: str = None,
    ) -> None:

        self.reader = reader
        self.frame_sync_info = frame_sync_info
        self.camera_view_names = camera_view_names
        self.img_load_size = img_load_size
        self.gt_online_mapping_dir = gt_online_mapping_dir

    def img(self):
        """Return multi-view img as a list."""
        img_paths = join_path(
            self.frame_sync_info["pack_dir"],
            [
                os.path.join(view_name, self.frame_sync_info[view_name])
                + ".jpg"
                for view_name in self.camera_view_names
            ],
        )
        return [
            self.reader.read_rgb(img_path, img_load_size)
            for img_path, img_load_size in zip(img_paths, self.img_load_size)
        ]

    def depth(self):
        """Return multi-view depth as a list."""
        img_paths = join_path(
            self.frame_sync_info["pack_dir"],
            [
                os.path.join(
                    "depth_" + view_name,
                    self.frame_sync_info[view_name] + ".png",
                )
                for view_name in self.camera_view_names
            ],
        )
        return [self.reader.read_depth(img_path) for img_path in img_paths]

    def seg(self):
        """Return multi-view segmentation as a list."""
        img_paths = join_path(
            self.frame_sync_info["pack_dir"],
            [
                os.path.join(
                    "seg_" + view_name,
                    self.frame_sync_info[view_name] + ".png",
                )
                for view_name in self.camera_view_names
            ],
        )
        return [self.reader.read_seg(img_path) for img_path in img_paths]

    def bev_seg(self):
        """Return bev segmentation."""
        bev_seg_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["bev_seg"] + ".png",
        )
        return self.reader.read_bev_seg(bev_seg_path)

    def bev_occlusion(self):
        """Return bev occlusion."""
        bev_occlusion_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["occlusion"] + ".png",
        )
        return self.reader.read_bev_occlusion(bev_occlusion_path)

    def read_static_anno(self):
        """Return bev_static key to get anno from annos file and bev_static_json_path."""  # noqa: E501
        bev_seg_key = os.path.join(
            self.frame_sync_info["camera_front"],
        )
        bev_seg_json_path = os.path.join(
            self.frame_sync_info["pack_dir"], "bevseg/annotations.json"
        )
        return self.reader.read_static_anno(bev_seg_key, bev_seg_json_path)

    def front_seg(self):
        """Return segmentation for front view."""
        front_seg_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            os.path.join(
                "seg_camera_front",
                self.frame_sync_info["camera_front"] + ".png",
            ),
        )
        return self.reader.read_seg(front_seg_path)

    def timestamp(self):
        """Return timestamp."""
        # NOTE: The first name of camera_view_names must be "front" camera
        # e.g. "camera_front" or "fisheye_front"
        cam_front_name = self.camera_view_names[0]
        timestamp = (
            np.array([self.frame_sync_info[cam_front_name]], dtype="float64")
            / 1000
        )

        return timestamp

    def front_pose(self):  # noqa: D205,D400
        """
        Return front pose from current timestamp to first timestamp in
        corresponding pack and represented as transformation matrix.
        """
        return self.reader.load_front_pose(
            self.frame_sync_info["pack_dir"], self.timestamp()
        )

    def bev_3d(self):
        """Return bev_3d key to get anno from annos file and bev_3d_json_path."""  # noqa: E501
        bev_3d_key = os.path.join(
            self.frame_sync_info["pack_dir"], self.frame_sync_info["bev_3d"]
        )
        bev_3d_json_path = os.path.join(
            self.frame_sync_info["pack_dir"], "bev3d/annotations.json"
        )
        return self.reader.read_bev_3d(bev_3d_key, bev_3d_json_path)

    def bev_discrete_obj(self):
        """Return bev_discrete_obj key to get anno from annos file."""
        bev_discrete_obj_key = self.frame_sync_info["camera_front"]
        return self.reader.read_bev_discrete_obj(bev_discrete_obj_key)

    def om(self):
        """Return om gt."""
        om_path = os.path.join(
            self.reader.root,
            self.frame_sync_info["pack_dir"],
            self.gt_online_mapping_dir,
            self.frame_sync_info["bev_seg"],
        )
        return self.reader.read_om(om_path)

    def bev_motion_flow(self):
        """Return bev motion flow."""
        bev_motion_flow_key = os.path.join(
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["bev_motion_flow"],
        )
        bev_motion_flow_json_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            "bev_motion_flow/annotations.json",
        )
        return self.reader.read_bev_3d(
            bev_motion_flow_key, bev_motion_flow_json_path
        )

    @property
    def pack_dir(self):
        return join_path(self.reader.root, self.frame_sync_info["pack_dir"])

    @property
    def img_paths(self):
        img_paths = [
            os.path.join(view_name, self.frame_sync_info[view_name]) + ".jpg"
            for view_name in self.camera_view_names
        ]

        return img_paths


@OBJECT_REGISTRY.register
class Auto3DV(data.Dataset):  # noqa: D205,D400
    """A dataset for generating specific data to support many
    kinds of 3dv task. e.g. 2.5d task(depth,pose,resflow), bev_seg task,
    bev_3d task, bev_motion_flow task and so on.

    Args:
        root str: Path of dataset root.
        camera_view_names list(str): sub directory name of each view.
        per_view_shape dict: img shape of each view.
        img_load_size list(str): img load size of each view
        sync_file str: path of sync_file.
        sync_file_lmdb str: The path of LMDB sync file
        num_samples int: total frame num in dataset.
        img_rec_path (str or list(str)):
            image rec path, load image data from rec file if not None,
            will load from original img otherwise.
        seg_rec_path (str or list(str)):
            seg rec path, load seg/obj_maks/ data from rec file
            if not None, will load from original seg file otherwise.
        depth_rec_path (str or list(str)):
            depth rec path, load depth from rec file if not None,
            will load from original depth file otherwise.
        bev_seg_rec_path (str or list(str)):
            bev seg rec path, load bev_seg data from rec file if not None,
            will load from original bev_seg file otherwise.
        bev_occlusion_rec_path (str or list(str)):
            bev occlusion rec path, load bev_occlusion data from rec file
            if not None, will load from original bev_occlusion file otherwise.
        bev_3d_lmdb_path (str):
            bev3d annotation lmdb path.
        bev_static_lmdb_path (str):
            bev static annotation lmdb path.
        bev_motion_flow_lmdb_path (str):
            bev motion flow annotation lmdb path.
        front_mask_path str: path of front mask file.
        load_intrinsics bool: whether load intrinsics.
        load_distortcoef bool: whether load distortion coefficient.
        homo_gen dict: HomoGenerator dict for compute homography.
    """

    def __init__(
        self,
        root: str,
        camera_view_names: Sequence[str],
        per_view_shape: Mapping,
        sync_file: str,
        transforms=None,
        img_load_size: Optional[Sequence] = None,
        sync_file_lmdb: Optional[str] = None,
        num_samples: Optional[int] = None,
        img_rec_path: Optional[Union[str, Sequence]] = None,
        seg_rec_path: Optional[Union[str, Sequence]] = None,
        depth_rec_path: Optional[Union[str, Sequence]] = None,
        bev_seg_rec_path: Optional[Union[str, Sequence]] = None,
        bev_occlusion_rec_path: Optional[Union[str, Sequence]] = None,
        bev_3d_lmdb_path: Optional[Union[str, Sequence]] = None,
        bev_static_lmdb_path: Optional[Union[str, Sequence]] = None,
        bev_motion_flow_lmdb_path: Optional[Union[str, Sequence]] = None,
        bev_discrete_obj_lmdb_path: Optional[Union[str, Sequence]] = None,
        gt_online_mapping_dir: Optional[str] = None,
        om_target_categorys: Optional[Union[str, Sequence]] = None,
        om_rec_type: Optional[str] = None,
        front_mask_path: Optional[str] = None,
        load_intrinsics: bool = False,
        load_distortcoef: bool = False,
        homo_gen: Optional[dict] = None,
    ):
        self.root = root
        self.camera_view_names = camera_view_names
        self.per_view_shape = per_view_shape
        if img_load_size is not None:
            assert len(img_load_size) == len(camera_view_names)
        else:
            img_load_size = [None] * len(camera_view_names)
        self.img_load_size = img_load_size

        self.transforms = transforms

        self.load_intrinsics = load_intrinsics
        self.load_distortcoef = load_distortcoef
        self.intrinsics = None
        self.distcoef = None
        self.gt_online_mapping_dir = gt_online_mapping_dir

        # get homography matrix

        self.homo_mat = None
        self.homo_offset = None
        self.homo_gen = homo_gen
        if homo_gen is not None:
            homo_genor = HomoGenerator(**homo_gen)
            self.homo_mat = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()
        self.front_mask_img = None
        if front_mask_path is not None:
            assert os.path.exists(
                front_mask_path
            ), f"{front_mask_path} not exist!"
            self.front_mask_img = pil_loader(front_mask_path, "F")
        om_rec_path = None
        if (
            gt_online_mapping_dir
            and os.path.splitext(gt_online_mapping_dir)[1] == ".rec"
        ):
            om_rec_path = os.path.splitext(gt_online_mapping_dir)[0]

        self.reader = Reader3DV(
            root=root,
            img_rec_path=img_rec_path,
            seg_rec_path=seg_rec_path,
            depth_rec_path=depth_rec_path,
            bev_seg_rec_path=bev_seg_rec_path,
            bev_occlusion_rec_path=bev_occlusion_rec_path,
            bev3d_lmdb_path=bev_3d_lmdb_path,
            bev_static_lmdb_path=bev_static_lmdb_path,
            bev_motion_flow_lmdb_path=bev_motion_flow_lmdb_path,
            bev_discrete_obj_lmdb_path=bev_discrete_obj_lmdb_path,
            om_rec_path=om_rec_path,
            om_target_categorys=om_target_categorys,
            om_rec_type=om_rec_type,
        )

        self.sync_info = SyncInfo(sync_file, sync_file_lmdb, num_samples)

    def __getstate__(self):
        state = self.__dict__
        state["homo_mat"] = None
        state["homo_offset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.homo_gen is not None:
            homo_genor = HomoGenerator(**self.homo_gen)
            self.homo_mat = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()

    def _load_homo_offset_path(self, homo_offset_path):
        """Load _load_homo_offset_path for each view."""

        H = []
        for sub_dir in self.camera_view_names:
            hom_file = os.path.join(homo_offset_path, sub_dir + ".npy")
            assert os.path.exists(hom_file), hom_file
            ori_homo = np.load(hom_file).astype("float32")
            H.append(ori_homo)
        return np.stack(H, axis=0)

    def _load_attr(self, pack_dir):
        attribute_file = os.path.join(self.root, pack_dir, "attribute.json")

        @timeout_decorator.timeout(60)
        def load():
            att_file = json.load(open(attribute_file))
            return att_file

        try:
            return load()
        except (timeout_decorator.TimeoutError, FileNotFoundError) as e:
            if isinstance(e, timeout_decorator.TimeoutError):
                raise FileNotFoundError(
                    f"read {attribute_file} timeout > 60 sec"
                )
            elif isinstance(e, FileNotFoundError):
                raise FileNotFoundError(f"{attribute_file} FileNotFoundError")

    def _set_intrinsics(self, pack_dir):
        """Load intrinsics from attribute file."""
        att_file = self._load_attr(pack_dir)
        cam_front_intri = np.array(
            att_file["calibration"]["camera_front"]["K"], dtype=np.float32
        )
        cam_front_intri[0] /= self.per_view_shape["camera_front"][1]
        cam_front_intri[1] /= self.per_view_shape["camera_front"][0]
        self.intrinsics = cam_front_intri

    def _set_distcoefs(self, pack_dir):
        """Load distort coefficients from attribute file."""
        att_file = self._load_attr(pack_dir)
        cam_front_discoef = np.array(
            att_file["calibration"]["camera_front"]["d"], dtype=np.float32
        )
        self.distcoef = cam_front_discoef

    def _get_frames(self, sync_infos: Sequence):
        return [
            Frame(
                self.reader,
                one_frame,
                self.camera_view_names,
                self.img_load_size,
                self.gt_online_mapping_dir,
            )
            for one_frame in sync_infos
        ]

    def __getitem__(self, idx):
        multi_frame_sync_info = self.sync_info[idx]
        frames = self._get_frames(multi_frame_sync_info)

        data_dict = {}
        data_dict["frames"] = frames

        if self.homo_mat is not None:
            data_dict["homography"] = self.homo_mat.copy()

        if self.homo_offset is not None:
            data_dict["homo_offset"] = self.homo_offset.copy()

        if self.front_mask_img is not None:
            data_dict["front_mask"] = self.front_mask_img.copy()

        if self.load_intrinsics:
            if self.intrinsics is None:
                self._set_intrinsics(self.sync_info[0][0]["pack_dir"])
            data_dict["intrinsics"] = self.intrinsics.copy()

        if self.load_distortcoef:
            if self.distcoef is None:
                self._set_distcoefs(self.sync_info[0][0]["pack_dir"])
            data_dict["distortcoef"] = self.distcoef.copy()

        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __len__(self):
        return len(self.sync_info)


class FileList2LMDB(Packer):
    """
    FileList2LMDB is used for converting file list to to LMDB format.

    NOTE: Make sure target_data_dir is not a bucket path.
    Otherwise it will be very slow.

    Args:
        file_list_path (str): The dir of original imagenet data.
        target_data_dir (str): Path for LMDB file.
    """

    def __init__(self, file_list_path, target_data_dir):

        with open(file_list_path, "r") as f:
            self.lines = f.read().splitlines()
        nums = len(self.lines)
        assert nums > 0, f"line nums in {file_list_path} must >0"
        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": nums,
            "map_async": True,
        }
        super(FileList2LMDB, self).__init__(
            uri=target_data_dir,
            max_data_num=len(self.lines),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def pack_data(self, idx):
        return self.lines[idx].encode()


class RecLst2LMDB(Packer):
    """RecLst2LMDB is used for converting rec lst files to LMDB.

    NOTE: Make sure target_data_dir is not a bucket path or gpfs path.

    Args:
        lst_file_list (list): List contains json files for packing.
        target_data_dir (str): Path for LMDB file.
    """

    def __init__(
        self,
        lst_file_list: Sequence,
        target_data_dir: str,
    ):
        lst_file_list = _as_list(lst_file_list)
        nums = len(lst_file_list)
        assert nums > 0, "nums of lst file(s) must >0"
        all_lst_data = {}
        for lst_file in lst_file_list:
            all_lst_data.update(_read_lst(lst_file))

        self.all_sample = []
        for key, value in all_lst_data.items():
            cur_sample = {}
            cur_sample["key"] = key
            cur_sample["value"] = json.dumps(value)
            self.all_sample.append(cur_sample)

        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": nums,
            "map_async": True,
        }
        super(RecLst2LMDB, self).__init__(
            uri=target_data_dir,
            max_data_num=len(self.all_sample),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def _write(self, idx, data):
        idx = data["key"]
        data = data["value"].encode()
        return super()._write(idx, data)

    def pack_data(self, idx):
        return self.all_sample[idx]


class Json2LMDB(Packer):
    """Json2LMDB is used for converting json files to LMDB.

    The format of each json file should be like:{key_1: val_1,
    key_2: val_2 ...}, for different json file, all the keys must
    be unique.

    NOTE: Make sure target_data_dir is not a bucket path or gpfs path.

    Args:
        json_file_list (list): List contains json files for packing.
        target_data_dir (str): Path for LMDB file.
    """

    def __init__(
        self,
        json_file_list: Sequence,
        target_data_dir: str,
    ):
        json_file_list = _as_list(json_file_list)
        nums = len(json_file_list)
        assert nums > 0, "nums of json file(s) must >0"
        all_json_data = {}
        for json_file in json_file_list:
            json_data = json.load(open(json_file, "rb"))
            all_json_data.update(json_data)

        self.all_sample = []
        for key, value in all_json_data.items():
            cur_sample = {}
            cur_sample["key"] = key
            cur_sample["value"] = json.dumps(value)
            self.all_sample.append(cur_sample)

        lmdb_kwargs = {
            "map_size": 1099511627776 * 2,
            "meminit": nums,
            "map_async": True,
        }
        super(Json2LMDB, self).__init__(
            uri=target_data_dir,
            max_data_num=len(self.all_sample),
            pack_type="lmdb",
            num_workers=1,
            **lmdb_kwargs,
        )

    def _write(self, idx, data):
        idx = data["key"]
        data = data["value"].encode()
        return super()._write(idx, data)

    def pack_data(self, idx):
        return self.all_sample[idx]


def generate_sync_info(
    pack_list_path,
    dataset_root,
    sync_info_path,
    frames_offset=(0,),
    frame_interval_time=30,
    to_lmdb=False,
    lmdb_path=None,
    drop_frame_num=10,
    fisheye=False,
):
    """
    Generate sync information file used for Auto3DV class.

    As a Example code.

    Args:
        pack_list_path str: Path of pack list.
        dataset_root str: dataset root path.
        sync_info_path str: result path of sync_info.
        frames_offset list(int):
            interval between the reference frame and current frame.
            NOTE: first element in frames_offset must be 0.
            suppose an img sequence is like [0,1,2....T]
            Example 1:
                (0) means we only load one frame in a sample.
                the sample that dataset will generate is [t] at index t.
            Example 2:
                (0,-1) means we load two frames in a sample.
                the sample that dataset will generate is [t,t-1] at index t.
            Example 3:
                (0,-1,-3) means we load three frames in a sample.
                the sample that dataset will generate is [t,t-1,t-3] at index t.  # noqa
            Example 3:
                (0,-1,-4,-5) means we load four frames in a sample.
                the sample that dataset will generate is [t,t-1,t-4,t-5] at index t,  # noqa
        frame_interval_time int: The interval between adjecent
        to_lmdb bool: whehter convert sync_info file to lmdb format.
        lmdb_path str: sync_info lmdb path.
        drop_frame_num int: frame index in [0,drop_frame_num] and [-drop_frame_num,-1]  # noqa
            are invalid and will be droped.
        fisheye bool: whether generate the sync_info for fisheye dataset, default: False.
    """

    def _check_valid_frame_interval(
        sync_infos, frames_offset, frame_interval_time, fisheye
    ):

        for i in range(1, len(sync_infos)):
            interval_time = -frames_offset[i] * frame_interval_time
            min_time, max_time = interval_time - 10, interval_time + 10
            cam_front_name = "camera_front" if not fisheye else "fisheye_front"
            cur_interval_time = abs(
                int(sync_infos[i][cam_front_name])
                - int(sync_infos[0][cam_front_name])
            )
            if cur_interval_time >= max_time or cur_interval_time <= min_time:
                return False
        return True

    frames_offset = (
        frames_offset if frames_offset[0] == 0 else [0] + frames_offset
    )
    # make sure all value in frames_offset is <=0
    assert all(
        list(map(lambda x: x <= 0, frames_offset))
    ), "frame interval must be less than or equal to 0"

    with open(pack_list_path, "r") as f:
        pack_list = f.read().splitlines()

    sub_dirs = (
        [
            "camera_front",
            "camera_front_left",
            "camera_front_right",
            "camera_rear_left",
            "camera_rear_right",
            "camera_rear",
        ]
        if not fisheye
        else [
            "fisheye_front",
            "fisheye_rear",
            "fisheye_left",
            "fisheye_right",
        ]
    )
    all_samples = []
    for _pack_idx, pack_dir in enumerate(pack_list):
        pack_path = os.path.join(dataset_root, pack_dir)
        pack_json = os.path.join(pack_path, "attribute.json")
        with open(pack_json, "r", encoding="utf8") as fp:
            json_data = json.load(fp)
            sync = json_data["sync"]
            total_frame = len(sync["lidar_top"])

            start_idx = drop_frame_num - frames_offset[-1]
            end_idx = total_frame - drop_frame_num

            for i in range(start_idx, end_idx):
                cur_sample = {}
                cur_sample["pack_dir"] = pack_dir
                sync_infos = []
                for offset in frames_offset:
                    sync_info = {}
                    for sub_dir in sub_dirs:
                        sync_info[sub_dir] = os.path.join(
                            str(sync[sub_dir][i + offset])
                        )
                    sync_info["bev_seg"] = os.path.join(
                        "bevGT_3", sync_info["camera_front"]
                    )
                    sync_info["bev_3d"] = str(sync["lidar_top"][i + offset])

                    sync_info["bev_motion_flow"] = str(
                        sync["camera_front"][i + offset]
                    )
                    # fill any other content to sync_info,
                    # set None if not exist.
                    # e.g.
                    # sync_info['lidar_top']=xxx
                    # sync_info['bev_motion_flow']=xxx

                    sync_infos.append(sync_info)
                if _check_valid_frame_interval(
                    sync_infos, frames_offset, frame_interval_time, fisheye
                ):
                    for key in sync_infos[0]:
                        cur_sample[key] = [
                            sync_info[key] for sync_info in sync_infos
                        ]

                    # do other check to decide whether append current sample
                    # only check bev_seg of frame idx 0 here
                    if os.path.exists(
                        os.path.join(
                            pack_path, sync_infos[0]["bev_seg"] + ".png"
                        )
                    ):
                        all_samples.append(json.dumps(cur_sample) + "\n")

    all_sample_num = len(all_samples)
    if all_sample_num > 0:
        logger.info(f"all sample num: {all_sample_num}")
        with open(sync_info_path, "w") as w:
            for one in all_samples:
                w.write(one)
        if to_lmdb:
            assert lmdb_path is not None, "please provide valid lmdb path"
            f = FileList2LMDB(sync_info_path, lmdb_path)
            f()


@OBJECT_REGISTRY.register
class Auto3DVFromImage(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.

    Args:
        data_path (str): The path where the image is stored.
        att_json_path: path of attribute.json.
        transforms: List of transform.
        per_view_shape : img shape of each view.
        camera_view_names : sub directory name of each view.
        only_front_view : bool
            only load front view img. setting True for
            depth_pose_resflow training.
        get_pack_dir: bool, whether to get the pack_dir,
            now only for inference of bev task when need
            6v images.
        homo_gen dict: HomoGenerator dict for compute homography.


    """

    def __init__(
        self,
        data_path: str,
        intrinsics: np.ndarray,
        distortcoef: np.ndarray,
        per_view_shape: Mapping,
        camera_view_names: Optional[Sequence] = None,
        img_load_size: Optional[Sequence] = None,
        att_json_path: Optional[Sequence] = None,
        transforms: Optional[Sequence] = None,
        only_front_view: Optional[bool] = True,
        get_pack_dir: Optional[bool] = True,
        homo_gen: Optional[dict] = None,
    ):
        self.data_path = os.path.split(data_path)[
            0
        ]  # strip the `camera_view` level of the path hierarchy  # noqa
        self.intrinsics = intrinsics
        self.distortcoef = distortcoef
        self.transforms = transforms
        self.att_json_path = att_json_path
        self.img_load_size = img_load_size
        self.per_view_shape = per_view_shape
        self.camera_view_names = camera_view_names
        self.only_front_view = only_front_view
        self.get_pack_dir = get_pack_dir
        self.front_img_name = camera_view_names[
            0
        ]  # fisheye_front or camera_front
        assert self.front_img_name in ["fisheye_front", "camera_front"]
        self.collect_samples()

        # get homography and homo offset matrix
        self.H_origin = None
        self.homo_offset = None
        self.homo_gen = homo_gen
        if homo_gen is not None:
            homo_genor = HomoGenerator(**homo_gen)
            self.H_origin = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()
        self.pers_view_intr_norm_mat = [
            np.array(
                [
                    [1 / self.per_view_shape[camera_view_name][1], 0, 0],
                    [0, 1 / self.per_view_shape[camera_view_name][0], 0],
                    [0, 0, 1],
                ],
                dtype="float32",
            ).reshape(3, 3)
            for camera_view_name in self.camera_view_names
        ]

    def __getstate__(self):
        state = self.__dict__
        state["H_origin"] = None
        state["homo_offset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.homo_gen is not None:
            homo_genor = HomoGenerator(**self.homo_gen)
            self.H_origin = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()

    def load_homography(self):
        """Load homography for each view and retuan as list."""
        H = []
        for camera_view_name in self.camera_view_names:
            hom_file = os.path.join(self.homo_path, camera_view_name + ".npy")
            assert os.path.exists(hom_file), hom_file
            H.append(np.load(hom_file).astype("float32"))
        return H

    def _load_homo_offset_path(self, homo_offset_path):
        """Load homo_offset_path for each view."""
        H = []
        for sub_dir in self.camera_view_names:
            hom_file = os.path.join(homo_offset_path, sub_dir + ".npy")
            assert os.path.exists(hom_file), hom_file
            ori_homo = np.load(hom_file).astype("float32")
            H.append(ori_homo)
        return np.stack(H, axis=0)

    def collect_samples(self):
        """Create data samples from data_path."""

        postfix = ".jpg"
        if self.front_img_name == "camera_front":
            att_json_path = os.path.join(self.data_path, "attribute.json")
        else:
            att_json_path = os.path.join(self.data_path, "attribute_v1.json")
        assert os.path.exists(att_json_path)
        att_file = json.load(open(att_json_path))

        # whether to use 30fps
        if (
            "unsync" in att_file.keys()
            and self.front_img_name in att_file["unsync"].keys()
        ):
            imgs_list = att_file["unsync"]
        else:
            imgs_list = att_file["sync"]

        self.sample_lines = []
        sync_list = att_file["sync"]
        for i in range(len(sync_list[self.front_img_name])):
            file_list_before = []
            file_list_current = []
            for camera_view_name in self.camera_view_names:
                if (
                    sync_list[camera_view_name][i]
                    >= imgs_list[camera_view_name][1]
                ):
                    # get index in imgs_list according sync_list
                    _time_stamp = sync_list[camera_view_name][i]
                    unsync_view_index = imgs_list[camera_view_name].index(
                        _time_stamp
                    )
                    file_list_before.append(
                        str(imgs_list[camera_view_name][unsync_view_index - 1])
                        + postfix
                    )
                    file_list_current.append(
                        str(imgs_list[camera_view_name][unsync_view_index])
                        + postfix
                    )
            if len(file_list_current) == len(self.camera_view_names):
                if self.only_front_view:
                    # for 2.5d
                    self.front_idx = self.camera_view_names.index(
                        self.front_img_name
                    )
                    self.sample_lines.append(
                        [
                            file_list_current[self.front_idx],
                            file_list_before[self.front_idx],
                        ]
                    )
                else:
                    # for bev
                    self.sample_lines.append(
                        [file_list_current, file_list_before]
                    )

    def get_image_path(self, img_dir, name, postfix=".jpg"):
        image_path = os.path.join(self.data_path, img_dir, name + postfix)
        assert os.path.exists(image_path), image_path
        return image_path

    def get_image(self, img_dir, name, postfix, mode="RGB", size=None):
        image = pil_loader(
            self.get_image_path(img_dir, name, postfix), mode, size=size
        )
        return image

    def __len__(self):
        return len(self.sample_lines)

    @property
    def sample_list(self):
        return self.sample_lines

    def __getitem__(self, index):
        data_dict = {}
        img_names = self.sample_lines[index]

        data_dict["pil_imgs"] = []
        for img_names_i in img_names:
            imgs = []
            if self.only_front_view:  # only load front view img
                imgs.append(
                    self.get_image(
                        self.front_img_name,
                        img_names_i,
                        "",
                        "RGB",
                        size=self.img_load_size,
                    )
                )
            else:
                for camera_view_name, img_name in zip(
                    self.camera_view_names, img_names_i
                ):
                    imgs.append(
                        self.get_image(
                            camera_view_name,
                            img_name,
                            "",
                            "RGB",
                            size=self.img_load_size,
                        )
                    )
            data_dict["pil_imgs"].append(imgs)

        # homograpy
        H = []
        if self.H_origin is not None:
            for camera_view_name, _img_name in zip(
                self.camera_view_names, img_names[-1]
            ):
                i = self.camera_view_names.index(camera_view_name)
                currennt_H = self.H_origin[i]
                currennt_H = self.pers_view_intr_norm_mat[i] @ currennt_H
                H.append(currennt_H)
            data_dict["homography"] = np.stack(H, axis=0)
        if self.homo_offset is not None:
            data_dict["homo_offset"] = self.homo_offset.copy()
        if self.only_front_view:
            data_dict["obj_mask"] = self.get_image(
                f"seg_{self.front_img_name}",
                img_names[1].replace("jpg", "png"),
                "",
                "F",
            )

        # To compatible with 2d task, set t-1 as img_name frame.
        # data_dict['img_name'] = _as_list(img_names[-1])[0]
        data_dict["img_name"] = _as_list(img_names[0])[0]
        cam_front_intri = self.intrinsics.copy()[:3, :3]
        cam_front_intri[0] /= self.per_view_shape[self.front_img_name][1]
        cam_front_intri[1] /= self.per_view_shape[self.front_img_name][0]
        data_dict["intrinsics"] = cam_front_intri
        data_dict["distortcoef"] = self.distortcoef.copy()
        if self.get_pack_dir:
            data_dict["pack_dir"] = self.data_path

        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        return repr_str


@OBJECT_REGISTRY.register
class Bev3DDatasetRec(data.Dataset):
    """The Bev3D dataset only for parsing real3d recs and \
        for bev3d training(validation).

    Args:
        img_rec_path (dict): Paths for recs.
        homo_path (str): Path of homograph matrix npy file.
        num_classes (int): Number of classes.
        transforms (Transform): Transforms that applies to the data.
        views (int): View number of dataset.
        per_view_shape (dict): Img shape of each view.
        sub_dirs (Sequence[str]): Sub directory name of each view.
        homo_gen (object): HomoGenerator object for compute homography.
    """

    def __init__(
        self,
        img_rec_path: Union[str, Sequence],
        num_classes: int,
        per_view_shape: Mapping,
        sub_dirs: Sequence[str],
        views: int = 6,
        transforms: Optional[Sequence] = None,
        homo_gen: Optional[object] = None,
    ):
        super(Bev3DDatasetRec, self).__init__()
        self.rec_paths = img_rec_path
        self.transforms = transforms
        self.num_classes = num_classes
        self.views = views
        self.sub_dirs = sub_dirs
        self.per_view_shape = per_view_shape

        assert self.num_classes in [
            1,
            3,
        ], "currently the number of classes must be 3 or 1"

        # get homography matrix.
        self.homo_mat = None
        self.homo_offset = None
        if homo_gen is not None:
            self.homo_gen = homo_gen
            homo_genor = HomoGenerator(**homo_gen)
            self.homo_mat = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()

        # load rec img
        self.load_rec(self.rec_paths)
        self.num_samples = self.acc_lengths[-1]

    def __getstate__(self):
        state = self.__dict__
        state["homo_mat"] = None
        state["homo_offset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.homo_gen is not None:
            homo_genor = HomoGenerator(**self.homo_gen)
            self.homo_mat = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()

    def load_rec(self, rec_paths):
        assert isinstance(rec_paths, list) and len(rec_paths) % 2 == 0
        self.recs = []
        for rec_paths in [
            rec_paths[i : i + 2] for i in range(0, len(rec_paths), 2)
        ]:
            recs_tmp = []
            for rec_path in rec_paths:
                logger.info(f"loading {rec_path}")
                rec_reader = Real3DRecReader(rec_path)
                recs_tmp.append(rec_reader)
            self.recs.append(recs_tmp)
        lengths = [len(rec[-1]) for rec in self.recs]
        self.acc_lengths = np.cumsum(lengths)

    def __len__(self):
        return self.num_samples

    def _get_rec_from_idx(self, idx):
        """Get recio from index."""
        assert idx < len(self)
        for length_idx in range(len(self.acc_lengths)):
            length_i = self.acc_lengths[length_idx]
            if idx > (length_i - 1):
                continue
            rec = self.recs[length_idx]
            if length_idx == 0:
                previous_idx = 0
            else:
                previous_idx = self.acc_lengths[length_idx - 1]
            idx_in_rec = idx - previous_idx
            assert idx_in_rec >= 0
            if idx_in_rec >= len(rec[-1]):
                idx_in_rec = len(rec[-1]) - 1
            return rec, idx_in_rec

    def __getitem__(self, index):
        anns, img, img_info = [], [], []
        rec, idx = self._get_rec_from_idx(index)
        imgs, labels = [rec[-1][idx][0]], [rec[-1][idx][-1]]
        for i in range(
            idx * (self.views - 1), idx * (self.views - 1) + (self.views - 1)
        ):
            imgs.append(rec[0][i][0])
            labels.append(rec[0][i][-1])

        # Adapt to different real3d data formats
        image_key = (
            "file_name"
            if "file_name" in labels[0]["meta"].keys()
            else "image_key"
        )

        # reorder the image by the sub_dir'
        for sub_dir in self.sub_dirs:
            for _label in labels:
                _label_view_name = _label["meta"][image_key].split("__")[1]
                if _label_view_name.split("_0820")[0] == sub_dir:
                    index = labels.index(_label)
                    img_info.append(_label["meta"])
                    img.append(Image.fromarray(imgs[index]))
                    anns.append(_label["objects"])
                    break

        img_id = [str(_iminfo["id"]) for _iminfo in img_info]
        image_name = [_iminfo[image_key] for _iminfo in img_info]

        data_dict = {
            "image_name": image_name,
            "image_id": img_id,
            "img": img,
            "timestamp": np.array([float(labels[0]["meta"]["timestamp"])]),
            "num_classes": self.num_classes,
            "annotations": anns,
        }
        if self.homo_mat is not None:
            data_dict["homography"] = self.homo_mat.copy()
        if self.homo_offset is not None:
            data_dict["homo_offset"] = self.homo_offset.copy()
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict
