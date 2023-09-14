# Copyright (c) Changan Auto. All rights reserved.

import os

import cv2
import lmdb
import msgpack
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from cap.registry import OBJECT_REGISTRY

__all__ = ["AutoDetPacker", "AutoSegPacker", "Auto2dFromLMDB"]


class AutoDetPacker(data.Dataset):
    """Packer used to create map-style dataset for the detection task.

    Your dataset should be organized according to the following directory:

    └── directory
        ├── annotations
        │   ├── train.json
        │   └── val.json
        ├── train
        └── val

    Args:
        directory (str): Path for dataset.
        split_name (str): Split name of data, such as train, val and so on.
    """

    def __init__(self, directory, split_name):
        self.directory = directory
        self.split_name = split_name
        self.coco = COCO(
            os.path.join(directory, "annotations", self.split_name + ".json")
        )
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])
        self.labels = {}  # {0: 'person', 1: 'bicycle', ..., 79: 'toothbrush'}
        self.coco_labels = {}  # {0: 1, 1: 2, ..., 79: 90}
        self.coco_labels_inverse = {}  # {1: 0, 2: 1, ..., 90: 79}
        for c in categories:
            self.coco_labels[len(self.labels)] = c["id"]
            self.coco_labels_inverse[c["id"]] = len(self.labels)
            self.labels[len(self.labels)] = c["name"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_info, img = self.load_image(idx)
        annotations = self.load_annotations(idx, image_info)
        results = {}
        results.update(image_info)
        results.update(annotations)
        results["img"] = img
        return results

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(
            self.directory,
            "{}".format(self.split_name),
            image_info["file_name"],
        )
        img = cv2.imread(path)
        return image_info, img

    def load_annotations(self, image_index, image_info):
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index],
            iscrowd=None,
        )
        annotations = {
            "gt_bboxes": np.zeros((0, 4)),
            "gt_classes": np.zeros(0),
        }

        if len(annotations_ids) == 0:
            # iscrowd=1 for all bbox
            return annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)

        gt_bboxes = []
        gt_classes = []
        for _i, ann in enumerate(coco_annotations):
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, image_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, image_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.labels:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            if ann.get("iscrowd", False):
                gt_classes.append(-1)
            else:
                gt_classes.append(self.coco_label_to_label(ann["category_id"]))

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_classes = np.array(gt_classes, dtype=np.int64)

        annotations = {"gt_bboxes": gt_bboxes, "gt_classes": gt_classes}

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image_info["width"]) / float(image_info["height"])


class AutoSegPacker(data.Dataset):  # noqa: D205,D400
    """Packer used to create map-style dataset for the semantic
    segmentation task.

    Your dataset should be organized according to the following directory:

    └── directory
        ├── annotations
        │   ├── train.json
        │   └── val.json
        ├── train
        └── val

    Args:
        directory (str): Path for dataset.
        split_name (str): Split name of data, such as train, val and so on.
        seg_map_dict (dict): A dict of gt png pixel value to label id.
    """

    def __init__(self, directory, split_name, seg_map_dict=None):
        self.directory = directory
        self.split_name = split_name
        self.seg_map_dict = seg_map_dict

        self.coco = COCO(
            os.path.join(directory, "annotations", self.split_name + ".json")
        )
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.labels = {}  # {0: 'road', 1: 'background', ...}
        for c in categories:
            self.labels[len(self.labels)] = c["name"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_info, img = self.load_image(idx)
        annotations = self.load_annotations(image_info)
        results = {}
        results.update(image_info)
        results["img"] = img
        results["gt_seg"] = annotations
        return results

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(
            self.directory,
            "{}".format(self.split_name),
            image_info["file_name"],
        )
        img = cv2.imread(path)
        return image_info, img

    def load_annotations(self, image_info):
        path = os.path.join(
            self.directory,
            "{}".format(self.split_name),
            image_info["file_name"],
        ).replace(".jpg", "_label.png")

        gt_seg_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if self.seg_map_dict is not None:
            new_gt_semantic_seg = gt_seg_map.copy()
            assert isinstance(self.seg_map_dict, dict)
            for src_label, target_label in self.seg_map_dict.items():
                idnex = gt_seg_map == src_label
                new_gt_semantic_seg[idnex] = target_label
            gt_seg_map = new_gt_semantic_seg

        return gt_seg_map

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image["width"]) / float(image["height"])


class Datum(object):
    """Class for serializing and deserializing.

    Args:
        datas (dict): Keys inclued `img`, `height`, `width`, `gt_classes`,
        `gt_bboxes`, `gt_seg` and so on.
    """

    def __init__(self, datas=None):
        self.datas = datas

    def SerializeToString(self):
        pack_dict = {}
        for k, v in self.datas.items():
            if k == "img":
                img = self.datas["img"][0].numpy()
                pack_dict["img"] = cv2.imencode(".jpg", img)[1].tobytes()
            elif k == "gt_seg":
                gt_seg = self.datas["gt_seg"][0].numpy()
                pack_dict["gt_seg"] = cv2.imencode(".png", gt_seg)[1].tobytes()
            elif k in ["height", "width", "gt_classes", "id"]:
                new_v = np.asarray(v, dtype=np.int64).tobytes()
                pack_dict[k] = new_v
            elif k == "gt_bboxes":
                new_v = np.asarray(v, dtype=np.float).tobytes()
                pack_dict[k] = new_v
            elif k == "file_name":
                new_v = bytes(v[0], encoding="utf8")
                pack_dict[k] = new_v
            else:
                assert "wrong key"
        return msgpack.packb(pack_dict)

    def ParseFromString(self, raw_data):
        self.image_info = {}
        self.image = np.frombuffer(raw_data["img"], dtype=np.uint8)
        self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
        for k, v in raw_data.items():
            if k == "img":
                continue
            elif k == "gt_seg":
                new_v = np.frombuffer(raw_data["gt_seg"], dtype=np.uint8)
                new_v = cv2.imdecode(new_v, cv2.IMREAD_UNCHANGED)
            elif k in ["height", "width", "gt_classes", "id"]:
                new_v = np.frombuffer(v, dtype=np.int64)
            elif k == "gt_bboxes":
                new_v = np.frombuffer(v, dtype=np.float)
                if len(new_v) != 0:
                    new_v = new_v.reshape((int(new_v.shape[0] / 4), -1))
            elif k == "file_name":
                new_v = bytes.decode(v)
            self.image_info[k] = new_v
        return self.image_info, self.image


def Auto2D2lmdb(
    lmdb_path, directory, split_name, task, num_workers, shuffle=True, **kwargs
):
    """Pack the original data into lmdb.

    Args:
        lmdb_path (str): Storage path of the generated lmdb file.
        directory (str): Storage path of the data to be packaged.
        split_name (str): Split name of data, such as train, val and so on.
        task (str): Task name, such as `train`, `val` and so on.
        num_workers (int): The num workers for reading data, same as
            DataLoader.
        shuffle (bool): Same as DataLoader.
        **kwargs (dict): Receive extra parameters.
    """
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    if "det" in task:
        dataset = AutoDetPacker(directory=directory, split_name=split_name)
    elif "seg" in task:
        dataset = AutoSegPacker(
            directory=directory,
            split_name=split_name,
            seg_map_dict=kwargs.get("seg_map_dict", None),
        )
    else:
        assert "wrong task type"

    data_loader = DataLoader(dataset, num_workers=num_workers, shuffle=shuffle)
    db = lmdb.open(
        lmdb_path, map_size=1099511627776 * 2, meminit=False, map_async=True
    )
    txn = db.begin(write=True)
    for idx, datas in enumerate(data_loader):
        base_data = Datum(datas=datas)
        txn.put(
            "{}".format(idx).encode("ascii"),
            base_data.SerializeToString(),
        )
        if idx % 500 == 0:
            print("[%d/%d]" % (idx, len(dataset)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    db.sync()
    db.close()


@OBJECT_REGISTRY.register
class Auto2dFromLMDB(data.Dataset):
    """Unpack data from lmdb.

    Args:
        data_path (str): Path of lmdb file.
        transforms (list): A list of transform.
        num_samples (int): As it says.
        to_rgb (bool): Whether transform color_space of img to `RGB`.
        return_orig_img (bool): Whether to return an extra original img,
            orig_img can usually be used on visualization.
    """

    def __init__(
        self,
        data_path,
        transforms=None,
        num_samples=1,
        to_rgb=False,
        return_orig_img=False,
    ):
        self.root = data_path
        self.transforms = transforms
        self.num_samples = num_samples
        self.samples = [
            "{}".format(idx).encode("ascii") for idx in range(num_samples)
        ]
        self.to_rgb = to_rgb
        self.txn = None
        self.datum = Datum()
        self.return_orig_img = return_orig_img

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__
        state["txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        env = lmdb.open(
            self.root,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = env.begin(write=False)

    def __getitem__(self, item):
        data = {}
        raw_data = self.txn.get(self.samples[item])
        raw_data = msgpack.unpackb(raw_data, raw=False)
        image_info, image = self.datum.ParseFromString(raw_data)
        if self.return_orig_img:
            data["orig_img"] = image
        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            color_space = "rgb"

        data.update(image_info)
        data["img_name"] = data.pop("file_name")
        data["img_height"] = data.pop("height")
        data["img_width"] = data.pop("width")
        data["img_id"] = data.pop("id")
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.root}, "
        repr_str += f"transforms={self.transforms}, "
        repr_str += f"num_samples={self.num_samples}, "
        repr_str += f"to_rgb={self.to_rgb}, "
        repr_str += f"return_orig_img={self.return_orig_img}"
        return repr_str
