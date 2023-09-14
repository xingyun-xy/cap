import json
import logging
import os
from multiprocessing import Pool

import msgpack
import msgpack_numpy
from torchvision.transforms import Compose

from cap.data.datasets.data_packer import Packer
from cap.registry import OBJECT_REGISTRY

__all__ = [
    "DetSeg2DPacker",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DetSeg2DAnnoPacker(Packer):
    def __init__(
        self,
        data_queue,
        output_dir,
        num_workers,
        anno_transform,
    ):
        self.data_queue = data_queue
        self.packed = 0
        self.anno_transform = anno_transform

        super(DetSeg2DAnnoPacker, self).__init__(
            output_dir,
            len(data_queue),
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        _img_dir, anno = self.data_queue[idx]
        anno_original, _anno_transformed = anno
        key = "image_uid" if "image_uid" in anno_original else "image_key"
        key = anno_original[key]

        if "parsing_map_urls" in _anno_transformed:
            with open(
                _anno_transformed["parsing_map_urls"],
                "rb",
            ) as seg_img:
                seg_img_bytes = seg_img.read()

            _anno_transformed["seg_label_img_format_bytes"] = seg_img_bytes

        _anno_transformed = msgpack.packb(
            _anno_transformed, default=msgpack_numpy.encode
        )

        return [key, _anno_transformed]

    def _write(self, idx, data):
        self.packed += 1
        self.pack_file.write(data[0], data[1])


@OBJECT_REGISTRY.register
class DetSeg2DImgPacker(Packer):
    def __init__(
        self,
        data_queue,
        max_data_num,
        output_dir,
        num_workers=16,
    ):
        self.data_queue = data_queue
        self.packed = 0
        super(DetSeg2DImgPacker, self).__init__(
            output_dir,
            max_data_num,
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        _img_dir, anno = self.data_queue[idx]
        anno_original, _anno_transformed = anno
        key = "image_uid" if "image_uid" in anno_original else "image_key"
        key = anno_original[key]

        with open(
            os.path.join(
                _img_dir,
                key,
            ),
            "rb",
        ) as image:
            image_bytes = image.read()
        self.packed += 1
        return [key, image_bytes]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


@OBJECT_REGISTRY.register
class DetSeg2DIdxPacker(Packer):
    def __init__(
        self,
        data_queue,
        output_dir,
        num_workers,
        anno_transform=None,
    ):
        self.anno_transform = anno_transform
        self.data_queue = data_queue

        super(DetSeg2DIdxPacker, self).__init__(
            output_dir,
            len(self.data_queue),
            "lmdb",
            num_workers,
        )
        self.packed = 0

    def pack_data(self, idx):
        _img_dir, anno = self.data_queue[idx]
        anno_original, _anno_transformed = anno
        key = "image_uid" if "image_uid" in anno_original else "image_key"
        key = anno_original[key]

        return [str(idx), key.encode("ascii")]

    def _write(self, idx, data):
        self.packed += 1
        self.pack_file.write(data[0], data[1])


@OBJECT_REGISTRY.register
class DetSeg2DPacker:
    def __init__(
        self,
        folder_anno_pairs,
        output_dir,
        num_workers,
        anno_transform=None,
    ):
        self.folder_anno_pairs = folder_anno_pairs
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.queue_maxsize = num_workers * 512
        self.data_queue = []
        self.start_method = "fork"

        if anno_transform is not None and isinstance(anno_transform, list):
            self.anno_transform = Compose(anno_transform)
        else:
            self.anno_transform = anno_transform
        self.preprocess_anno_list()

    @property
    def idx_packer(self):
        if not hasattr(self, "_idx_packer"):
            self._idx_packer = DetSeg2DIdxPacker(
                self.data_queue,
                os.path.join(self.output_dir, "idx"),
                self.num_workers,
                self.anno_transform,
            )
        return self._idx_packer

    @property
    def img_packer(self):
        if not hasattr(self, "_img_packer"):
            self._img_packer = DetSeg2DImgPacker(
                self.data_queue,
                self.idx_packer.num_blocks,
                os.path.join(self.output_dir, "img"),
                self.num_workers,
            )
        return self._img_packer

    @property
    def anno_packer(self):
        if not hasattr(self, "_anno_packer"):
            self._anno_packer = DetSeg2DAnnoPacker(
                self.data_queue,
                os.path.join(self.output_dir, "anno"),
                self.num_workers,
                self.anno_transform,
            )
        return self._anno_packer

    def preprocess_anno_list(self):
        if self.anno_transform is not None:
            anno_list = []
            for img_dir_url, anno_file_url in self.folder_anno_pairs:
                with open(anno_file_url) as f:
                    for line in f.readlines():
                        anno = json.loads(line)
                        anno_list.append((img_dir_url, anno))

            if self.num_workers > 0:
                with Pool(self.num_workers) as p:
                    transformed_annos = p.map(self.transfrom_anno, anno_list)

            else:
                transformed_annos = map(self.transfrom_anno, anno_list)

            for img_dir_url, item in transformed_annos:
                if item[1] is not None:
                    self.data_queue.append((img_dir_url, item))
        else:
            for img, anno in self.folder_anno_pairs:
                self.data_queue.append((img, (anno, anno)))

    def transfrom_anno(self, x):
        return (x[0], (x[1], self.anno_transform((x[0], x[1]))))

    def pack_idx(self):
        idx_packer = self.idx_packer
        idx_packer()

    def pack_img(self):
        img_packer = self.img_packer
        img_packer()

    def pack_anno(self):
        anno_packer = self.anno_packer
        anno_packer()
