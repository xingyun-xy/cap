import os
from io import BytesIO

import msgpack
import msgpack_numpy
from PIL import Image

from cap.data.datasets.data_packer import Packer
from cap.registry import OBJECT_REGISTRY


class RecImgToLmdbPacker(Packer):
    def __init__(
        self,
        datasets,
        total_len,
        output_dir,
        num_workers,
        anno_transform=None,
    ):
        self.datasets = datasets
        self.anno_transform = anno_transform

        super(RecImgToLmdbPacker, self).__init__(
            output_dir,
            total_len,
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        curr_idx = idx
        for dataset in self.datasets:
            if curr_idx - len(dataset) > 0:
                curr_idx = curr_idx - len(dataset)
                continue
            else:
                img, anno = dataset[curr_idx]
                break
        img_pil = Image.fromarray(img)
        img_bytes = BytesIO()
        img_pil.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        key = anno["meta"]["file_name"]

        return [key, img_bytes]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


class RecIdxToLmdbPacker(Packer):
    def __init__(
        self,
        datasets,
        total_len,
        output_dir,
        num_workers,
        anno_transform=None,
    ):
        self.datasets = datasets
        self.anno_transform = anno_transform
        super(RecIdxToLmdbPacker, self).__init__(
            output_dir,
            total_len,
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        curr_idx = idx
        for dataset in self.datasets:
            if curr_idx - len(dataset) > 0:
                curr_idx = curr_idx - len(dataset)
                continue
            else:
                img, anno = dataset[curr_idx]
                break

        key = anno["meta"]["file_name"]

        return [str(idx), key.encode("ascii")]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


class RecAnnoToLmdbPacker(Packer):
    def __init__(
        self,
        datasets,
        total_len,
        output_dir,
        num_workers,
        anno_transform=None,
    ):
        self.datasets = datasets
        self.anno_transform = anno_transform
        super(RecAnnoToLmdbPacker, self).__init__(
            output_dir,
            total_len,
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        curr_idx = idx
        for dataset in self.datasets:
            if curr_idx - len(dataset) > 0:
                curr_idx = curr_idx - len(dataset)
                continue
            else:
                img, anno = dataset[curr_idx]
                break

        key = anno["meta"]["file_name"]
        anno = msgpack.packb(anno, default=msgpack_numpy.encode)

        return [key, anno]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


@OBJECT_REGISTRY.register
class RecToLmdbPacker:
    def __init__(
        self,
        rec_datasets,
        output_dir,
        num_workers,
        anno_transform=None,
    ):
        self.rec_datasets = rec_datasets
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.anno_transform = anno_transform
        self.queue_maxsize = num_workers * 512
        self.start_method = "spawn"
        self.total_len = sum([len(x) for x in self.rec_datasets])

    @property
    def idx_packer(self):
        if not hasattr(self, "_idx_packer"):
            self._idx_packer = RecIdxToLmdbPacker(
                self.rec_datasets,
                self.total_len,
                os.path.join(self.output_dir, "idx"),
                self.num_workers,
                self.anno_transform,
            )
        return self._idx_packer

    @property
    def img_packer(self):
        if not hasattr(self, "_img_packer"):
            self._img_packer = RecImgToLmdbPacker(
                self.rec_datasets,
                self.total_len,
                os.path.join(self.output_dir, "img"),
                self.num_workers,
            )
        return self._img_packer

    @property
    def anno_packer(self):
        if not hasattr(self, "_anno_packer"):
            self._anno_packer = RecAnnoToLmdbPacker(
                self.rec_datasets,
                self.total_len,
                os.path.join(self.output_dir, "anno"),
                self.num_workers,
                self.anno_transform,
            )
        return self._anno_packer

    def pack_img(self):
        img_packer = self.img_packer
        img_packer()

    def pack_anno(self):
        anno_packer = self.anno_packer
        anno_packer()

    def pack_idx(self):
        idx_packer = self.idx_packer
        idx_packer()
