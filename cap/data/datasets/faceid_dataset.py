import logging
import numbers
import struct
from typing import List, Optional

import cv2

try:
    import mxnet as mx
    from mxnet.recordio import IRHeader
except ImportError:
    mx = None
    IRHeader = None

import numpy as np
from torch.utils.data import Dataset

from cap.registry import OBJECT_REGISTRY
from .pack_type import PackTypeMapper

__all__ = ["DeepInsightRecordDataset"]

_IR_FORMAT_64 = "IdQQ"
_IR_SIZE_64 = struct.calcsize(_IR_FORMAT_64)

logger = logging.getLogger(__name__)


def unpack_fp64(s):
    header = IRHeader(*struct.unpack(_IR_FORMAT_64, s[:_IR_SIZE_64]))
    s = s[_IR_SIZE_64:]
    if header.flag > 0:
        header = header._replace(
            label=np.frombuffer(s, np.float64, header.flag)
        )
        s = s[header.flag * 8 :]
    return header, s


@OBJECT_REGISTRY.register
class DeepInsightRecordDataset(Dataset):
    """
    DeepInsightRecordDataset provides the method of reading mx faceid rec data.

    Args:
        rec_path : the path of faceid rec file.
        idx_path : the path of faceid rec idx file.
        transfroms : transfroms of data before using.
        unpack64 : use unpack_fp64.
    """

    def __init__(
        self,
        rec_path: str,
        idx_path: str,
        transforms: Optional[List] = None,
        unpack64: bool = True,
    ):
        super(DeepInsightRecordDataset, self).__init__()

        if mx is None:
            raise ModuleNotFoundError(
                "mxnet is required by ImageNetFromRecord"
            )

        self.unpack64 = unpack64
        self.transforms = None
        self.path_imgrec = rec_path
        self.path_imgidx = idx_path
        self.pack_type = PackTypeMapper["mxrecord"]

        if transforms is not None:
            self.transforms = transforms
        self._init_pack()

    def _init_pack(self):
        self.pack_file = self.pack_type(
            uri=self.path_imgrec, idx_path=self.path_imgidx, writable=False
        )
        self.pack_file.open()
        s = self.pack_file.read(0)

        if self.unpack64:
            logger.info(
                "Using unpack64, "
                "recpath: {}, idxpath: {}".format(
                    self.path_imgrec, self.path_imgidx
                )
            )
            header0, _ = unpack_fp64(s)
        else:
            logger.info(
                "Using unpack32, "
                "recpath: {}, idxpath: {}".format(
                    self.path_imgrec, self.path_imgidx
                )
            )
            header0, _ = mx.recordio.unpack(s)

        if header0.flag > 0:
            self.id_seq = list(
                range(int(header0.label[0]), int(header0.label[1]))
            )

            self.id2range = {}
            self.id_num = {}
            self.imgidx = []
            self.imgid2range = {}
            for identity in self.id_seq:
                id_offset = len(self.imgidx)
                s = self.pack_file.read(identity)
                header, _ = unpack_fp64(s)
                id_start, id_end = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (id_start, id_end)
                num_sample = id_end - id_start
                self.id_num[identity] = num_sample
                self.imgidx += list(range(*self.id2range[identity]))
                self.imgid2range[identity] = (
                    id_offset,
                    id_offset + num_sample,
                )
        else:
            self.imgidx = np.array(list(self.pack_file.get_keys()))

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._init_pack()

    def __getitem__(self, index: int):
        idx = self.imgidx[index]
        s = self.pack_file.read(idx)
        if self.unpack64:
            header, img = unpack_fp64(s)
        else:
            header, img = mx.recordio.unpack(s)

        label = header.label

        if not isinstance(label, numbers.Number):
            label = label[0]
        sample = cv2.imdecode(
            np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        data = {"img": sample, "labels": int(label), "layout": "hwc"}

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.imgidx)
