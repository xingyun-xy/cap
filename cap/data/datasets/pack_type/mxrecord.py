# Copyright (c) Changan Auto. All rights reserved.
import os
from typing import Optional

from .base import PackType

try:
    import mxnet as mx
except ImportError:
    mx = None

__all__ = ["MXRecord"]


class MXRecord(PackType):
    """
    Abstract class of RecordIO, include all operators.

    While write_part_size > 1, multi recs will be got,
    names like: *.rec.part0; *.rec.part1.

    Args:
        uri (str): Path to record file.
        idx_path (str): Path to idx file.
        writable (bool): Writable flag for opening MXRecord.
        write_part_size (int): The size(MB) for each part
            if you want to split rec into parts.
            Non positive value means we do not do partition.
        key_type (type): Data type for record keys.
    """

    def __init__(
        self,
        uri: str,
        idx_path: Optional[str] = None,
        writable: bool = True,
        write_part_size: int = -1,
        key_type: type = int,
    ):
        if mx is None:
            raise ModuleNotFoundError("mxnet is required by MXRecord")

        self.uri = uri + ".rec" if os.path.isdir(uri) else uri
        self.idx_path = uri + ".idx" if idx_path is None else idx_path
        self.flag = "w" if writable else "r"
        self.write_part_size = write_part_size
        self.first_reopen_in_process = True
        self.part_id = 0
        self.key_type = key_type

        self.record = None
        self.open()

    def read(self, idx: int) -> bytes:
        """Read mxrecord file."""
        assert self.record is not None, "Please open mxrecord before read."
        return self.record.read_idx(idx)

    def write(self, idx: int, record: bytes):
        """Write record data into mxrecord file."""
        assert self.record is not None, "Please open mxrecord before write."

        # mxrecord should open and write in same process.
        if self.first_reopen_in_process:
            self.close()
            if self.write_part_size > 0:
                self.open(self.part_id)
            else:
                self.open()
            self.first_reopen_in_process = False

        if (
            self.write_part_size > 0
            and self.record.tell() > self.write_part_size * 1024 * 1024
        ):
            self.part_id += 1
            self.close()
            self.open(self.part_id)

        self.record.write_idx(idx, record)

    def open(self, part_id: int = -1):
        """Open mxrecord file."""
        if self.record is not None:
            return
        if part_id < 0:
            self.record = mx.recordio.MXIndexedRecordIO(
                idx_path=self.idx_path,
                uri=self.uri,
                flag=self.flag,
                key_type=self.key_type,
            )
        else:
            self.record = mx.recordio.MXIndexedRecordIO(
                idx_path=self.idx_path + ".part{}".format(part_id),
                uri=self.uri + ".part{}".format(part_id),
                flag=self.flag,
                key_type=self.key_type,
            )

    def close(self):
        """Close mxrecord file."""
        if self.record:
            self.record.close()
            self.record = None

    def reset(self):
        """Reset the pointer to first item."""
        if self.record is not None:
            self.record.reset()
        else:
            self.open()

    def get_keys(self):
        """Get all keys."""
        try:
            keys = self.record.keys
            assert len(keys) > 0
            return range(len(keys))
        except Exception:
            # traversal may be slow while too much keys
            keys = []
            cnt = 0
            while self.record.read():
                keys.append(cnt)
                cnt += 1
            return keys

    def __getstate__(self):
        state = self.__dict__
        self.close()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.open()
