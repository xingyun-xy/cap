# Copyright (c) Changan Auto. All rights reserved.
from typing import Union

import lmdb

from .base import PackType

__all__ = ["Lmdb"]


class Lmdb(PackType):
    """
    Abstact class of LMDB, which include all operators.

    Args:
        uri (str): Path to lmdb file.
        writable (bool): Writable flag for opening LMDB.
        reopen_before_write (bool): Whether to reopen file while writing.
        kwargs (dict): Kwargs for open lmdb file.

    """

    def __init__(
        self,
        uri: [str],
        writable: [bool] = True,
        reopen_before_write: [bool] = False,
        **kwargs,
    ):
        self.uri = uri
        self.writable = writable
        self.reopen_before_write = reopen_before_write
        self.kwargs = kwargs

        # default lmdb settings
        self.kwargs["map_size"] = self.kwargs.get("map_size", 1024 ** 4)
        self.kwargs["meminit"] = self.kwargs.get("meminit", False)
        self.kwargs["map_async"] = self.kwargs.get("map_async", True)

        if not writable:
            self.kwargs["readonly"] = True
            self.kwargs["lock"] = False

        # LMDB env
        self.env = None
        self.txn = None
        if self.reopen_before_write:
            self.open()

    def read(self, idx: Union[int, str]) -> bytes:
        """Read data by idx."""
        idx = "{}".format(idx).encode("ascii")
        return self.txn.get(idx)

    def write(self, idx: Union[int, str], record: bytes):
        """Write data into lmdb file."""
        if not self.reopen_before_write:
            self.open()
        self.txn.put("{}".format(idx).encode("ascii"), record)
        self.txn.commit()
        if not self.reopen_before_write:
            self.close()

    def open(self):
        """Open lmdb file."""
        self.env = lmdb.open(self.uri, **self.kwargs)
        self.txn = self.env.begin(write=self.writable)

    def close(self):
        """Close lmdb file."""
        if self.env is not None and self.txn is not None:
            self.env.close()
            self.env = None
            self.txn = None

    def reset(self):
        """Reset open file."""
        if self.env is None and self.txn is None:
            self.open()
        else:
            self.close()
            self.open()

    def get_keys(self):
        """Get all keys."""
        try:
            idx = "{}".format("__len__").encode("ascii")
            return range(int(self.txn.get(idx)))
        except Exception:
            # traversal may be slow while too much keys
            keys = []
            for key, _value in self.txn.cursor():
                keys.append(key)
            return keys

    def __getstate__(self):
        state = self.__dict__
        self.close()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.open()
