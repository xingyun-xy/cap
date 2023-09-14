# Copyright (c) Changan Auto. All rights reserved.

from .base import PackType
from .lmdb import Lmdb
from .mxrecord import MXRecord
from .utils import get_packtype_from_path

PackTypeMapper = {"lmdb": Lmdb, "mxrecord": MXRecord}


__all__ = ["PackType", "Lmdb", "MXRecord", "get_packtype_from_path"]
