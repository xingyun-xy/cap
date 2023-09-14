# Copyright (c) Changan Auto. All rights reserved.
import os
from pathlib import Path
from typing import Union

from .lmdb import Lmdb
from .mxrecord import MXRecord

__all__ = ["get_packtype_from_path"]


def get_packtype_from_path(path: Union[str, Path]):
    """Get PackType from provided path.

    Args:
        path (str or Path): Provided path for pack data.
    Returns:
        PackType.

    """

    path = str(path)

    assert os.path.exists(path), f"{path} does not exist!"

    if os.path.isdir(path) and {"data.mdb", "lock.mdb"}.issubset(
        os.listdir(path)
    ):
        return Lmdb
    elif path.endswith(".rec"):
        return MXRecord
    else:
        raise NotImplementedError("Cannot automatically find pack_type.")
        return
