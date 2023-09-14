# Copyright (c) Changan Auto. All rights reserved.
import copy
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np
from capbc.utils import deprecated
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, is_list_of_type, to_cuda

__all__ = [
    "ListToDict",
    "DeleteKeys",
    "RenameKeys",
    "Undistortion",
    "PILToTensor",
    "TensorToNumpy",
    "AddKeys",
    "CopyKeys",
    "Cast",
]


@OBJECT_REGISTRY.register
class ListToDict(object):
    """Convert list args to dict.

    Args:
        keys: keys for each object in args.
    """

    def __init__(self, keys: List[str]):
        assert is_list_of_type(
            keys, str
        ), "expect list/tuple of str, but get%s" % type(keys)
        self.keys = keys

    def __call__(self, args):
        assert len(self.keys) == len(args), "%d vs. %d" % (
            len(self.keys),
            len(args),
        )
        return {k: v for k, v in zip(self.keys, args)}


@deprecated("Please use `ListToDict` instead")
@OBJECT_REGISTRY.register
class List2Dict(ListToDict):
    def __init__(self, *args):
        super(List2Dict, self).__init__(*args)


@OBJECT_REGISTRY.register
class DeleteKeys(object):
    """Delete keys in input dict.

    Args:
        keys: key list to detele

    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            if key in data:
                data.pop(key)
        return data


@OBJECT_REGISTRY.register
class RenameKeys(object):
    """Rename keys in input dict.

    Args:
        keys: key list to rename, in "old_name | new_name" format.

    """

    def __init__(self, keys: List[str], split="|"):
        self.split = split
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            assert self.split in key
            old_key, new_key = key.split(self.split)
            old_key = old_key.strip()
            new_key = new_key.strip()
            if old_key in data:
                data[new_key] = data.pop(old_key)
        return data


@OBJECT_REGISTRY.register
class Undistortion(object):  # noqa: D205,D400
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to
        undistor ``PIL Image`` or ``numpy.ndarray``.

    """

    def _to_pillow(self, data, mode="I"):
        # convert a numpy.ndarray img to pillow img
        if isinstance(data, (list, tuple)):
            return [self._to_pillow(_, mode) for _ in data]
        else:
            return Image.fromarray(data).convert(mode)

    def _undistort(self, data, intrinsic, distor_coeff):
        # scale intrinsics matrix
        intrinsic = intrinsic.copy()
        h, w = data.shape[0:2]
        intrinsic[0, :] *= w
        intrinsic[1, :] *= h

        mapx, mapy = cv2.initUndistortRectifyMap(
            intrinsic, distor_coeff, None, intrinsic, (w, h), cv2.CV_32FC1
        )
        img_ud = cv2.remap(data, mapx, mapy, interpolation=cv2.INTER_NEAREST)

        return img_ud

    def _to_undistort_array(
        self, data, intrinsic, distor_coeff, is_depth=False
    ):
        # convert a pillow img to undistor numpy.ndarray img

        if isinstance(data, (list, tuple)):
            res2 = []
            for frame in data:
                res1 = []
                for img in frame:
                    res1.append(
                        self._undistort(np.array(img), intrinsic, distor_coeff)
                    )
                res2.append(res1)
            return res2
        else:
            res = self._undistort(np.array(data), intrinsic, distor_coeff)
            return res

    def __call__(self, data):

        data["color_imgs"] = self._to_pillow(
            self._to_undistort_array(
                data["pil_imgs"], data["intrinsics"], data["distortcoef"]
            ),
            mode="RGB",
        )

        if "obj_mask" in data:
            data["obj_mask"] = self._to_pillow(
                self._to_undistort_array(
                    data["obj_mask"], data["intrinsics"], data["distortcoef"]
                ),
                mode="I",
            )

        if "front_mask" in data:
            data["front_mask"] = self._to_pillow(
                self._to_undistort_array(
                    data["front_mask"], data["intrinsics"], data["distortcoef"]
                ),
                mode="I",
            )

        return data

    def __repr__(self):
        return "Undistortion"


@OBJECT_REGISTRY.register
class PILToTensor(object):
    r"""Convert PIL Image to Tensor."""

    def __init__(self):
        super(PILToTensor, self).__init__()

    def __call__(self, data):
        if isinstance(data, Image.Image):
            data = F.pil_to_tensor(data)
        elif isinstance(data, dict):
            for k in data:
                print(k)
                data[k] = self(data[k])
        elif isinstance(data, Sequence) and not isinstance(data, str):
            data = type(data)(self(d) for d in data)
        return data


@OBJECT_REGISTRY.register
class TensorToNumpy(object):
    r"""Convert tensor to numpy."""

    def __init__(self):
        super(TensorToNumpy, self).__init__()

    def __call__(self, data):
        if isinstance(data, Tensor):
            data = data.cpu().numpy().squeeze()
        if isinstance(data, dict):
            for k in data:
                data[k] = self(data[k])
        elif isinstance(data, Sequence) and not isinstance(data, str):
            data = type(data)(self(d) for d in data)
        return data


@OBJECT_REGISTRY.register
class ToCUDA(object):
    r"""
    Move Tensor to cuda device.

    Args:
        device (int, optional): The destination GPU device idx.
            Defaults to the current CUDA device.
    """

    def __init__(self, device: int = None):
        super(ToCUDA, self).__init__()
        self.device = device

    def __call__(self, data):
        return to_cuda(data, self.device)


@OBJECT_REGISTRY.register
class AddKeys(object):
    """Add new key-value in input dict.

    Frequently used when you want to add dummy keys to data dict
    but don't want to change code.

    Args:
        kv: key-value data dict.

    """

    def __init__(self, kv: Dict[str, Any]):
        assert isinstance(kv, Mapping)
        self._kv = kv

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self._kv:
            assert key not in data, f"{key} already exists in data."
            data[key] = self._kv[key]
        return data


@OBJECT_REGISTRY.register
class CopyKeys(object):
    """Copy new key in input dict.

    Frequently used when you want to cache keys to data dict
    but don't want to change code.

    Args:
        kv: key-value data dict.

    """

    def __init__(self, keys: List[str], split="|"):
        self.split = split
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            assert self.split in key
            old_key, new_key = key.split(self.split)
            old_key = old_key.strip()
            new_key = new_key.strip()
            if old_key in data:
                data[new_key] = copy.deepcopy(data[old_key])
        return data


class Cast(object):
    """Data type transformer.

    Parameters
    ----------
    dtype: str
        Transform input to dtype.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, data):
        assert isinstance(data, np.ndarray)
        return (
            data.astype(self.dtype)
            if np.dtype(data.dtype) != np.dtype(self.dtype)
            else data
        )


def _call_ts_func(transformers, func_name, *args, **kwargs):
    for ts_i in _as_list(transformers):
        if hasattr(ts_i, func_name) and callable(getattr(ts_i, func_name)):
            getattr(ts_i, func_name)(*args, **kwargs)
