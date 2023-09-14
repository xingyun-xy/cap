# Copyright (c) Changan Auto. All rights reserved.

import changan_plugin_pytorch.nn.bgr_to_yuv444 as b2y
import torch

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list

try:
    import timm
except ImportError:
    timm = None

__all__ = [
    "ConvertLayout",
    "BgrToYuv444",
    "OneHot",
    "LabelSmooth",
    "TimmTransforms",
    "TimmMixup",
]


@OBJECT_REGISTRY.register
class ConvertLayout(object):
    """
    ConvertLayout is used for layout convert.

    .. note::
        Affected keys: 'img'.

    Args:
        hwc2chw (bool): Whether to convert hwc to chw.
        keys (list)：make layout convert for the data[key]
    """

    def __init__(self, hwc2chw=True, keys=None):
        self.hwc2chw = hwc2chw
        self.keys = _as_list(keys) if keys else ["img"]

    def __call__(self, data):
        for key in self.keys:
            assert key in data
            if isinstance(data[key], list):
                for i, image in enumerate(data[key]):
                    if self.hwc2chw:
                        image = image.permute((2, 0, 1))
                    else:
                        image = image.permute((1, 2, 0))
                    data[key][i] = image
            if isinstance(data[key], torch.Tensor):
                image = data[key]
                if self.hwc2chw:
                    image = image.permute((2, 0, 1))
                else:
                    image = image.permute((1, 2, 0))
                data[key] = image
        return data


@OBJECT_REGISTRY.register
class BgrToYuv444(object):
    """
    BgrToYuv444 is used for color format convert.

    .. note::
        Affected keys: 'img'.

    Args:
        rgb_input (bool): The input is rgb input or not.
    """

    def __init__(self, rgb_input=False):
        self.rgb_input = rgb_input

    def __call__(self, data):
        image = data["img"] if isinstance(data, dict) else data
        ndim = image.ndim
        if ndim == 3:
            image = torch.unsqueeze(image, 0)
        if image.shape[1] == 6:
            image1 = b2y.bgr_to_yuv444(image[:, :3], self.rgb_input).float()
            image2 = b2y.bgr_to_yuv444(image[:, 3:], self.rgb_input).float()
            image = torch.cat((image1, image2), dim=1)
        else:
            image = b2y.bgr_to_yuv444(image, self.rgb_input)
            image = image.float()
        if ndim == 3:
            image = image[0]
        if isinstance(data, dict):
            data["img"] = image
            return data
        else:
            return image


@OBJECT_REGISTRY.register
class OneHot(object):
    """
    OneHot is used for convert layer to one-hot format.

    .. note::
        Affected keys: 'labels'.

    Args:
        num_classes (int): Num classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        assert "labels" in data
        target = data["labels"]
        if len(target.shape) == 1:
            # for scatter
            target = torch.unsqueeze(target, dim=1)
        batch_size = target.shape[0]
        one_hot = torch.zeros(batch_size, self.num_classes)
        one_hot = one_hot.scatter_(1, target, 1)
        data["labels"] = one_hot
        return data


@OBJECT_REGISTRY.register
class LabelSmooth(object):
    """
    LabelSmooth is used for label smooth.

    .. note::
        Affected keys: 'labels'.

    Args:
        num_classes (int): Num classes.
        eta (float): Eta of label smooth.
    """

    def __init__(self, num_classes, eta=0.1):
        self.num_classes = num_classes
        self.on_value = torch.tensor([1 - eta + eta / num_classes])
        self.off_value = torch.tensor([eta / num_classes])

    def __call__(self, data):
        assert "labels" in data
        target = data["labels"]
        if len(target.shape) == 1:
            # for scatter
            target = torch.unsqueeze(target, dim=1)
        batch_size = target.shape[0]
        one_hot = torch.zeros(batch_size, self.num_classes)
        one_hot = one_hot.scatter_(1, target, 1)
        target = torch.where(one_hot == 0, self.off_value, self.on_value)
        data["labels"] = target
        return data


@OBJECT_REGISTRY.register
class TimmTransforms(object):
    """
    Transforms of timm.

    .. note::
        Affected keys: 'img'.

    Args:
        args are the same as timm.data.create_transform
    """

    def __init__(self, *args, **kwargs):
        if timm is None:
            raise ModuleNotFoundError("timm is required!")
        self.transform = timm.data.create_transform(*args, **kwargs)

    def __call__(self, data):
        data["img"] = self.transform(data["img"])
        return data


@OBJECT_REGISTRY.register
class TimmMixup(object):
    """
    Mixup of timm.

    .. note::
        Affected keys: 'img', 'labels'.

    Args:
        args are the same as timm.data.Mixup
    """

    def __init__(self, *args, **kwargs):
        if timm is None:
            raise ModuleNotFoundError("timm is required!")
        self.mixup = timm.data.Mixup(*args, **kwargs)

    def __call__(self, data):
        x, target = data["img"], data["labels"]
        x, target = self.mixup(x, target)
        data["img"], data["labels"] = x, target
        return data
