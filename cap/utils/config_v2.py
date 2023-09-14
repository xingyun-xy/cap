# Copyright (c) Changan Auto. All rights reserved.
import json
import logging
import os.path as osp
import sys
from ast import literal_eval
from collections.abc import Mapping
from enum import Enum
from functools import reduce
from importlib import import_module
from pathlib import PurePath
from typing import Dict

import numpy as np
import torch
import yaml

from .jsonable import is_jsonable

__all__ = [
    "Config",
    "filter_configs",
]
logger = logging.getLogger(__name__)


class ConfigVersion(Enum):
    v1 = 1
    v2 = 2


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_jsonable(obj):
            return super(JSONEncoder, self).default(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return str(obj)


class Config(object):
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.
    """

    @staticmethod
    def fromfile(filename):
        if isinstance(filename, PurePath):
            filename = filename.as_posix()
        filename = osp.abspath(osp.expanduser(filename))
        if not osp.isfile(filename):
            raise KeyError("file {} does not exist".format(filename))
        if filename.endswith(".py"):
            module_name = osp.basename(filename)[:-3]
            if "." in module_name:
                raise ValueError("Dots are not allowed in config file path.")
            config_dir = osp.dirname(filename)

            old_module = None
            if module_name in sys.modules:
                old_module = sys.modules.pop(module_name)

            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            }
            # IMPORTANT: pop to avoid `import_module` from cache, to avoid the
            # cfg sharing by multiple processes or functions, which may cause
            # interference and get unexpected result.
            sys.modules.pop(module_name)

            if old_module is not None:
                sys.modules[module_name] = old_module

        elif filename.endswith((".yml", ".yaml")):
            with open(filename, "r") as fid:
                cfg_dict = yaml.load(fid, Loader=yaml.Loader)
        else:
            raise IOError(
                "Only py/yml/yaml type are supported now, "
                f"but found {filename}!"
            )
        return Config(cfg_dict, filename=filename)

    def __init__(self, cfg_dict=None, filename=None, encoding="utf-8"):
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            raise TypeError(
                "cfg_dict must be a dict, but got {}".format(type(cfg_dict))
            )

        super(Config, self).__setattr__("_cfg_dict", cfg_dict)
        super(Config, self).__setattr__("_filename", filename)
        if filename:
            with open(filename, "r", encoding=encoding) as f:
                super(Config, self).__setattr__("_text", f.read())
        else:
            super(Config, self).__setattr__("_text", "")

    def merge_from_list_or_dict(self, cfg_opts, overwrite=False):
        """Merge config (keys, values) in a list or dict into this cfg.

        Examples:
            cfg_opts is a list:
            >>> cfg_opts = [
                                'model.backbone.type', 'ResNet18',
                                'model.backbone.num_classes', 10,
                            ]
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet50'))))
            >>> cfg.merge_from_list_or_dict(cfg_opts)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...    model=dict(backbone=dict(type="ResNet18", num_classes=10)))

            cfg_opts is a dict:
            >>> cfg_opts = {'model.backbone.type': "ResNet18",
            ...            'model.backbone.num_classes':10}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet50'))))
            >>> cfg.merge_from_list_or_dict(cfg_opts)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...    model=dict(backbone=dict(type="ResNet18", num_classes=10)))
        Args:
            cfg_opts (list or dict): list or dict of configs to merge from.
            overwrite (bool): Weather to overwrite existing (keys, values).
        """

        if isinstance(cfg_opts, list):
            assert len(cfg_opts) % 2 == 0, (
                "Override list has odd length: "
                f"{cfg_opts}; it must be a list of pairs"
            )
            opts_dict = {}
            for k, v in zip(cfg_opts[0::2], cfg_opts[1::2]):
                opts_dict[k] = v
        elif isinstance(cfg_opts, dict):
            opts_dict = cfg_opts
        else:
            raise ValueError(
                f"cfg_opts should be list or dict, but is {type(cfg_opts)}"
            )

        for full_key, v in opts_dict.items():
            d = self
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, {})
                d = d[subkey]
            subkey = key_list[-1]
            try:
                value = literal_eval(v)
            except Exception:
                raise ValueError(
                    f"The incoming value of key `{full_key}` should be str, "
                    f"list or tuple, but get {v}"
                )
            if isinstance(value, dict):
                raise ValueError(
                    f"The incoming value of key `{full_key}` should be str, "
                    f"list or tuple, but get a dict."
                )

            if subkey in d:
                if overwrite:
                    value = _check_and_coerce_cfg_value_type(
                        value, d[subkey], subkey, full_key
                    )
                    logger.debug(
                        f"'{full_key}: {d[subkey]}' will be overwritten "
                        f"with '{full_key}: {value}'"
                    )
                    d[subkey] = value
                else:
                    logger.warning(
                        f"The incoming `{full_key}` already exists in config, "
                        f"but the obtained `overwrite = false`, which will "
                        f"still use the `{full_key}: {value}` in config."
                    )
            else:
                d[subkey] = value

    def dump_json(self):
        return json.dumps(self._cfg_dict, cls=JSONEncoder, sort_keys=True)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return "Config (path: {}): {}".format(
            self.filename, self._cfg_dict.__repr__()
        )

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        try:
            return getattr(self._cfg_dict, name)
        except AttributeError as e:
            if isinstance(self._cfg_dict, dict):
                try:
                    return self.__getitem__(name)
                except KeyError:
                    raise AttributeError(name)
            raise e

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        self._cfg_dict.__setitem__(name, value)

    def __setitem__(self, name, value):
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)


def filter_configs(
    configs: Config,
    threshold: int = 100,
    return_dict: bool = True,
) -> Dict or Config:
    """
    Filter config to for pprint.format.

    Inplace Tensor with Tensor.shape and inplace numpy with numpy.shape.

    Args:
        configs (Config): configs for pprint.
        threshold (int): threshold of filter to convert shape.
        return_dict (bool): Whether to return dict or config.
    Returns:
        configs (Dict or Config): configs to pprint.
    """

    def filter_elem(cfg):
        if isinstance(cfg, (torch.Tensor, np.ndarray)):
            n_elem = reduce(lambda x, y: x * y, cfg.shape)
            if n_elem >= threshold:
                return cfg.shape
            else:
                return cfg
        elif isinstance(cfg, Mapping):
            return {key: filter_elem(cfg[key]) for key in cfg.keys()}
        elif isinstance(cfg, (list, tuple)):
            return [filter_elem(c) for c in cfg]
        else:
            return cfg

    new_configs = {}
    for k, v in configs._cfg_dict.items():
        new_configs[k] = filter_elem(v)
    if return_dict:
        return new_configs
    else:
        return Config(cfg_dict=new_configs)


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Check that `replacement`, which is intended to replace `original` is \
    of the right type. The type is correct if it matches exactly or is one of \
    a few cases in which the type can be easily coerced.

    Copied from `yacs <https://github.com/rbgirshick/yacs>`.

    """
    _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}

    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # If either of them is None, allow type convert to one of the valid types
    if (replacement is None and original_type in _VALID_TYPES) or (
        original is None and replacement_type in _VALID_TYPES
    ):
        return replacement

    # Cast replacement from from_type to to_type if the replacement and
    # original types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )
