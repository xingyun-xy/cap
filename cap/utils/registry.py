# Copyright (c) Changan Auto. All rights reserved.
import copy

from cap.registry import Registry
from cap.registry import build_from_cfg as _build_from_cfg

# this file is left for backward compatible

__all__ = ["Registry", "build_from_cfg"]


def build_from_cfg(cfg, registry, default_args=None):

    if not isinstance(cfg, dict):
        raise TypeError("Expected dict, but get {}".format(type(cfg)))

    if default_args is not None:
        if not isinstance(default_args, dict):
            raise TypeError(
                "Expected dict, but get {}".format(type(default_args))
            )  # noqa
        cfg = copy.copy(cfg)
        cfg.update(default_args)

    return _build_from_cfg(registry, cfg)
