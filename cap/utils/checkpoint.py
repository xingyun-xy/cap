# Copyright (c) Changan Auto. All rights reserved.
import functools
import logging
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from capbc.filestream.bucket.client import BucketClient

from cap.utils.hash import get_hash_file_if_hashed_and_local
from .filesystem import get_filesystem

__all__ = [
    "load_checkpoint",
    "load_state_dict",
    "update_state_dict_by_strip_prefix",
]

logger = logging.getLogger(__name__)


def load_checkpoint(
    path_or_dict: Union[str, IO, Path, dict],
    map_location: Optional[str] = None,
    strip_prefix: str = "module.",
    state_dict_update_func: Optional[Callable] = None,
    check_hash: bool = True,
) -> Dict:
    """Load checkpoint from path,url or SDAModel.

    Args:
        path_or_dict : Provided path for state_dict.
            If you want to load checkpoint from SDAModel,
            the path_or_dict should be named as
            sda://model_name/model_version/stage.
        map_location: Target device for checkpoint.
        strip_prefix: The prefix to strip, will deprecated in future.
        state_dict_update_func: `state_dict` update function.
        check_hash: Whether to check the file hash.

    Returns:
        state_dict: State dict for checkpoint.
    """

    if strip_prefix and not state_dict_update_func:
        warnings.warn(
            "`strip_prefix` will deprecated in future, "
            "use `state_dict_update_func` instead",
            DeprecationWarning,
        )

    if isinstance(path_or_dict, dict):
        checkpoint = path_or_dict
    else:
        path = str(path_or_dict)
        if path.startswith("http://"):
            checkpoint = torch.hub.load_state_dict_from_url(
                str(path), map_location=map_location, check_hash=check_hash
            )
        elif path.startswith("sda://"):
            client = get_sda_client()
            _, _, model_name, model_version, stage = path.split(os.sep)
            with TemporaryDirectory(
                "w", dir=os.path.abspath(".")
            ) as output_dir:
                path = client.model.download(
                    output_dir, model_name, model_version, stage
                )
                path = get_hash_file_if_hashed_and_local(
                    path, check_hash=check_hash
                )
                fs = get_filesystem(path)
                with fs.open(path, "rb") as f:
                    checkpoint = torch.load(f, map_location=map_location)
        else:
            if path.startswith("dmpv2://"):
                bkt_clt = BucketClient()
                path = bkt_clt.url_to_local(str(path))
            path = get_hash_file_if_hashed_and_local(
                path, check_hash=check_hash
            )
            fs = get_filesystem(path)
            with fs.open(path, "rb") as f:
                checkpoint = torch.load(f, map_location=map_location)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # TODO(min.du): deprecated, remove in v1.0 #
    if strip_prefix and list(state_dict.keys())[0].startswith(strip_prefix):
        prefix_len = len(strip_prefix)
        state_dict = {k[prefix_len:]: v for k, v in state_dict.items()}

    if state_dict_update_func:
        assert callable(state_dict_update_func), (
            f"{state_dict_update_func} " "is not callable."
        )
        state_dict = state_dict_update_func(state_dict)

    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = state_dict
    else:
        checkpoint = state_dict

    return checkpoint


def update_state_dict_by_strip_prefix(
    state_dict: Dict, strip_prefix: str = "module."
) -> Dict:
    """
    Strip prefix in state dict keys, used as default state_dict_update_func.

    Args:
        state_dict: Model state dict.
        strip_prefix: The prefix to strip.

    Return:
        state_dict: Processed state dict.

    """
    if list(state_dict.keys())[0].startswith(strip_prefix):
        prefix_len = len(strip_prefix)
        state_dict = {k[prefix_len:]: v for k, v in state_dict.items()}
    else:
        logger.warning(
            "{} is not at the beginning of state dict".format(strip_prefix)
        )
    return state_dict


def load_state_dict(
    model: nn.Module,
    path_or_dict: Union[dict, str, Path],
    map_location: Optional[str] = None,
    strip_prefix: str = "module.",
    state_dict_update_func: Optional[Callable] = None,
    check_hash: bool = True,
    allow_miss: bool = False,
    ignore_extra: bool = False,
    verbose: bool = False,
) -> nn.Module:
    """
    Load state_dict from file to model.

    Args:
        model: Model for loading checkpoint.
        path_or_dict : Path of checkpoint or state_dict.
        map_location: Target device for checkpoint.
        strip_prefix: The prefix to strip.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        verbose: Show unexpect_key and miss_key info.
    Returns:
        model: Model with pretrained checkpoint.
    """

    if state_dict_update_func is None:
        state_dict_update_fn = functools.partial(
            update_state_dict_by_strip_prefix, strip_prefix=strip_prefix
        )
    else:
        state_dict_update_fn = state_dict_update_func
    checkpoint = load_checkpoint(
        path_or_dict,
        map_location,
        state_dict_update_func=state_dict_update_fn,
        check_hash=check_hash,
    )
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    miss_key, unexpect_key = model.load_state_dict(state_dict, strict=False)

    logger.info("state_dict in checkpoint num: {}".format(len(state_dict)))
    logger.info("state_dict in model num: {}".format(len(model.state_dict())))
    logger.warning("miss_key num: {}".format(len(miss_key)))
    if verbose:
        logger.warning("miss_key: {}".format(" ".join(miss_key)))
    logger.warning("unexpect_key num: {}".format(len(unexpect_key)))
    if verbose:
        logger.warning("unexpect_key: {}".format(" ".join(unexpect_key)))

    if len(miss_key) > 0 and not allow_miss:
        raise ValueError("set allow_miss=True to skip this check")
    if len(unexpect_key) > 0 and not ignore_extra:
        raise ValueError("set ignore_extra=True to skip this check")
    return model
