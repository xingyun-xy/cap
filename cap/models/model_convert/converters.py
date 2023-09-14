# Copyright (c) Changan Auto. All rights reserved.

import logging
from abc import abstractmethod
from typing import Callable, List, Optional

import changan_plugin_pytorch as changan
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY
from cap.utils import qconfig_manager
from cap.utils.checkpoint import load_checkpoint, load_state_dict
from cap.utils.global_var import set_value
from cap.utils.logger import MSGColor, format_msg
from cap.utils.model_helpers import (
    match_children_modules_by_name,
    match_children_modules_by_regex,
)

__all__ = [
    "Float2QAT",
    "Float2Calibration",
    "Calibration2QAT",
    "QATFusePartBN",
    "QAT2Quantize",
    "LoadCheckpoint",
]

logger = logging.getLogger(__name__)


class BaseConverter(object):
    """Base class for defining the process of model convert."""

    @abstractmethod
    def __call__(self, model):
        raise NotImplementedError


@OBJECT_REGISTRY.register
class Float2QAT(BaseConverter):
    """Define the process of convert float model to qat model."""

    def __init__(self):
        super(Float2QAT, self).__init__()

    def __call__(self, model):
        # make sure the input model is a float model
        model.fuse_model()
        model.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(model, "set_qconfig"):
            model.set_qconfig()
        else:
            raise RuntimeError("`model` should implement `set_qconfig()`")
        changan.quantization.prepare_qat(model, inplace=True)
        logger.info(
            format_msg(
                "Successfully convert float model to qat model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class QATFusePartBN(BaseConverter):
    """Define the process of fusing bn in a QAT model.

    Usually used in step fuse bn. Note that module do fuse bn only when
    block implement block."fuse_method"().

    Args:
        qat_fuse_patterns: Regex, compile by re.
        fuse_method: Fuse bn method that block calls.
        regex: Whether to match by regex. if not, match by module name.
        strict: Whether the regular expression is required to be all matched.
    """

    def __init__(
        self,
        qat_fuse_patterns: List[str],
        fuse_method: str = "fuse_norm",
        regex: bool = True,
        strict: bool = False,
    ):
        self.qat_fuse_patterns = qat_fuse_patterns
        self.fuse_method = fuse_method
        self.regex = regex
        self.strict = strict
        super(QATFusePartBN, self).__init__()

    def _fuse_bn(self, model: nn.Module):
        if hasattr(model, self.fuse_method):
            return getattr(model, self.fuse_method)()
        else:
            names = []
            for n, m in model.named_children():
                names.append(n)
                setattr(model, n, self._fuse_bn(m))
            return model

    @property
    def get_match_method(self):
        if self.regex:
            return match_children_modules_by_regex
        else:
            return match_children_modules_by_name

    def __call__(self, model):
        # check qat mode in with bn.
        assert changan.qat_mode.get_qat_mode() in [
            "with_bn",
            "with_bn_reverse_fold",
        ], (
            f"QATFusePartBN only support in with bn mode."
            f"But get {changan.qat_mode.get_qat_mode()}"
        )

        gen = self.get_match_method
        for n, m in gen(model, self.qat_fuse_patterns, strict=self.strict):
            setattr(model, n, self._fuse_bn(m))

        logger.info(
            format_msg(
                "Successfully qat float model to qat fuse bn model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class Float2Calibration(BaseConverter):
    """Define the process of convert float model to calibration model."""

    def __init__(self):
        super(Float2Calibration, self).__init__()

    def __call__(self, model):
        # make sure the input model is a float model
        model.fuse_model()
        model.qconfig = qconfig_manager.get_default_calibration_qconfig()
        if hasattr(model, "set_calibration_qconfig"):
            model.set_calibration_qconfig()
        changan.quantization.prepare_calibration(model, inplace=True)
        logger.info(
            format_msg(
                "Successfully convert float model to calibration model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class Calibration2QAT(BaseConverter):
    """Define the process of convert calibration model to qat model."""

    def __init__(self):
        super(Calibration2QAT, self).__init__()

    def __call__(self, model):
        # make sure the input model is a calibration model
        if hasattr(model, "set_qconfig"):
            model.set_qconfig()
        else:
            raise RuntimeError("`model` should implement `set_qconfig()`")
        changan.quantization.prepare_qat(model, inplace=True)
        logger.info(
            format_msg(
                "Successfully convert calibration model to qat model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class QAT2Quantize(BaseConverter):
    """Define the process of convert qat model to quantize model."""

    def __init__(self):
        super(QAT2Quantize, self).__init__()

    def __call__(self, model):
        # make sure the input model is a qat model
        changan.quantization.convert(model.eval(), inplace=True)
        logger.info(
            format_msg(
                "Successfully convert qat model to quantize model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class LoadCheckpoint(BaseConverter):
    """Load the checkpoint from file to model and return the checkpoint.

    LoadCheckpoint usually happens before or after BaseConverter.It means
    the model needs to load parameters before or after BaseConverter.

    Args:
        checkpoint_path: Path of the checkpoint file.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        verbose: Show unexpect_key and miss_key info.
        return_checkpoint: whether return the values of the checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str,
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        verbose: bool = False,
    ):
        super(LoadCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.state_dict_update_func = state_dict_update_func
        self.check_hash = check_hash
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.verbose = verbose

    def __call__(self, model):
        model_checkpoint = load_checkpoint(
            path_or_dict=self.checkpoint_path,
            map_location="cpu",
            state_dict_update_func=self.state_dict_update_func,
            check_hash=self.check_hash,
        )
        set_value("model_checkpoint", model_checkpoint)
        model = load_state_dict(
            model,
            path_or_dict=model_checkpoint,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model
