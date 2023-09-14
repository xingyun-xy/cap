# Copyright (c) Changan Auto. All rights reserved.
import logging
import re
from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import flatten, is_list_of_type
from cap.utils.model_helpers import (
    fuse_norm_recursively,
    get_binding_module,
    has_normalization,
)
from .callbacks import CallbackMixin

__all__ = ["FreezeModule", "FuseBN"]

logger = logging.getLogger(__name__)


class OnlineModelTrick(CallbackMixin, ABC):
    """Base class for dynamic model trick."""

    def __init__(
        self,
        modules: List[List[str]],
        step_or_epoch: List[int],
        update_by: str,
        strict_match: bool = True,
    ):
        self.modules = modules
        self.step_or_epoch = step_or_epoch
        assert is_list_of_type(modules, list), f"{modules} is not list of list"
        assert is_list_of_type(
            step_or_epoch, int
        ), f"{step_or_epoch} is not list of int"
        assert len(modules) == len(step_or_epoch)

        self.update_by = update_by
        assert update_by in ("step", "epoch")

        self.strict_match = strict_match

    def on_loop_begin(self, model, **kwargs):
        """Check module name."""
        # compatible with DDP/DP
        _model = get_binding_module(model)

        if self.strict_match:
            # assert: all names in self.modules should be sub-module of model
            names, _ = flatten(self.modules)
            for name in names:
                assert hasattr(
                    _model, name
                ), f"{name} not found in model, please recheck"

    def on_epoch_begin(self, model, epoch_id, **kwargs):
        """Do process on epoch begin if need."""
        if self.update_by != "epoch":
            return
        if epoch_id not in self.step_or_epoch:
            return

        # compatible with DDP/DP
        _model = get_binding_module(model)

        # get current module index
        index = self.step_or_epoch.index(epoch_id)

        # do process for each module
        for module_name in self.modules[index]:
            self.process(_model, module_name)

    def on_step_begin(self, model, global_step_id, **kwargs):
        """Do process on step begin if need."""
        if self.update_by != "step":
            return
        if global_step_id not in self.step_or_epoch:
            return

        # compatible with DDP/DP
        _model = get_binding_module(model)

        # get current module index
        index = self.step_or_epoch.index(global_step_id)

        # do process for each module
        for module_name in self.modules[index]:
            self.process(_model, module_name)

    @abstractmethod
    def process(self, model: nn.Module, name: str) -> None:
        """Process sub_module in model given name."""
        pass


@OBJECT_REGISTRY.register
class FreezeModule(OnlineModelTrick):
    """
    Freeze module parameter while training. Useful in finetune case.

    Args:
        module: sub model names.
        step_or_epoch: when to freeze module, same length as module.
        update_by: by step or by epoch.
        only_batchnorm: Only freeze batchnorm, with valid gradient.
            Default is False.

    Example:
        >>> freeze_module_callback = FreezeModule(
        ...    modules=[['backbone'], ['neck']],
        ...    step_or_epoch=[10000, 15000],
        ...    update_by='step',
        ...    only_batchnorm=True,
        ... )
    """

    def __init__(
        self,
        modules: List[List[str]],
        step_or_epoch: List[int],
        update_by: str,
        strict_match: bool = True,
        only_batchnorm: bool = False,
    ):
        modules = [[re.compile(_m) for _m in m] for m in modules]
        super().__init__(
            modules,
            step_or_epoch,
            update_by,
            strict_match=strict_match,
        )
        self.only_batchnorm = only_batchnorm

    def on_loop_begin(self, model, **kwargs):
        pass

    def process(self, model: nn.Module, pattern):
        """Freeze module inplace."""
        names = []
        for name, _ in model.named_children():
            match_flag = pattern.match(name)
            if match_flag:
                names.append(name)
        if not names:
            if self.strict_match:
                raise AttributeError(pattern)
            else:
                logger.warning(f"Skip pattern {pattern}")
                return

        # try:
        #     m = getattr(model, name)
        # except AttributeError as e:
        #     if self.strict_match:
        #         raise e
        #     else:
        #         logger.warning(e)
        #         logger.warning(f"Skip {name}")
        #         return

        # # set batchnorm and dropout in eval mode
        # m.eval()
        for name in names:
            m = getattr(model, name)
            m.eval()
            if self.only_batchnorm:
                logger.info(f"[FreezeModule] freeze bn in {name}.")
            else:
                # disable grad
                logger.info(f"[FreezeModule] freeze {name} to disable grad.")
                for param in m.parameters():
                    param.requires_grad = False


@OBJECT_REGISTRY.register
class FuseBN(OnlineModelTrick):
    """
    Fuse batchnorm layer in float training.

    Usually batchnorm is fused in QAT, but sometimes you can do it
    float training.

    Args:
        module: sub model names to fuse bn.
        step_or_epoch: when to fusebn, same length as module.
        update_by: by step or by epoch.

    Note:
        Only Conv+BN inside nn.Sequential or nn.ModuleList can be merged.

    Example:
        >>> fuse_bn_callback = FuseBN(
        ...    modules=[['backbone'], ['neck']],
        ...    step_or_epoch=[10000, 15000],
        ...    update_by='step',
        ... )
    """

    def process(self, model, name):
        """Fuse bn inplace."""
        logger.info(f"[FuseBN] fuse bn in {name}")
        node = getattr(model, name)
        node_fused = fuse_norm_recursively(node, fuse_list=["bn"])
        assert not has_normalization(node_fused, check_list=["bn"])
        setattr(model, name, node_fused)
