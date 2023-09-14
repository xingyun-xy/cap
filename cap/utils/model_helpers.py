# Copyright (c) Changan Auto. All rights reserved.
import copy
import logging
import re
from typing import List

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from cap.models.base_modules import GroupNorm2d

__all__ = [
    "has_normalization",
    "get_binding_module",
    "fuse_norm_in_list",
    "fuse_norm_recursively",
]

logger = logging.getLogger(__name__)


def has_normalization(model, check_list=None):
    """
    Check normalization layer exists in model.

    Args:
        model: Source model
        check_list: A str list to setting fusion normalization
        for example: ['bn', 'gn']
    Returns:
        A bool flag of checking
    """
    assert check_list is not None
    model = get_binding_module(model)
    for _name, m in model.named_modules():
        if isinstance(m, _BatchNorm) and "bn" in check_list:
            return True
        if isinstance(m, GroupNorm2d) and "gn" in check_list:
            return True
    return False


def get_binding_module(model):
    """Get binding module inside model if model is wrapped by DP or DDP."""
    if isinstance(
        model,
        (
            torch.nn.parallel.DataParallel,
            torch.nn.parallel.DistributedDataParallel,
        ),
    ):
        return model.module
    else:
        return model


def fuse_norm_in_list(block: nn.Module, fuse_list: list):
    """
    Take a sequential block and fuses the normalization with convolution.

    Migrated from https://github.com/MIPT-Oulu/pytorch_bn_fusion/blob/master/bn_fusion.py   # noqa

    Args:
        block: Source model
        fuse_list: A str list.
        for example: ['bn', 'gn']
    Returns:
        Converted block
    """
    assert fuse_list is not None
    if not isinstance(block, (nn.Sequential, nn.ModuleList)):
        return block
    if len(block) == 0:
        return block
    stack = []

    for m in block.children():
        if not isinstance(m, (GroupNorm2d, _BatchNorm)):
            stack.append(m)
            continue
        if isinstance(stack[-1], nn.Conv2d):
            if isinstance(m, _BatchNorm) and "bn" in fuse_list:
                logger.info("fusing conv + bn ...")
            elif isinstance(m, GroupNorm2d) and "gn" in fuse_list:
                logger.info("fusing conv + gn ...")
            else:
                continue

            fusedconv = copy.deepcopy(stack[-1])

            norm_st_dict = m.state_dict()
            conv_st_dict = stack[-1].state_dict()

            # BatchNorm params
            eps = m.eps
            mu = norm_st_dict["running_mean"]
            var = norm_st_dict["running_var"]
            gamma = norm_st_dict["weight"]

            if "bias" in norm_st_dict:
                beta = norm_st_dict["bias"]
            else:
                beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

            # Conv params
            W = conv_st_dict["weight"]
            if "bias" in conv_st_dict:
                bias = conv_st_dict["bias"]
            else:
                bias = torch.zeros(W.size(0)).float().to(gamma.device)

            W, bias = torch.nn.utils.fuse_conv_bn_weights(
                W, bias, mu, var, eps, gamma, beta
            )
            """
            # Below is origin code from
            # https://github.com/MIPT-Oulu/pytorch_bn_fusion
            # We keep it here just for reference.

            denom = torch.sqrt(var + eps)
            b = beta - gamma.mul(mu).div(denom)
            A = gamma.div(denom)
            bias *= A
            A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

            W.mul_(A)
            bias.add_(b)

            stack[-1].weight.data.copy_(W)
            if stack[-1].bias is None:
                stack[-1].bias = torch.nn.Parameter(bias)
            else:
                stack[-1].bias.data.copy_(bias)
            """
            fusedconv.weight = W
            fusedconv.bias = bias
            stack[-1] = fusedconv

    if isinstance(block, nn.Sequential):
        return nn.Sequential(*stack)
    elif isinstance(block, nn.ModuleList):
        return nn.ModuleList(stack)
    else:
        raise ValueError("You should't be here.")


def fuse_norm_recursively(model: nn.Module, fuse_list: list) -> nn.Module:
    """
    Fuse Normalization in model recursively.

    Only conv+bn and conv+gn in sequential or modulelist supported.

    Args:
        model: Source model
        fuse_list: A str list to setting fusion normalization
        for example: ['bn', 'gn']
    Returns:
        Converted model

    """
    for module_name in model._modules:
        model._modules[module_name] = fuse_norm_in_list(
            model._modules[module_name], fuse_list=fuse_list
        )
        if len(model._modules[module_name]._modules) > 0:
            fuse_norm_recursively(
                model._modules[module_name], fuse_list=fuse_list
            )
    return model


def match_children_modules_by_name(
    model: nn.Module, names: List[str], strict: bool = False
):
    """Match children modules by name.

    This function returns the matched modules as an iterator.

    Args:
        model: torch.nn.Module
        names: the list of name of the modules hope to match.
    Yield:
        n: the modules name that matched.
        m: the modules that matched.
    """
    names = set(names)
    if strict:
        for n in names:
            assert hasattr(model, n), f"{n} should be a submodule of model"

    for n, m in model.named_children():
        if n in names:
            yield n, m


def match_children_modules_by_regex(
    model: nn.Module, patterns: List[str], strict: bool = False
):
    """Match children modules by regex.

    This function returns the matched modules as an iterator.

    Args:
        model: torch.nn.Module
        patterns: the list of regex patterns of the modules hope to match.
    Yield:
        a: the modules name that matched.
        m: the modules that matched.
    """
    patterns = [re.compile(p) for p in patterns]

    attrs = set()
    for p in patterns:
        matched = []
        for n, _ in model.named_children():
            if p.match(n):
                matched.append(n)
                attrs.add(n)
        if strict:
            assert matched, f"pattern {p} fails to match"

    for a in attrs:
        logger.info(f"Matching module in {a}.")
        yield a, getattr(model, a)
