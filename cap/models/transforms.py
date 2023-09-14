import logging
import re
from typing import Callable, List

import torch.nn as nn

logger = logging.getLogger(__name__)


def gen_children_by_name(
    model: nn.Module, names: List[str], strict: bool = False
):
    names = set(names)
    if strict:
        for n in names:
            assert hasattr(model, n), f"{n} should be a submodule of model"

    for n, m in model.named_children():
        if n in names:
            yield n, m


def gen_children_by_regex(
    model: nn.Module, patterns: List[str], strict: bool = False
):
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
        logger.info(f"Fuse BN in {a}.")
        yield a, getattr(model, a)


def transform_by_pattern(
    model: nn.Module,
    transform: Callable,
    patterns: List[str],
    regex: bool = False,
    strict: bool = False,
):
    if regex:
        gen = gen_children_by_regex
    else:
        gen = gen_children_by_name

    for n, m in gen(model, patterns, strict=strict):
        setattr(model, n, transform(m))

    return model


def freeze_bn(model: nn.Module):
    return model.eval()


def fuse_bn(model: nn.Module):
    if hasattr(model, "fuse_norm"):
        return model.fuse_norm()
    else:
        names = []
        for n, m in model.named_children():
            names.append(n)
            setattr(model, n, fuse_bn(m))
        return model


def qat_fuse_bn_by_patterns(
    model: nn.Module,
    patterns: List[str],
    regex: bool = False,
    strict: bool = False,
):
    return transform_by_pattern(
        model, fuse_bn, patterns, regex=regex, strict=strict
    )
