import inspect
import sys
from collections import OrderedDict
from typing import Any, Dict

import torch

from .quantization_types import Pattern

# TODO(future PR): fix the typing on QuantizeHandler
# (currently a circular dependency)
QuantizeHandler = Any

# pattern for op fusion
DEFAULT_FUSION_PATTERNS = OrderedDict()


# Example use of register pattern function:
# @register_fusion_pattern(torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))  # noqa
# class ConvBNReLUFusion():
#     def __init__(...):
#         ...
#
# Note: The order of patterns is important! match function will take whatever
# is matched first, so we'll need to put the fusion patterns before
# single patterns. For example, add_relu should be registered come before relu.
# decorators are applied in the reverse order we see. Also when we match the
# nodes in the graph with these patterns, we'll start from the last node of
# the graph and traverse back.


def register_fusion_pattern(pattern):
    def insert(fn):
        DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn

    return insert


def get_default_fusion_patterns() -> Dict[Pattern, QuantizeHandler]:
    return DEFAULT_FUSION_PATTERNS


DEFAULT_QUANTIZATION_PATTERNS = OrderedDict()
# a map from pattern to activation_post_process(observer/fake_quant)
# constructor for output activation
# e.g. pattern: torch.sigmoid,
#      output_activation_post_process: default_affine_fixed_qparam_fake_quant
DEFAULT_OUTPUT_ACTIVATION_POST_PROCESS_MAP = dict()


# Register pattern for both static quantization and qat
def register_quant_pattern(pattern, output_activation_post_process=None):
    def insert(fn):
        DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if output_activation_post_process is not None:
            DEFAULT_OUTPUT_ACTIVATION_POST_PROCESS_MAP[
                pattern
            ] = output_activation_post_process
        return fn

    return insert


# Get patterns for both static quantization and qat
def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]:
    return DEFAULT_QUANTIZATION_PATTERNS


# a map from pattern to output activation post process constructor
# e.g. torch.sigmoid -> default_affine_fixed_qparam_fake_quant
def get_default_output_activation_post_process_map() -> Dict[
    Pattern, torch.quantization.observer.ObserverBase
]:
    return DEFAULT_OUTPUT_ACTIVATION_POST_PROCESS_MAP


class MatchAllNode:
    """A node pattern that matches all nodes,
    used to indicate any input to the pattern.
    The node matched with this will not be fused"""

    pass


def is_match(modules, node, pattern, max_uses=sys.maxsize):
    """Matches a node in fx against a pattern"""
    if isinstance(pattern, tuple):
        # (Callable, arg0, arg1, ...)
        self_match, *arg_matches = pattern
        if self_match is getattr:
            assert (
                len(pattern) <= 2
            ), "Expect getattr pattern to have at most 2 elements"
    else:
        self_match = pattern
        arg_matches = []

    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):
        return True

    if len(node.users) > max_uses:
        # avoid intermediate result used by other nodes
        return False

    if isinstance(self_match, type) and issubclass(
        self_match, torch.nn.Module
    ):
        # match nn.Module
        if (
            node.op != "call_module"
            or not type(modules[node.target]) == self_match
        ):
            return False

    elif self_match is getattr:
        if node.op != "get_attr":
            return False

        if arg_matches:
            # march the target rather than args
            assert len(arg_matches) == 1, "getattr only allowed to have 1 args"
            if isinstance(arg_matches[0], str):
                return arg_matches[0] == node.target
            elif isinstance(arg_matches[0], type):
                return arg_matches[0] == type(modules[node.target])  # noqa
            else:
                raise ValueError("arg of getattr must be str or type")

    elif callable(self_match):
        # try to get the owner for class method
        owner = None
        if inspect.ismethod(self_match):
            for cls in inspect.getmro(self_match.__self__.__class__):
                if self_match.__name__ in cls.__dict__:
                    owner = cls
        else:
            owner = getattr(
                inspect.getmodule(self_match),
                self_match.__qualname__.split(".<locals>", 1)[0].rsplit(
                    ".", 1
                )[0],
                None,
            )
        if owner is None or owner is self_match:
            # for plain function
            if node.op != "call_function" or node.target is not self_match:
                return False
        elif inspect.isclass(owner):
            # for class method
            if node.op != "call_method" or node.target != self_match.__name__:
                return False
            getattr_node = node.args[0]
            if not (
                getattr_node.op == "get_attr"
                and getattr_node.target in modules
                and type(modules[getattr_node.target]) == owner
            ):
                return False
            return all(
                is_match(modules, node, arg_match, max_uses=1)
                for node, arg_match in zip(node.args[1:], arg_matches)
            )
        else:
            if node.op != "call_function" or node.target is not self_match:
                return False
            elif node.target is getattr:
                if node.args[1] != pattern[1]:
                    return False
            else:
                # return False for unknown conditions
                return False

    elif isinstance(self_match, str):
        # match call_method on the output of node (include get_attr node)
        if node.op != "call_method" or node.target != self_match:
            return False

    elif node.target != self_match:
        # TODO: do not know when use this branch
        return False

    else:
        # return False for unknown conditions
        return False

    if not arg_matches:
        # end of matching
        return True

    if len(arg_matches) != len(node.args):
        return False

    return all(
        is_match(modules, node, arg_match, max_uses=1)
        for node, arg_match in zip(node.args, arg_matches)
    )
