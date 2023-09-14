r""" Extended tracer and wrap of torch.fx
This file defines a inherit tracer of torch.fx.Tracer and a extended wrap to
allow wrapping of user-defined Module or method, which help users do some
optimization of their own module by torch.fx
"""
import inspect
from distutils.version import LooseVersion
from inspect import isclass, isfunction, ismethod
from types import FunctionType
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import fx
from torch.fx.node import Argument

__all__ = ["wrap", "CustomTracer", "get_supported_method"]

if LooseVersion(torch.__version__) >= LooseVersion("1.10"):
    from torch.fx._symbolic_trace import (
        _wrapped_fns_to_patch,
        _wrapped_methods_to_patch,
    )
else:
    from torch.fx.symbolic_trace import (
        _wrapped_fns_to_patch,
        _wrapped_methods_to_patch,
    )
# A global list of custom-defined modules to be traced as leaf module
_wrapped_modules_to_patch = []


def wrap(
    cls_or_method: Union[FunctionType, str, type],
    method_list: Optional[List[str]] = None,
):
    """
    This function can be:
    1) called or used as a decorator on a string to:
        register a builtin function as a "leaf function"
    2) called or used as a decorator on a function to:
        register this function as a "leaf function"
    3) called or used as a decorator on subclass of torch.nn.Module to:
        register this module as a "leaf module", and
        register user defined method in this class as "leaf method"

    Args:
        method_list (Optional[List[str]]): Specify the method names to
            register. Pass None to automatically register all methods
            in __dict__. Default: None
    """

    if isinstance(cls_or_method, str):
        # wrap("sum")
        fn_name = cls_or_method
        currentframe = inspect.currentframe()
        assert currentframe is not None
        f = currentframe.f_back
        _wrapped_fns_to_patch.append((f.f_globals, fn_name))

    elif isfunction(cls_or_method):
        owner = None
        # ismethod only recognize the method of instance, for example
        # class A:
        #     def func(self):
        #         pass
        # a = A()
        # ismethod(a.func) == True
        # ismethod(A.func) == False
        if ismethod(cls_or_method):
            for cls in inspect.getmro(cls_or_method.__self__.__class__):
                if cls_or_method.__name__ in cls.__dict__:
                    owner = cls
        else:
            owner = getattr(
                inspect.getmodule(cls_or_method),
                cls_or_method.__qualname__.split(".<locals>", 1)[0].rsplit(
                    ".", 1
                )[0],
                None,
            )

        if isclass(owner):
            # class CLASS():
            #     def method(self):
            #         pass
            #
            # wrap(CLASS.method)
            map = (owner, cls_or_method.__name__)
            if map not in _wrapped_methods_to_patch:
                _wrapped_methods_to_patch.append(map)
        else:
            # def func():
            #     pass
            #
            # wrap(func)
            fn_name = cls_or_method.__code__.co_name
            currentframe = inspect.currentframe()
            assert currentframe is not None
            f = currentframe.f_back
            _wrapped_fns_to_patch.append((f.f_globals, fn_name))

    elif isclass(cls_or_method):
        assert issubclass(cls_or_method, torch.nn.Module)
        if method_list is None:
            # wrap all methods by default
            method_names = set(cls_or_method.__dict__.keys())
            # wrap call_module by default
            method_names -= set(torch.nn.Module.__dict__.keys())
            if cls_or_method not in _wrapped_modules_to_patch:
                _wrapped_modules_to_patch.append(cls_or_method)
        else:
            method_names = method_list

        for method_name in method_names:
            maybe_method = getattr(cls_or_method, method_name)
            if isfunction(maybe_method):
                wrapped_method = (cls_or_method, method_name)
                if wrapped_method not in _wrapped_methods_to_patch:
                    _wrapped_methods_to_patch.append(wrapped_method)
    else:
        raise RuntimeError("Wrap arg must be a str or function or class")

    return cls_or_method


def get_supported_method():
    """
    Get a mapping from a class to its registered method names
    """
    ret = {}
    for cls, method_name in _wrapped_methods_to_patch:
        if cls in ret:
            ret[cls].append(method_name)
        else:
            ret[cls] = [method_name]
    return ret


class CustomTracer(fx.Tracer):
    def __init__(self):
        super().__init__()

    def is_leaf_module(
        self, m: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        return isinstance(
            m, tuple(_wrapped_modules_to_patch)
        ) or super().is_leaf_module(m, module_qualified_name)

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, np.number):
            return a
        return super().create_arg(a)
