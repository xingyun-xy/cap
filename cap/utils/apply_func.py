# Copyright (c) Changan Auto. All rights reserved.
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
from inspect import signature
from typing import Any, Callable, Optional, Tuple, Union

import changan_plugin_pytorch
import numpy as np
import torch
from six.moves import map, zip
from torch._six import string_classes

__all__ = [
    "_as_list",
    "_as_numpy",
    "convert_numpy",
    "_get_keys_from_dict",
    "_is_increasing_sequence",
    "multi_apply",
    "flatten",
    "regroup",
    "apply_to_collection",
    "to_flat_ordered_dict",
    "is_list_of_type",
    "to_cuda",
    "is_namedtuple",
    "check_type",
]

# TODO(min.du, 0.1): remove prefix _ in interface name #


def _as_list(obj: Any) -> Sequence:
    """Convert the argument to a list if it is not already."""

    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


def _as_numpy(a):
    """Convert a (list of) numpy into numpy.ndarray.

    # TODO(min.du, 0.1): need refactor #

    """
    if isinstance(a, (list, tuple)):
        out = list(a)
        try:
            out = np.concatenate(out, axis=0)
        except ValueError:
            out = np.array(out)
        return out
    return a


def convert_numpy(
    data: Any,
    to_list: bool = False,
    dtype: Optional[str] = None,
) -> Any:
    r"""Convert each Tensor array data field into a numpy, recursively."""
    elem_type = type(data)
    if (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if dtype:
            data = data.astype(dtype)
        data = data.tolist() if to_list else data
        return data
    elif isinstance(data, torch.Tensor):
        if isinstance(data, changan_plugin_pytorch.qtensor.QTensor):
            data = data.as_subclass(torch.Tensor)
        data = data.detach().cpu().numpy()
        if dtype:
            data = data.astype(dtype)
        if to_list:
            data = data.tolist()
        return data
    elif isinstance(data, Mapping):
        return {
            key: convert_numpy(data[key], to_list=to_list, dtype=dtype)
            for key in data
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(
            *(convert_numpy(d, to_list=to_list, dtype=dtype) for d in data)
        )
    elif isinstance(data, Sequence) and not isinstance(data, string_classes):
        return [convert_numpy(d, to_list=to_list, dtype=dtype) for d in data]
    else:
        return data


def _get_keys_from_dict(target_dict, field):
    """
    Get keys from dict recrusively.

    # TODO(min.du, 0.1): need refactor #

    """
    field_found = []
    for k, v in target_dict.items():
        if k == field:
            field_found.append(v)
        elif isinstance(v, dict):
            results = _get_keys_from_dict(v, field)
            for result in results:
                field_found.append(result)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    more_results = _get_keys_from_dict(item, field)
                    for another_result in more_results:
                        field_found.append(another_result)
    return field_found


def flatten(obj):
    """Flatten list/tuple/dict object and get layout.

    you can use `regroup` to get original object.

    Args:
        obj (Union[list, tuple, dict]): Object.

    Returns:
        flat (tuple): The flatten object.
        fmts (tuple): Layout of original object.

    Example::

        >>> data = [0, [1, 2]]
        >>> flatten_data, data_layout = flatten(data)
        >>> print(flatten_data)
        (0, 1, 2)
        >>> print(data_layout)
        (<class 'list'>, (<class 'object'>, (<class 'list'>, (<class 'object'>, <class 'object'>))))
        >>> group_data, left_data = regroup(flatten_data, data_layout)
        >>> print(group_data)
        [0, [1, 2]]
        >>> print(left_data)
        ()

    """  # noqa
    if isinstance(obj, dict):
        flat = []
        fmts = [dict, []]
        for key, value in obj.items():
            obj_i, fmt_i = flatten(value)
            flat.extend(obj_i)
            fmts[1].append((key, fmt_i))
        fmts[1] = tuple(fmts[1])
        return tuple(flat), tuple(fmts)
    elif isinstance(obj, (list, tuple)):
        flat = []
        fmts = [type(obj), []]
        for value in obj:
            obj_i, fmt_i = flatten(value)
            flat.extend(obj_i)
            fmts[1].append(fmt_i)
        fmts[1] = tuple(fmts[1])
        return tuple(flat), tuple(fmts)
    else:
        return (obj,), object


def regroup(obj, fmts):
    """Regroup a list/tuple of objects.

    Args:
        obj (Union[list, tuple]): List of flatten objects.
        fmts (Union[list, tuple]): Layout of original objects.

    Returns:
        group_data (Union[list, tuple, dict]): The grouped objects.
        left_data (Union[list, tuple]): The ungrouped objects.
    """
    if fmts is object:
        return obj[0], obj[1:] if len(obj) > 1 else ()
    assert isinstance(fmts, (list, tuple))
    obj_type = fmts[0]
    if obj_type is dict:
        ret = {}
        for key, fmt_i in fmts[1]:
            ret[key], obj = regroup(obj, fmt_i)
        return ret, obj
    elif obj_type in (list, tuple):
        ret = []
        for fmt_i in fmts[1]:
            res, obj = regroup(obj, fmt_i)
            ret.append(res)
        if obj_type is tuple:
            ret = tuple(ret)
        return ret, obj
    else:
        raise TypeError(f"Unknown type: {obj_type}")


def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs,
) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.

    Migrated from pytorch_lightning.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of
            ``function``)
        wrong_dtype: the given function won't be applied if this type is
            specified and the given collections is of the :attr:`wrong_type`
            even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to calls of
            ``function``)

    Returns:
        the resulting collection
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (
        wrong_dtype is None or not isinstance(data, wrong_dtype)
    ):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type(
            {
                k: apply_to_collection(v, dtype, function, *args, **kwargs)
                for k, v in data.items()
            }
        )

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(
            *(
                apply_to_collection(d, dtype, function, *args, **kwargs)
                for d in data
            )
        )

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type(
            [
                apply_to_collection(d, dtype, function, *args, **kwargs)
                for d in data
            ]
        )

    # data is neither of dtype, nor a collection
    return data


def is_namedtuple(obj):  # noqa: D205,D400
    """Check whether `obj` is instance of tuple subclass which was created from
    collections.namedtuple.
    """
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_fields")
        and hasattr(obj, "_asdict")
    )


def to_flat_ordered_dict(
    obj: Any,
    key_prefix: Optional[str] = "",
    flat_condition: Optional[Callable[[Any], bool]] = None,
):  # noqa: D205,D400
    """Flatten a dict/list/tuple object into an `OrderedDict` object,
    the key of which is automatically generated.

    Args:
        obj: Object to be flattened.
        key_prefix: Prefix of keys of result dict.
        flat_condition: Function with (`key`, `values`) as input,
            return `True/False` means whether flat this `values` or not.

    Examples::

        >>> obj = dict(
        ...     a=[dict(c=1)],
        ...     d=(2, 3)
        ... )

        >>> to_flat_ordered_dict(obj, key_prefix='test')
        OrderedDict([('test_a_0_c', 1), ('test_d_0', 2), ('test_d_1', 3)])

        >>> to_flat_ordered_dict(obj, key_prefix='test',
        ...     flat_condition=lambda k, v: not isinstance(v, tuple))
        OrderedDict([('test_a_0_c', 1), ('test_d', (2, 3))])

    """
    assert isinstance(key_prefix, str), type(key_prefix)
    if flat_condition is not None:
        assert callable(flat_condition)

    def _append(x):
        return "%s_%s" % (key_prefix, x) if key_prefix != "" else x

    def _flat():
        if isinstance(obj, dict):
            for k, v in obj.items():
                name2val.update(
                    to_flat_ordered_dict(v, _append(k), flat_condition)
                )
        elif is_namedtuple(obj):
            for k, v in zip(obj._fields, obj):
                name2val.update(
                    to_flat_ordered_dict(v, _append(k), flat_condition)
                )
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                name2val.update(
                    to_flat_ordered_dict(v, _append(str(i)), flat_condition)
                )
        else:
            name2val[key_prefix] = obj

    assert isinstance(key_prefix, str), type(key_prefix)

    if flat_condition is not None:
        # check flat_condition lambda input args
        assert callable(flat_condition)
        formal_args = sorted(signature(flat_condition).parameters)
        assert len(formal_args) == 2, (
            "flat_condition input should be "
            f"(key, value), found {formal_args}"
        )

    name2val = OrderedDict()
    if flat_condition is None:
        _flat()
    elif flat_condition(key_prefix, obj):
        _flat()
    else:
        name2val[key_prefix] = obj

    return name2val


def _is_increasing_sequence(obj, strict: bool = True) -> bool:
    """Return whether an given sequence is increasing order.

    Args:
        obj: list/tuple of comparable, Sequence to be checked.
        strict: whether allow equal or not.

    Returns:
        flag: True means yes.

    # TODO(min.du, 0.1): input type requiring check #

    """
    obj = _as_list(obj)
    pre = obj[0]
    for x in obj[1:]:
        if strict:
            if x <= pre:
                return False
        elif x < pre:
            return False
        pre = x
    return True


def multi_apply(func: Callable, *args, **kwargs) -> Tuple:
    """Use func on different objects and merge public attributes.

    Args:
        func: Function handle
        args: Args of all objects.
        kwargs: Shared on different objects.

    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, *args))

    if isinstance(map_results[0], tuple):
        map_results = map(list, zip(*map_results))
    return tuple(map_results)


def is_list_sorted(lst: object):
    return isinstance(lst, (list, tuple)) and all(
        [lst[i] < lst[i + 1] for i in range(len(lst) - 1)]
    )


def is_list_of_type(lst: object, element_type):  # noqa: D205,D400
    """Check whether `lst` is a list/tuple, as well as it's elements are
    instances of `element_type`.

    Args:
        lst: Object to be check.
        element_type: Target element type.

    Returns:
        Return True if `lst` is a list/tuple of 'element_type', else return
        False.
    """
    return isinstance(lst, (list, tuple)) and all(
        isinstance(elem, element_type) for elem in lst
    )


def to_cuda(
    obj: Any,
    device: torch.device = None,
    non_blocking: Optional[bool] = False,
    memory_format: Optional[torch.memory_format] = torch.preserve_format,
    inplace: Optional[bool] = False,
) -> Any:
    """
    Move object to cuda.

    Args:
        obj (Any): Any data type containing tensor, such as tensor container,
            list tuple and dict, and also optimizer.
        device (:class:`torch.device`): The destination GPU device.
            Defaults to the current CUDA device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired
            memory format of returned Tensor.
            Default: ``torch.preserve_format``.
        inplace (bool, optional): If `obj` is optimizer, inplace must be True.

    """
    flats, fmt = flatten(obj)
    flats = list(flats)
    for i in range(len(flats)):
        if isinstance(flats[i], torch.Tensor):
            if inplace:
                raise NotImplementedError
            flats[i] = flats[i].cuda(
                device, non_blocking, memory_format=memory_format
            )
        elif isinstance(flats[i], torch.optim.Optimizer):
            assert inplace, (
                "Please set inplace=True when apply " "`to_cuda` on optimizer"
            )
            optimizer = flats[i]
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(
                            device, non_blocking, memory_format=memory_format
                        )
        else:
            pass
    obj, left = regroup(tuple(flats), fmt)
    assert len(left) == 0, len(left)
    return obj


def check_type(obj: Any, allow_types: Union[Any, Tuple[Any]]) -> None:
    """Check whether the type of input obj meets the expectation.

    Args:
        obj: Input object.
        allow_types: Expected type.
    """
    if not isinstance(obj, allow_types):
        raise TypeError(f"Expected {allow_types}, but get {type(obj)}")
