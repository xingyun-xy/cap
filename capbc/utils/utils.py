import warnings
import json
import time
import os
import random
import datetime
import inspect
from typing import Callable, Tuple, Any
from enum import Enum as _Enum
from copy import deepcopy
import logging
from dataclasses_json import DataClassJsonMixin


__all__ = [
    '_as_list', '_flatten', '_regroup',
    'unset_flag', 'map_aggregate', 'DataClassConfig',
    'wait_until_finish', 'UnsetFlag', 'Enum', '_UnsetFlag',
    'call_and_retry', '_check_type', 'identity', 'multi_apply',
    'now_time', "get_operator_name", "get_enum_value"
]


logger = logging.getLogger(__name__)


def _as_list(obj):
    """A utility function that converts the argument to a list if it is not
    already.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list or tuple, return it. Otherwise, return `[obj]` as a
    single-element list.

    """
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


def _flatten(obj):
    """flatten list/tuple and get layout, you can use `_regroup` to regroup
    the flatten list/tuple.

    Examples
    --------
    >>> data = [0, 1, 2, 3, 4, 5, 6]
    >>> flatten_data, data_layout = _flatten(data)
    >>> print(flatten_data)
    [0, 1, 2, 3, 4, 5, 6]
    >>> print(data_layout)
    [0, 0, 0, 0, 0, 0, 0]
    >>> group_data, left_data = _regroup(flatten_data, data_layout)
    >>> print(group_data)
    [0, 1, 2, 3, 4, 5, 6]
    >>> print(left_data)
    []

    >>> data = [0, 1, [2, 3, 4], 5, 6]
    >>> flatten_data, data_layout = _flatten(data)
    >>> print(flatten_data)
    [0, 1, 2, 3, 4, 5, 6]
    >>> print(data_layout)
    [0, 0, [0, 0, 0], 0, 0]
    >>> group_data, left_data = _regroup(flatten_data, data_layout)
    >>> print(group_data)
    [0, 1, [2, 3, 4], 5, 6]
    >>> print(left_data)
    []

    >>> data = [0, 1, [2, [3], 4], 5, 6]
    >>> flatten_data, data_layout = _flatten(data)
    >>> print(flatten_data)
    [0, 1, 2, 3, 4, 5, 6]
    >>> print(data_layout)
    [0, 0, [0, [0], 0], 0, 0]
    >>> group_data, left_data = _regroup(flatten_data, data_layout)
    >>> print(group_data)
    [0, 1, [2, [3], 4], 5, 6]
    >>> print(left_data)
    []
    >>> group_data, left_data = _regroup(flatten_data, data_layout[:-1])
    >>> print(group_data)
    [0, 1, [2, [3], 4], 5]
    >>> print(left_data)
    [6]

    Parameters
    ----------
    obj : list/tuple of object

    Returns
    -------
    flat: list of object
        The flatten object.
    layout: list/tuple
        Layout of original object.
    """
    if isinstance(obj, dict):
        flat = []
        fmts = [dict, []]
        for key, value in obj.items():
            obj_i, fmt_i = _flatten(value)
            flat.extend(obj_i)
            fmts[1].append((key, fmt_i))
        fmts[1] = tuple(fmts[1])
        return tuple(flat), tuple(fmts)
    elif isinstance(obj, (list, tuple)):
        flat = []
        fmts = [list, []]
        for value in obj:
            obj_i, fmt_i = _flatten(value)
            flat.extend(obj_i)
            fmts[1].append(fmt_i)
        fmts[1] = tuple(fmts[1])
        return tuple(flat), tuple(fmts)
    else:
        return (obj, ), object


def _regroup(obj, fmt):
    """Regroup an flatten list/tuple of objects.

    Parameters
    ----------
    obj : list/tuple of int
        Flatten list/tuple of objects.
    layout : list/tuple
        Layout of original objects.

    Returns
    -------
    group_data: list/tuple of object
        The grouped data
    left_data: list/tuple of object
        The ungrouped data
    """
    if fmt is object:
        return obj[0], obj[1:]
    assert isinstance(fmt, (list, tuple))
    obj_type = fmt[0]
    if obj_type is dict:
        ret = {}
        for key, fmt_i in fmt[1]:
            ret[key], obj = _regroup(obj, fmt_i)
        return ret, obj
    elif obj_type is list:
        ret = []
        for fmt_i in fmt[1]:
            res, obj = _regroup(obj, fmt_i)
            ret.append(res)
        return ret, obj
    else:
        raise TypeError


class UnsetFlag:
    """
    Special flag for unset values.
    """
    pass


# for backward compatible
_UnsetFlag = UnsetFlag

unset_flag = UnsetFlag


def map_aggregate(a, fn):
    """Apply fn to each Node appearing arg.
    arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(a, (tuple, list)):
        return type(a)(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return type(a)(dict(((k, map_aggregate(v, fn)) for k, v in a.items())))
    elif isinstance(a, slice):
        return type(a)(map_aggregate(a.start, fn), map_aggregate(a.stop, fn),
                       map_aggregate(a.step, fn))
    else:
        return fn(a)


def _compact(value):
    """Internal function to compact Empty object

    Parameters
    ----------
    value : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    type_for_resursive = (list, tuple, dict)

    def empty_elt(v):
        return (v is None or (isinstance(v, (list, tuple)) and len(v) == 0)
                or v == {})

    if isinstance(value, (list, tuple)):
        ret = deepcopy(value)
        to_delete = []
        for i, v in enumerate(value):
            if isinstance(v, type_for_resursive):
                ret[i] = _compact(v)
            if empty_elt(v):
                to_delete.append(i)
        if len(to_delete) > 0:
            to_delete.sort(reverse=True)
        for i in to_delete:
            ret.pop(i)
        if len(ret) == 0:
            ret = None
        return ret
    elif isinstance(value, dict):
        ret = deepcopy(value)
        to_delete = []
        for k, v in value.items():
            if isinstance(v, type_for_resursive):
                v = _compact(v)
            if empty_elt(v):
                to_delete.append(k)
            ret[k] = v
        for i in to_delete:
            ret.pop(i)
        if len(ret) == 0:
            ret = None
        return ret
    return value


class DataClassConfig(DataClassJsonMixin):
    def to_json(self, compact: bool = True) -> str:
        obj = self.to_dict()
        if compact:
            obj = _compact(obj)
        return json.dumps(obj)


def wait_until_finish(query_func, timeout=None, query_interval=5,
                      log_interval=30, wait_what=None):
    """
    Wait until finish.

    Parameters
    ----------
    query_func : callable
        The way to query whether the task is finished or not, return True
        or False, this function is called is the following way

        .. code-block:: python

            flag = query_func()
    timeout : float, optional
        Maximum waiting time, by default None
    query_interval : float, optional
        Wait until `query_interval` and then query again, by default 5
    log_interval : float, optional
        Log every `log_interval` seconds, by default 30
    wait_what : str, optional
        Waiting what, used for logging, by default None

    Raises
    ------
    TimeoutError
    """
    tic = time.time()
    pre_log_interval = log_interval
    while True:
        if query_func():
            break
        time.sleep(query_interval)
        time_cost = time.time() - tic
        if timeout is not None and time_cost > timeout:
            raise TimeoutError
        if time_cost > pre_log_interval:
            if wait_what:
                logger.info(f'{wait_what}: cost {time_cost} seconds')
            else:
                logger.info(f'Waiting cost {time_cost} seconds')
            pre_log_interval += log_interval


def call_and_retry(func, args=None, kwargs=None, stop_cond=None,
                   catch_exception=(Exception, ),
                   raise_catch_exception_when_failed=True,
                   max_retry=0, retry_interval=1):
    """
    Call function with retry.

    Parameters
    ----------
    func : callable
        Function
    args : tuple/list, optional
        Function input args, by default None
    kwargs : dict, optional
        Function kwargs, by default None
    stop_cond : callable, optional
        The way to parse function outputs, return True or False, by default None.
        This function is called in the following way:

        .. code-block:: python

            ret = func(*args, **kwargs)
            flag = stop_cond(ret)
            if flag:
                return (True, ret)
            else:
                # retry
                pass


    catch_exception : tuple, optional
        Handled exceptions, by default ('Exception', )
    raise_catch_exception_when_failed : bool, optional
        Whether to raise exception when retry failed, by default True
    max_retry : int, optional
        The maximum number of retry, by default 0
    retry_interval : int or tuple, optional
        Retry interval, by default 1

    Returns
    -------
    (bool, Any):
        The first one indicates the function is executed success or not, the
        second is the outputs.
    """  # noqa
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()
    if stop_cond is not None:
        assert callable(stop_cond)
    assert isinstance(catch_exception, (list, tuple))
    assert max_retry >= 0

    if isinstance(retry_interval, (int, float)):
        retry_interval = [retry_interval, retry_interval]
    assert isinstance(retry_interval, (list, tuple))
    assert len(retry_interval) == 2
    assert retry_interval[1] >= retry_interval[0] and retry_interval[0] >= 0

    retry_idx = 0

    while retry_idx <= max_retry:

        exception = None
        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            if not catch_exception or not isinstance(e, catch_exception):  # noqa
                raise
            exception = e

        if exception is not None:
            logger.warning(f'Call {func} with args={args}, kwargs={kwargs} failed! With exception = {exception}! Retry {retry_idx} times')  # noqa
            # raise
            if raise_catch_exception_when_failed and retry_idx >= max_retry:
                raise exception
            else:
                outputs = (False, exception)
        elif stop_cond is not None and not stop_cond(ret):
            logger.warning(f'Call {func} with args={args}, kwargs={kwargs} not success! Outputs = {ret}! Retry {retry_idx} times')  # noqa
            outputs = (False, ret)
        else:
            outputs = (True, ret)
            break

        time.sleep(random.uniform(retry_interval[0], retry_interval[1]))
        retry_idx += 1

    return outputs


class Enum(_Enum):

    @classmethod
    def value_of(cls, enum):
        if isinstance(enum, cls):
            pass
        elif isinstance(enum, str) and enum in cls._member_map_:
            enum = cls[enum]
        else:
            enum = cls(enum)
        return enum.value

    @classmethod
    def contains_value(cls, value):
        return value in cls._value2member_map_


def get_enum_value(x: Any) -> Any:
    """Turn an enum object into its value.

    If the parameter is not an enum object, return itself.
    """
    if isinstance(x, _Enum):
        return x.value
    return x


def _check_type(obj, expected_type, obj_name):
    assert isinstance(obj, expected_type), \
        f'{obj_name} should be instance of {expected_type}, but get {str(type(obj))}'  # noqa


def identity(x):
    """
    Identity pass input

    Parameters
    ----------
    x : object
    """
    return x


def multi_apply(func: Callable, *args) -> Tuple:
    """Use func on different objects and merge public attributes.

    Parameters
    ----------
    func: callable
        Function handle
    args: object
        Args of all objects.
    """
    map_results = list(map(func, *args))

    if isinstance(map_results[0], tuple):
        map_results = map(list, zip(*map_results))
    return tuple(map_results)


def now_time(to_string=True):
    time = datetime.datetime.now()
    if to_string:
        return time.strftime('%Y%m%d%H%M%S')


def get_operator_name(op, with_module_name=False):
    """Get operator name."""

    if inspect.isfunction(op) or inspect.isbuiltin(op):
        op_name = op.__name__
    else:
        op_name = op.__class__.__name__

    if with_module_name:
        op_name = op.__module__ + '.' + op_name

    return op_name
