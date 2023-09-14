from typing import Dict, Union, Callable, Sequence, Any, Optional
import functools
import inspect
import types
import copy
import threading
import logging
from enum import Enum
from capbc.patch import Patcher
from capbc.utils import _as_list, map_aggregate, get_operator_name
from capbc.utils.color_message import Color, ShowMethod, color_message
from .symbol import Node, Symbol, group
from .exception import SkipDownStream, OpSkipFlag
from types import MethodType

__all__ = [
    "make_traceable",
    "GraphTracer",
    "ControlFlow",
    "ControlFlowPause",
    "get_traced_graph",
    "get_traced_outputs",
]


logger = logging.getLogger(__name__)


def _make_function_traceable(fn, *, allow_input_skip=False):
    """Wrap function `fn` to make it traceable under GraphTracer scope
    (no-op when outside it).
    """

    @functools.wraps(fn)
    def _wrap(*args, **kwargs):
        """This function is wrapped by `capbc.workflow` to make it traceable
        under GraphTracer scope (no-op when outside it).
        """
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            return fn(*args, **kwargs)
        else:
            return execute_and_trace(
                fn,
                args,
                kwargs,
                __workflow_allow_input_skip__=allow_input_skip,
            )  # noqa

    _wrap.__workflow_traceable__ = True
    _wrap.__workflow_allow_input_skip__ = allow_input_skip
    return _wrap


def _make_class_traceable(
    obj, *, return_patch_func_only=False, allow_input_skip=False
):  # noqa
    """Wrap __call__ function of class `obj` to make it traceable under
    GraphTracer scope (no-op when outside it).
    """
    assert hasattr(obj, "__call__"), "object should have the __call__ function"
    assert isinstance(
        obj.__call__, types.FunctionType
    ), "the __call__ function of object should not be static"

    fn = obj.__call__

    @functools.wraps(fn)
    def _wrap(self, *args, **kwargs):
        """This function is wrapped by `capbc.workflow` to make it traceable
        under GraphTracer scope (no-op when outside it).
        """
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            return fn(self, *args, **kwargs)
        else:
            return execute_and_trace(
                self,
                args,
                kwargs,
                __workflow_allow_input_skip__=allow_input_skip,
            )  # noqa

    _wrap.__workflow_traceable__ = True
    _wrap.__workflow_allow_input_skip__ = allow_input_skip

    if return_patch_func_only:
        return _wrap
    else:
        obj.__call__ = _wrap
        return obj


def make_traceable(obj=None, *, allow_input_skip=False):
    """A decorator that make a callable object to be an workflow operator.

    Parameters
    ----------
    obj : callable, optional
        Wrapped callable object, by default None
    allow_input_skip : bool, optional
        Whether allow input is skip, for more information, please see the workflow tutorial, by default False
    """  # noqa

    def _impl(obj):
        assert callable(obj), f"{obj} should be callable"

        if is_workflow_traceable(obj):
            return obj

        if inspect.isfunction(obj) or isinstance(obj, MethodType):
            return _make_function_traceable(
                obj, allow_input_skip=allow_input_skip
            )
        else:
            return _make_class_traceable(
                obj, allow_input_skip=allow_input_skip
            )

    if obj is None:
        return _impl
    else:
        return _impl(obj)


def is_workflow_traceable(obj):
    def _trace_attr_is_true(fn):
        return (
            hasattr(fn, "__workflow_traceable__") and fn.__workflow_traceable__
        )

    if inspect.isfunction(obj):
        return _trace_attr_is_true(obj)
    elif hasattr(obj, "__call__"):
        return _trace_attr_is_true(obj.__call__)
    else:
        return False


class GraphTracer(object):
    """
    Tracing workflows when running under this scope.

    Parameters
    ----------
    imperative : bool, optional
        Whether tracing graph using imperative mode, by default False.

        Imperative means executing the operator and then tracing, otherwise
        just tracing without executing.

        .. code-block:: python

            @make_traceable
            def add(a, b):
                ret = a + b
                print(ret)  # when executing, will print the result
                return ret

            a = Variable('a', 1)
            b = Variable('b', 2)

            with GraphTracer(imperative=True):
                c = add(a, b)  # the terminal will output 3

            with GraphTracer(imperative=False):
                c = add(a, b)  # the terminal will output nothing
    """

    # thread id 2 current GraphTracer instance or None
    _tid2current = dict()

    _TRACEABLE_CLASSES = ()
    _BASIC_TYPES = (int, float, str, type(None), bool, Enum)
    _OP_NAME_FUNC = dict()

    def __init__(self, imperative=False):
        self._old_scope = None
        self._op2count = dict()
        self.imperative = imperative
        self._patcher = Patcher()

    def __enter__(self):
        self._old_scope = GraphTracer.current()

        if self._old_scope is not None:
            self._op2count = copy.copy(self._old_scope._op2count)

        GraphTracer.set_current(self)

        self._patcher.__enter__()
        for obj in GraphTracer._TRACEABLE_CLASSES:
            if not is_workflow_traceable(obj):
                self._patcher.patch(
                    obj,
                    "__call__",
                    _make_class_traceable(obj, return_patch_func_only=True),
                )

        return self

    def __exit__(self, ptype, value, trace):
        self._patcher.__exit__(ptype, value, trace)

        if self._old_scope is not None:
            self._old_scope._op2count.update(self._op2count)

        # recovery current
        if self._old_scope is None:
            tid = threading.get_ident()
            # pop to avoid dict `_tid2current` explosion.
            GraphTracer._tid2current.pop(tid)
        else:
            GraphTracer.set_current(self._old_scope)

    def trace_op(self, op):
        op_type_name = get_operator_name(op)
        if op_type_name not in self._op2count:
            self._op2count[op_type_name] = 0
        self._op2count[op_type_name] += 1

    def get_op_name(self, op, args, kwargs):
        op_type_name = get_operator_name(op)
        count = self._op2count[op_type_name]
        if op_type_name in GraphTracer._OP_NAME_FUNC:
            return GraphTracer._OP_NAME_FUNC[op_type_name](
                op, args, kwargs, count
            )
        else:
            return f"{op_type_name}{count}"

    @classmethod
    def register_class_traceable_under_scope(cls, fn):
        assert type(fn).__name__ == "type", f"{fn} is not class"
        assert hasattr(fn, "__call__"), f"{fn} should be callable"
        if fn not in cls._TRACEABLE_CLASSES:
            cls._TRACEABLE_CLASSES += (fn,)

    @classmethod
    def register_basic_types(cls, obj):
        if obj not in cls._BASIC_TYPES:
            cls._BASIC_TYPES += (obj,)

    @classmethod
    def register_op_name_function(cls, name: str, func: Callable):
        """Register name function for operator named `name`.

        Parameters
        ----------
        name : str
            Operator name.
        func : callable
            This function is called in the following way

            .. code-block:: python

                name = func(op, op_args, op_kwargs, conut)
        """
        if name in cls._OP_NAME_FUNC:
            raise KeyError(f"{name} already registered")
        cls._OP_NAME_FUNC[name] = func

    @classmethod
    def current(cls) -> Union["GraphTracer", None]:
        """Get current GraphTracer instance of current thread.

        .. note::

            Different thread owns different `current`.

        Returns
        -------
        current : Union[GraphTracer, None]
            Current GraphTracer instance.
        """
        tid = threading.get_ident()
        cls._tid2current.setdefault(tid, None)
        return cls._tid2current[tid]

    @classmethod
    def set_current(cls, current: Union["GraphTracer", None]):
        """Set current GraphTracer instance of current thread.

        .. note::

            Different thread owns different `current`.

        Parameters
        ----------
        current: Union[GraphTracer, None]
            Current GraphTracer instance.
        """
        assert current is None or isinstance(current, GraphTracer), type(
            current
        )
        tid = threading.get_ident()
        cls._tid2current[tid] = current

    @classmethod
    def is_active(cls):
        """
        Whether is under the GraphTracer scope.
        """
        return cls.current() is not None


def get_current_graph_tracer():
    return GraphTracer.current()


class GraphTracerPause(object):
    def __init__(self):
        self._old_scope = None

    def __enter__(self):
        self._old_scope = GraphTracer.current()
        GraphTracer.set_current(None)

    def __exit__(self, ptype, value, trace):
        GraphTracer.set_current(self._old_scope)


class ControlFlow:
    _current = None

    def __init__(self, *, wait=()):
        self._old = None
        self.wait_nodes = list(_as_list(wait))

    def __enter__(self):
        self._old = ControlFlow._current
        if self._old is None:
            ControlFlow._current = self
            return self
        else:
            ControlFlow._current = ControlFlow._current.copy()
            ControlFlow._current.wait_nodes.extend(self.wait_nodes)
            return ControlFlow._current

    def __exit__(self, ptype, value, trace):
        ControlFlow._current = self._old
        self._old = None

    def copy(self):
        return ControlFlow(wait=copy.copy(self.wait_nodes))

    @classmethod
    def current(cls):
        return cls._current

    @classmethod
    def is_active(cls):
        return cls._current is not None


class ControlFlowPause:

    def __init__(self):
        self._old = None

    def __enter__(self):
        self._old = ControlFlow._current
        ControlFlow._current = None
        return self

    def __exit__(self, ptype, value, trace):
        ControlFlow._current = self._old
        self._old = None


def execute_and_trace(
    op, args=None, kwargs=None, *, __workflow_allow_input_skip__=False
):
    """
    Executing an operator and trace if under the :py:class:`GraphTracer` scope.
    """

    from .proxy import WorkflowVariable, _ProxyDataUnsetFlag

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    graph_tracer = get_current_graph_tracer()
    graph_tracer_active = graph_tracer is not None

    has_op_skip_flag = False

    wait_nodes = []

    def _map_func(x, take_node=False, check_type=True):
        if isinstance(x, WorkflowVariable):
            if take_node:
                assert len(x._isymbol) == 1
                return x._isymbol._outputs[0]
            else:
                assert (
                    x._idata is not _ProxyDataUnsetFlag
                ), f"You should bind data with WorkflowVariable {x._isymbol.name} when tracing with imperative mode or in dynamic mode"  # noqa
                if x._idata is OpSkipFlag:
                    nonlocal has_op_skip_flag
                    has_op_skip_flag = True
                return x._idata
        if check_type:
            assert isinstance(
                x, GraphTracer._BASIC_TYPES
            ), f"Invalid type {type(x)}, only allow {GraphTracer._BASIC_TYPES}"  # noqa
        return x

    if graph_tracer is not None and ControlFlow.is_active():
        for node in ControlFlow.current().wait_nodes:
            if isinstance(node, WorkflowVariable):
                wait_nodes.append(node)
                if node._idata is OpSkipFlag:
                    has_op_skip_flag = True
            elif node is OpSkipFlag:
                has_op_skip_flag = True

    if graph_tracer is None or graph_tracer.imperative:

        with GraphTracerPause():

            new_args = map_aggregate(
                args,
                functools.partial(_map_func, check_type=graph_tracer_active),
            )
            new_kwargs = map_aggregate(
                kwargs,
                functools.partial(_map_func, check_type=graph_tracer_active),
            )

            if has_op_skip_flag and not __workflow_allow_input_skip__:
                logger.warning(
                    f"[Tracing]: Some child node has raise the {SkipDownStream} exception, ignore executing {op}"  # noqa
                )
                output = OpSkipFlag
            else:
                try:
                    output = op(*new_args, **new_kwargs)
                except SkipDownStream as e:
                    msg = f"[Tracing]: {op} raise the {SkipDownStream}, args={new_args}, kwargs={new_kwargs}"  # noqa
                    if e.args:
                        msg += f", reason={e.args}"
                    msg += ", will ignore downstream operators"
                    msg = color_message(
                        msg,
                        show_method=ShowMethod.highlight,
                        background_color=Color.red,
                    )
                    logger.warning(f"\n\n{msg}\n\n")  # noqa
                    output = OpSkipFlag

    else:

        output = _ProxyDataUnsetFlag

    if graph_tracer is None:
        return output

    new_args = map_aggregate(
        args, functools.partial(_map_func, take_node=True)
    )
    new_kwargs = map_aggregate(
        kwargs, functools.partial(_map_func, take_node=True)
    )

    graph_tracer.trace_op(op)
    node = Node(
        op=op,
        name=graph_tracer.get_op_name(op, args, kwargs),
        args=new_args,
        kwargs=new_kwargs,
        attr=dict(__workflow_allow_input_skip__=__workflow_allow_input_skip__),
    )
    symbol = Symbol(node)
    ret = WorkflowVariable(symbol=symbol, data=output)

    if wait_nodes:
        ret.wait(wait_nodes)

    return ret


@make_traceable
def call_obj(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@make_traceable
def get_attr_and_call(obj, attr_name, *args, **kwargs):
    attr = getattr(obj, attr_name)
    return attr(*args, **kwargs)


def get_traced_graph(*outputs):
    """
    Getting graph by traced outputs.
    """

    # for backward compatible
    if len(outputs) == 1:
        outputs = _as_list(outputs[0])

    from .proxy import WorkflowVariable

    assert all(
        [isinstance(out_i, WorkflowVariable) for out_i in _as_list(outputs)]
    )  # noqa

    return group(*(out_i._isymbol for out_i in _as_list(outputs)))


def get_traced_outputs(*outputs):

    # for backward compatible
    if len(outputs) == 1:
        outputs = _as_list(outputs[0])

    assert len(outputs) >= 1, "At least one output"

    from .proxy import WorkflowVariable

    assert all(
        [isinstance(out_i, WorkflowVariable) for out_i in _as_list(outputs)]
    )  # noqa

    outputs = list((out_i._idata for out_i in _as_list(outputs)))
    return outputs[0] if len(outputs) == 1 else outputs
