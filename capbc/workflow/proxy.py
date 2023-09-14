import inspect
import dis
import operator
from deprecated import deprecated
from capbc.utils import _as_list
from .trace import (
    execute_and_trace,
    call_obj,
    get_attr_and_call,
    get_traced_graph,
    get_traced_outputs,
    GraphTracer,
)
from .symbol import Node, Symbol, group


__all__ = ["WorkflowVariable", 'Variable', 'OptionalVariable', 'Constant',
           'get_traced_graph', 'get_traced_outputs']


class _ProxyDataUnsetFlag:
    pass


def _not_implemented(func):

    def _inner(*args, **kwargs):
        name = func.__name__
        raise NotImplementedError(f'{WorkflowVariable} does not support function {name}')  # noqa

    return _inner


class WorkflowVariable(object):
    """Bind data with :py:class:`Symbol`. This data structure is used to trace
    the computational graph.

    .. note::

        We need this data strucure to trace the computational graph because
        the input and output of our workflow operators are arbitrary.

    Parameters
    ----------
    data : object
        Data
    symbol : :py:class:`Symbol`
        Symbolic graph.
    """
    def __init__(self, symbol, data=_ProxyDataUnsetFlag):
        self._isymbol = symbol
        self._idata = data

    @property
    def _iop(self):
        return self._isymbol._outputs[0].op

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise RuntimeError(f'Cannot modify type(self)')
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self.__dict__:
            return super().__getattr__(name)
        else:
            variable = execute_and_trace(getattr, args=(self, name))
            if isinstance(variable, WorkflowVariable):
                return Attribute(pre_variable=self, variable=variable,
                                 attr_name=name)
            else:
                return variable

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()

    __deepcopy__ = None

    @_not_implemented
    def __hash__(self):
        pass

    @_not_implemented
    def __str__(self):
        pass

    @_not_implemented
    def __bool__(self):
        pass

    def __call__(self, *args, **kwargs):
        return call_obj(self, *args, **kwargs)

    def __iter__(self):
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst = list(dis.get_instructions(calling_frame.f_code))[calling_frame.f_lasti // 2]  # noqa
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[idx] for idx in range(inst.argval))
        elif inst.opname == 'GET_ITER':
            if isinstance(self._idata, (list, tuple)):
                return iter([self[idx] for idx in range(len(self._idata))])
            elif isinstance(self._idata, dict):
                return iter(self._idata)
            elif self._idata is _ProxyDataUnsetFlag:
                raise NotImplementedError('Cannot return iter when data is unset')  # noqa
            else:
                raise NotImplementedError(f'Unsupported iter for {type(self._idata)}')  # noqa
        else:
            raise NotImplementedError

    def keys(self):
        assert self._idata is not _ProxyDataUnsetFlag
        return self._idata.keys()

    def __len__(self):
        assert self._idata is not _ProxyDataUnsetFlag
        return len(self._idata)

    def wait(self, variables):
        """
        Wait for computation to be finished
        """
        for variable_i in _as_list(variables):
            assert isinstance(variable_i, WorkflowVariable), \
                f"Expected type {WorkflowVariable}, but get {type(variable_i)}"
            self._isymbol.wait(variable_i._isymbol)

    def after(self, *args, **kwargs):
        return self.wait(*args, **kwargs)


class Variable(WorkflowVariable):
    """
    Bind data with name.

    Parameters
    ----------
    data : object
        Data
    name : str
        Name, should be uniquely
    """
    def __init__(self, name, data=_ProxyDataUnsetFlag):
        assert isinstance(name, str), f'Expected str, but get {type(name)}'
        super().__init__(Symbol.create_placeholder(name), data)
        self._iname = name


class OptionalVariable(WorkflowVariable):
    def __init__(self, name, default):
        assert isinstance(name, str), f'Expected str, but get {type(name)}'
        assert default is not _ProxyDataUnsetFlag, f'data cannot be {_ProxyDataUnsetFlag} for {type(self)}'  # noqa
        super().__init__(Symbol.create_optional_placeholder(name, default), default)  # noqa
        self._iname = name


class Constant(WorkflowVariable):
    def __init__(self, data, name=None):
        super().__init__(Symbol.create_constant(data, name), data)
        self._iname = self._isymbol.name


class Attribute(WorkflowVariable):
    """
    Proxy get object attribute or get object attribute and call

    Parameters
    ----------
    pre_variable : :py:class`WorkflowVariable`
        Previous variable
    variable : :py:class:`WorkflowVariable`
        Variable for get attribute operator
    attr_name : str
        Get attribute name
    """
    def __init__(self, pre_variable, variable, attr_name):
        assert isinstance(pre_variable, WorkflowVariable)
        self._ipre_variable = pre_variable
        assert isinstance(variable, WorkflowVariable)
        self._ivariable = variable
        self._iattr_name = attr_name

    @property
    def _isymbol(self):
        return self._ivariable._isymbol

    @property
    def _idata(self):
        return self._ivariable._idata

    def __call__(self, *args, **kwargs):
        return get_attr_and_call(self._ipre_variable, self._iattr_name,
                                 *args, **kwargs)  # noqa


reflectable_magic_methods = ('add', 'sub', 'mul', 'floordiv', 'truediv',
                             'div', 'mod', 'pow', 'lshift', 'rshift',
                             'and', 'or', 'xor')
magic_methods = ('eq', 'ne', 'lt', 'gt', 'le', 'ge', 'pos', 'neg',
                 'invert', 'getitem') + reflectable_magic_methods


# patch magic methods
for method_i in magic_methods:

    def wrap(method):

        def _impl(self, *args, **kwargs):
            return execute_and_trace(getattr(operator, method),
                                     args=(self, ) + args, kwargs=kwargs)

        _impl.__name__ = method
        setattr(WorkflowVariable, f'__{method}__', _impl)

    wrap(method_i)

# patch reflectable magic methods
for method_i in reflectable_magic_methods:

    def wrap_reflectable(method):

        def _impl(self, rhs):
            return execute_and_trace(getattr(operator, method),
                                     args=(rhs, self), kwargs=None)

        _impl.__name__ = method
        setattr(WorkflowVariable, f'__r{method}__', _impl)

    wrap_reflectable(method_i)
