from abc import ABCMeta, abstractmethod
from capbc.utils import (
    _as_list, _flatten, _regroup, unset_flag
)
from .trace import GraphTracer
from .proxy import (
    get_traced_graph,
    Variable,
    OptionalVariable,
)
from .engine import SymbolExecutor


__all__ = ['Block']


class Block(metaclass=ABCMeta):
    """
    Workflow Block. Supports symbolic execution and imperative execution. Users
    can call the :py:meth:`hybridize` method to enable symbolic execution.
    """
    def __init__(self):
        self._cached_graph = None
        self._output_fmt = None
        self._graph_executor = None
        self._hybridize = False
        self._input_names = set()
        self._optional_input_names = set()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Subclass should implement this function.
        """
        pass

    def hybridize(self, enabled=True):
        """
        Hybridize the block and then can run in symbolic mode.

        Parameters
        ----------
        enabled : bool, optional
            Whether enabled hybridization.
        """
        self._hybridize = enabled
        self._cached_graph = None
        self._output_fmt = None
        self._graph_executor = None
        self._input_names = set()

    def __call__(self, *args, **kwargs):
        if not self._hybridize:
            return self.forward(*args, **kwargs)
        else:
            return self.forward_graph(*args, **kwargs)

    def _visit_flat_args(self, args, fn):
        for arg_i in args:
            assert isinstance(arg_i, (Variable, OptionalVariable))
            fn(arg_i)

    def trace_graph(self, *args, **kwargs):
        """
        Trace the computational graph, all inputs are expected to be instance
        of :py:class:`Variable`

        Returns
        -------
        graph : :py:class:`Symbol`
            The computational graph.
        """

        if self._cached_graph is not None:
            return self._cached_graph

        def _fn(arg_i):
            assert arg_i._iname not in self._input_names, f'Duplicate name {arg_i._iname}'  # noqa
            assert arg_i._iname not in self._optional_input_names, f'Duplicate name {arg_i._iname}'  # noqa
            if isinstance(arg_i, Variable):
                self._input_names.add(arg_i._iname)
            else:
                self._optional_input_names.add(arg_i._iname)

        flat_args, fmt = _flatten(args)
        self._visit_flat_args(flat_args, _fn)
        args = _regroup(flat_args, fmt)[0]

        flat_kwargs, fmt = _flatten(kwargs)
        self._visit_flat_args(flat_kwargs, _fn)
        kwargs = _regroup(flat_kwargs, fmt)[0]

        with GraphTracer():
            output = self.forward(*args, **kwargs)
            flat_output, output_fmt = _flatten(output)
            self._cached_graph = get_traced_graph(flat_output)
            self._output_fmt = output_fmt

        return self._cached_graph

    def forward_graph(self, *args, **kwargs):
        """
        Forward the computational graph.
        """
        if self._cached_graph is None:
            self.trace_graph(*args, **kwargs)

        new_kwargs = dict()

        def _fn(arg_i):
            if isinstance(arg_i, Variable):
                assert arg_i._iname in self._input_names
            else:
                assert arg_i._iname in self._optional_input_names
            assert arg_i._idata is not unset_flag
            new_kwargs[arg_i._iname] = arg_i._idata

        flat_args, _ = _flatten(args)
        self._visit_flat_args(flat_args, _fn)

        flat_kwargs, _ = _flatten(kwargs)
        self._visit_flat_args(flat_kwargs, _fn)

        if self._graph_executor is None:
            self._graph_executor = SymbolExecutor(self._cached_graph)

        flat_output = self._graph_executor(new_kwargs)
        if len(self._cached_graph) == 1 and isinstance(flat_output, (list, tuple)):  # noqa
            flat_output = [flat_output, ]
        output, left = _regroup(_as_list(flat_output), self._output_fmt)
        assert not left, f'Failed to regroup, ungrouped elements: {left}'

        return output
