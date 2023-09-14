import functools
from abc import ABCMeta, abstractmethod
import inspect
from .trace import get_current_graph_tracer, execute_and_trace


__all__ = ['Operator']


class Operator(metaclass=ABCMeta):
    def __init__(self, *, allow_input_skip=False):
        assert isinstance(allow_input_skip, bool), f"Expected type `bool`, get but {type(allow_input_skip)}"  # noqa
        self.__workflow_allow_input_skip__ = allow_input_skip

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            return self.forward(*args, **kwargs)
        else:
            return execute_and_trace(self, args, kwargs,
                                     __workflow_allow_input_skip__=self.__workflow_allow_input_skip__)  # noqa
