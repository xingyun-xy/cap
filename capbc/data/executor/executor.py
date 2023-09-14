from typing import Union, Optional

from capbc.data.backend import Backend
from capbc.data.datapipe.utils import DataPipeOp
from capbc.data.executor.base import Executor, EXECUTOR_REGISTRY
from capbc.workflow.symbol import Symbol


__all__ = ["DataPipeExecutor", ]


GRAPH_TYPE = Union["DataPipe", Symbol]


class DataPipeExecutor(Executor):
    """Executor of :py:class:`capbc.data.DataPipe`."""

    _current = None

    def __init__(self, backend: Union[Backend, str],
                 num_workers: Optional[int] = None,
                 address: Optional[str] = None):
        """

        Parameters
        ----------
        backend : Union[Backend, str]
            Executor backend, see :py:class:`capbc.data.Backend` for more details.
        num_workers : Optional[int]
            The number of workers, by default None
        address : Optional[str]
            Master address or scheduler address, by default None.
        """  # noqa
        self.backend = Backend(backend)
        if self.backend not in EXECUTOR_REGISTRY:
            raise ValueError(f"{self.backend} is not supported!")
        executor = EXECUTOR_REGISTRY.get(self.backend)(num_workers=num_workers)
        self.executor = executor
        self._old = None
        super().__init__(num_workers=num_workers)

    def _post_init(self):
        pass

    def __enter__(self):
        self._old = DataPipeExecutor._current
        DataPipeExecutor._current = self
        DataPipeOp.set_backend(self.backend)
        return self

    def __exit__(self, ptype, value, trace):
        if self._old is None:
            DataPipeOp.reset_backend()
        DataPipeExecutor._current = self._old
        self._old = None

    @classmethod
    def get_current(cls, required_initialized=True):
        if required_initialized and cls._current is None:
            raise RuntimeError("DataPipeExecutor is not initialized!")
        return cls._current

    def _get_graph(self, graph: GRAPH_TYPE) -> Symbol:

        from capbc.data.datapipe import DataPipe

        if isinstance(graph, DataPipe):
            graph = graph.graph

        return graph

    def compute(self, graph: GRAPH_TYPE):
        with self:
            return self.executor.compute(self._get_graph(graph))

    def iter_batches(self, graph: GRAPH_TYPE, batch_size: int):
        with self:
            return self.executor.iter_batches(self._get_graph(graph, batch_size))  # noqa
