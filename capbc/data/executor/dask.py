import logging

from capbc.data.backend import Backend
from capbc.data.executor.base import (
    Executor,
    EXECUTOR_REGISTRY
)
from capbc.data.executor.utils import get_dask_client
from capbc.workflow.symbol import Symbol
from capbc.workflow import SymbolExecutor


logger = logging.getLogger(__name__)


@EXECUTOR_REGISTRY.register(name=Backend.Dask)
class DaskExecutor(Executor):

    def _post_init(self):

        self.client = get_dask_client(
            address=self.address,
            num_workers=self.num_workers
        )
        self._caches = dict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["client"] = None
        state["_caches"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self._post_init()

    def compute(self, graph: Symbol):

        import dask

        executor = SymbolExecutor(graph)
        dask_graph = executor(dict(), caches=self._caches)

        if dask.is_dask_collection(dask_graph):
            return dask_graph.compute()
        else:
            raise TypeError(f"Unsupported type {type(dask_graph)}")

    def iter_batches(self, graph, batch_size):
        raise NotImplementedError
