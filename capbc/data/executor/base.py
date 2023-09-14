from typing import Optional
from abc import ABCMeta, abstractmethod

from capbc.registry import Registry
from capbc.workflow.symbol import Symbol


EXECUTOR_REGISTRY = Registry("EXECUTOR")


class Executor(metaclass=ABCMeta):

    def __init__(self, num_workers: Optional[int] = None,
                 address: Optional[str] = None):
        self.num_workers = num_workers
        self.address = address
        self._post_init()

    @abstractmethod
    def _post_init(self):
        pass

    @abstractmethod
    def compute(self, graph: Symbol):
        pass

    def iter_batches(self, graph: Symbol, batch_size: int):
        raise NotImplementedError
