from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from capbc.utils import _as_list
from .base import ClientMixin


class ConcurrnetPoolExecutorClient(ClientMixin):
    def __init__(self, workers):
        self.workers = _as_list(workers)
        if len(self.workers) == 0:
            raise ValueError('At least one worker')
        assert all([isinstance(worker_i, (ProcessPoolExecutor, ThreadPoolExecutor))  # noqa
                    for worker_i in self.workers])
        self._idx = -1
        self._num_workers = sum([worker_i._max_workers for worker_i in self.workers])  # noqa
        self._idx2pool_idx_map = dict()
        self._build_idx2pool_idx_map()

    def _inc_idx(self):
        self._idx += 1

    def _build_idx2pool_idx_map(self):
        pre = 0
        for pool_idx, pool_i in enumerate(self.workers):
            for idx in range(pre, pool_i._max_workers + pre):
                self._idx2pool_idx_map[idx] = pool_idx
            pre += pool_i._max_workers

    @property
    def num_workers(self):
        return self._num_workers

    def submit(self, fn, *args, **kwargs):
        self._inc_idx()
        idx = self._idx2pool_idx_map[self._idx % self.num_workers]
        worker = self.workers[idx]
        return worker.submit(fn, *args, **kwargs)
