import multiprocessing
from capbc.utils import _as_list
from .base import ClientMixin


__all__ = ['MultiProcessingClient', ]


class _MPClientFuture(object):
    """
    Wrap :py:class:`multiprocessing.pool.ApplyResult` to have the method
    `result`

    Parameters
    ----------
    apply_result : :py:class:`multiprocessing.pool.ApplyResult`
    """
    def __init__(self, apply_result):
        self.apply_result = apply_result

    def result(self, timeout=None):
        return self.apply_result.get(timeout)

    def done(self):
        return self.apply_result.ready()


class MultiProcessingClient(ClientMixin):
    """
    Wrap :py:class:`multiprocessing.pool.Pool` to be class of
    :py:class:`ClientMixin`

    Parameters
    ----------
    worker_pools : list/tuple of :py:class:`multiprocessing.pool.Pool`
    """
    def __init__(self, worker_pools):
        self.worker_pools = _as_list(worker_pools)
        if len(self.worker_pools) == 0:
            raise ValueError('At least one worker pool')
        assert all([isinstance(pool_i, multiprocessing.pool.Pool)
                    for pool_i in self.worker_pools])
        self._idx = -1
        self._num_workers = sum([pool_i._processes for pool_i in self.worker_pools])  # noqa
        self._idx2pool_idx_map = dict()
        self._build_idx2pool_idx_map()

    def _inc_idx(self):
        self._idx += 1

    def _build_idx2pool_idx_map(self):
        pre = 0
        for pool_idx, pool_i in enumerate(self.worker_pools):
            for idx in range(pre, pool_i._processes + pre):
                self._idx2pool_idx_map[idx] = pool_idx
            pre += pool_i._processes

    @property
    def num_workers(self):
        return self._num_workers

    def submit(self, fn, *args, **kwargs):
        self._inc_idx()
        worker_pool = self.worker_pools[self._idx2pool_idx_map[self._idx % self.num_workers]]  # noqa
        ret = worker_pool.apply_async(fn, args, kwargs)
        return _MPClientFuture(ret)
