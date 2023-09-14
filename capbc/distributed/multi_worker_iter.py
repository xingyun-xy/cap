import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from ..utils import _as_list
from .client import (
    ClientMixin,
    MainProcessClient,
    MultiProcessingClient,
    ConcurrnetPoolExecutorClient,
)


logger = logging.getLogger(__name__)


__all__ = ['MultiWorkerIter', ]


class MultiWorkerIter(object):
    """
    Using multiple worker pools to process input iter one by one.

    Parameters
    ----------
    input_iter : iterable
        Iterable inputs
    worker_fn : callable
        This function is used in the following way

        .. code-block:: python

            input_iter = iter(input_iter)
            r = next(input_iter)
            output = worker_fn(r)
    client : :py:class:`capbc.distributed.client.ClientMixin`
        Worker client. A client can submit job to workers
    max_prefetch : int
        Maximum number of prefetching, by default 2 * num_workers
    keep_order : bool, optional
        Whether keep output order the same with input_iter, by default True
    return_input_data : bool, optional
        Whether return original input data.
    skip_exceptions : list/tuple of Exception
        Skip exception error
    """  # noqa
    def __init__(self, input_iter, worker_fn, client=None, max_prefetch=None,
                 keep_order=True, return_input_data=False,
                 skip_exceptions=None):

        if client is None:
            client = MainProcessClient()
        else:
            if not isinstance(client, ClientMixin):
                # compatible with multiprocessing.pool.Pool
                if all([isinstance(client_i, multiprocessing.pool.Pool)
                        for client_i in _as_list(client)]):
                    client = MultiProcessingClient(client)
                elif all([isinstance(client_i, (ProcessPoolExecutor, ThreadPoolExecutor))  # noqa
                          for client_i in _as_list(client)]):
                    client = ConcurrnetPoolExecutorClient(client)
        assert isinstance(client, ClientMixin), f"Expected type {ClientMixin}, but get {type(client)}"  # noqa

        if max_prefetch is None:
            max_prefetch = 2 * client.num_workers

        self._client = client
        self._worker_fn = worker_fn
        self._iter = iter(input_iter)
        self._data_buffer = dict()
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._keep_order = keep_order
        self._return_input_data = return_input_data
        self._skip_exceptions = skip_exceptions

        for _ in range(max_prefetch):
            self._push_next()

    def _push_next(self):
        try:
            r = next(self._iter)
        except StopIteration:
            return
        ret = self._client.submit(self._worker_fn, r)
        self._data_buffer[self._sent_idx] = dict(ret=ret)
        if self._return_input_data:
            self._data_buffer[self._sent_idx]["input_data"] = r
        self._sent_idx += 1

    def __iter__(self):
        return self

    def __next__(self):
        self._push_next()
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, "Data buffer should be empty at this moment"  # noqa
            raise StopIteration
        assert self._rcvd_idx < self._sent_idx, \
            "rcvd_idx must be smaller than sent_idx"
        if self._keep_order:
            assert self._rcvd_idx in self._data_buffer, \
                "fatal error with _push_next, rcvd_idx missing"
            ret = self._data_buffer.pop(self._rcvd_idx)
            ret["ret"] = ret["ret"]
        else:
            while True:
                for key, value in self._data_buffer.items():
                    if value["ret"].done():
                        break
                ret = self._data_buffer.pop(key)
                ret["ret"] = ret["ret"]
                break

        self._rcvd_idx += 1

        if self._skip_exceptions is not None:
            try:
                ret["ret"] = ret["ret"].result()
            except self._skip_exceptions as e:
                logger.warn(f"Skip error: {e}")
                return self.__next__()
        else:
            ret["ret"] = ret["ret"].result()

        if self._return_input_data:
            return (ret["input_data"], ret["ret"])
        else:
            return ret["ret"]

    def __del__(self):
        if hasattr(self, "_rcvd_idx"):
            while self._rcvd_idx < self._sent_idx:
                if self._rcvd_idx in self._data_buffer:
                    ret = self._data_buffer.pop(self._rcvd_idx)
                    if self._client is not None:
                        try:
                            _ = ret["ret"].result()
                        except Exception as e:
                            logger.warn(f"Skip error: {e}")
                self._rcvd_idx += 1
