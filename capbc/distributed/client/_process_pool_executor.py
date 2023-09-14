# Extend concurrent.futures.ProcessPoolExecutor to have
# the initializer args and mp_context args
import sys
import multiprocessing as mp
from concurrent.futures import _base
from concurrent.futures.process import (
    ProcessPoolExecutor as _ProcessPoolExecutor,
    _process_worker,
    EXTRA_QUEUED_CALLS,
)

if sys.version_info[:2] < (3, 7):

    def _process_worker_ex(call_queue, result_queue, initializer, initargs):
        if initializer is not None:
            try:
                initializer(*initargs)
            except BaseException:
                _base.LOGGER.critical('Exception in initializer:', exc_info=True)  # noqa
                return
        _process_worker(call_queue, result_queue)

    class ProcessPoolExecutor(_ProcessPoolExecutor):
        """
        Extend :py:class:`concurrent.futures.ProcessPoolExecutor` to
        support mp_context and initializer as input.
        """
        def __init__(self, max_workers=None, mp_context=None,
                     initializer=None, initargs=()):
            super().__init__(max_workers)

            if mp_context is None:
                mp_context = mp.get_context()
            self._mp_context = mp_context

            self._call_queue = mp_context.Queue(self._max_workers + EXTRA_QUEUED_CALLS)  # noqa
            self._result_queue = mp_context.SimpleQueue()

            if initializer is not None and not callable(initializer):
                raise TypeError("initializer must be a callable")
            self._initializer = initializer
            self._initargs = initargs

        def _adjust_process_count(self):
            for _ in range(len(self._processes), self._max_workers):
                p = self._mp_context.Process(
                    target=_process_worker_ex,
                    args=(
                        self._call_queue,
                        self._result_queue,
                        self._initializer,
                        self._initargs
                    ))
                p.start()
                self._processes[p.pid] = p

else:

    ProcessPoolExecutor = _ProcessPoolExecutor
