import logging
import multiprocessing
from abc import abstractmethod
import functools
import time
from typing import List
import copy
import threading
import pickle
import dill
try:
    from distributed import WorkerPlugin, get_worker
except ImportError:
    WorkerPlugin = None
    get_worker = None
from capbc.utils import _as_list
from capbc.distributed.multi_worker_iter import MultiWorkerIter
from capbc.distributed.utils import close_worker_pools
from capbc.distributed.client import (
    MultiProcessingClient,
    DaskClient,
    ConcurrnetPoolExecutorClient,
)
from capbc.distributed.client._process_pool_executor import (
    ProcessPoolExecutor
)
from capbc.distributed.client.dask import (
    get_worker_rank
)
from capbc.resource_manager import (
    ContextResourceManager, get_current_resource_manager
)
from .basic import (
    BaseSymbolExecutor,
    SymbolExecutor,
)
from ..exception import OpSkipFlag
from ..symbol import Symbol
from ..env import (
    get_serializer,
    set_serializer,
)


dill.extend(False)


__all__ = ['Reducer', 'UniqueReducer', 'ConcatenateReducer',
           'SimpleParallelContext',
           'SimpleParallelExecutor', ]


logger = logging.getLogger(__name__)


class Reducer(object):

    @abstractmethod
    def __call__(self, outputs, part):
        pass


class UniqueReducer(Reducer):
    def __init__(self):
        self._cached = set()

    def __call__(self, outputs: List, part: List):
        for part_i in _as_list(part):
            if part_i in self._cached:
                continue
            self._cached.add(part_i)
            outputs.append(part_i)
        return outputs


class ConcatenateReducer(Reducer):
    def __call__(self, outputs: List, part: List):
        outputs.extend(_as_list(part))
        return outputs


class SimpleParallelContext(object):
    _current = None

    def __init__(self, worker_id: int, num_worker: int):
        self._worker_id = worker_id
        self._num_worker = num_worker
        assert 0 <= worker_id < num_worker
        self._old = None

    def __enter__(self):
        self._old = SimpleParallelContext._current
        SimpleParallelContext._current = self
        return self

    def __exit__(self, ptype, value, trace):
        SimpleParallelContext._current = self._old

    @property
    def worker_id(self):
        return self._worker_id

    @property
    def num_worker(self):
        return self._num_worker

    @staticmethod
    def get_current():
        return SimpleParallelContext._current


def get_current_simple_parallel_context():
    return SimpleParallelContext._current


class _InputIter(object):
    def __init__(self, inputs, partition_input_keys,
                 partition_input_as_iterable=False, batch_size=1):
        assert isinstance(inputs, dict)
        for key in partition_input_keys:
            assert key in inputs
        self.inputs = inputs
        self.partition_input_keys = partition_input_keys
        self.partition_input_as_iterable = partition_input_as_iterable

    def __iter__(self):
        input_iterators = dict()
        inputs = dict()
        for name, value in self.inputs.items():
            if name in self.partition_input_keys:
                input_iterators[name] = iter(value)
            else:
                inputs[name] = value
        while True:
            inputs = copy.copy(inputs)
            end_flag = False
            for name, iterator in input_iterators.items():
                try:
                    data = next(iterator)
                    if end_flag:
                        raise RuntimeError(
                            'Input length of {self.partition_input_keys} are not equal')  # noqa
                except StopIteration:
                    end_flag = True
                    continue
                if self.partition_input_as_iterable:
                    inputs[name] = (data, )
                else:
                    inputs[name] = data
            if end_flag:
                return
            yield inputs


if WorkerPlugin is not None:

    class _DaskWorkerInitializer(WorkerPlugin):
        def __init__(self, serializer, symbol_buf, resource_manager_buf,
                     num_worker, with_lock=True):  # noqa
            self.symbol_buf = symbol_buf
            self.serializer = serializer
            self.resource_manager_buf = resource_manager_buf
            self.num_worker = num_worker
            self.with_lock = with_lock
            self.worker_id = None
            self.lock = None
            self.params = None

        def setup(self, worker):

            set_serializer(self.serializer)

            self.worker_id = get_worker_rank(worker)
            if self.with_lock:
                self.lock = threading.Lock()
            else:
                self.lock = None
            time.sleep(2 * int(self.worker_id))
            resource_manager = dill.loads(self.resource_manager_buf)
            if resource_manager is None:
                resource_manager = ContextResourceManager()
            self.params = dict(
                worker_id=self.worker_id,
                num_worker=self.num_worker,
                executor=SymbolExecutor(Symbol.loads(self.symbol_buf)),
                resource_manager=resource_manager,
            )

    def _dask_worker_fn(inputs, key):
        worker = get_worker()
        plugin = worker.plugins[key]
        if plugin.lock is not None:
            with plugin.lock:
                return _worker_fn(inputs, plugin.params)
        else:
            return _worker_fn(inputs, plugin.params)


_worker_fn_param = None


def _worker_initializer(worker_id, num_worker, symbol_buf,
                        resource_manager_buf, serializer):
    global _worker_fn_param
    if resource_manager_buf is not None:
        resource_manager = dill.loads(resource_manager_buf)
    else:
        resource_manager = None
    if resource_manager is None:
        resource_manager = ContextResourceManager()
    if serializer is not None:
        set_serializer(serializer)
    _worker_fn_param = dict(
        worker_id=worker_id,
        num_worker=num_worker,
        executor=SymbolExecutor(Symbol.loads(symbol_buf)),
        resource_manager=resource_manager,
    )


def _worker_fn(inputs, param):
    with param['resource_manager'], SimpleParallelContext(
            worker_id=param['worker_id'], num_worker=param['num_worker']):
        ret = param['executor'](inputs)
        return ret


def _mp_worker_fn(inputs):
    global _worker_fn_param
    return _worker_fn(inputs, _worker_fn_param)


class SimpleParallelExecutor(BaseSymbolExecutor):
    VALID_BACKEND = [None, 'mp_spawn', 'dask', 'concurrent_mp_spawn']
    """
    Parallel graph executor.

    Parameters
    ----------
    symbol : :py:class:`capbc.workflow.symbol.Symbol`
        Computational graph
    partition_input_keys : list/tuple of str
        Input keys to be partitioned
    partition_input_as_iterable : book, optional
        Whether wrap partitioned inputs as list, by default True
    reducers : list/tuple of :py:class:`Reducer`, optional
        Output result reducer, by default None
    num_worker : int, optional
        The number of workers, by default 0
    backend : str, optional
        Parallel backend, optional values are
        {None, mp_spawn, mp_fork, dask, concurrnet_mp_spawn, concurrnet_mp_fork},
        by defualt ``mp_spawn``.
        ``mp_spawn`` means multiprocessing using starting
        method spawn,
        ``mp_fork`` means multiprocessing using starting method
        fork,
        ``dask`` means using :py:class:`dask.distributed.Client`,
        ``concurrent_mp_spawn`` means using :py:class:`concurrent.futures.ProcessPoolExecutor` with
        multiprocessing starting method spawn,
        ``concurrent_mp_fork`` means using
        :py:class:`concurrent.futures.ProcessPoolExecutor` with multiprocessing
        starting method fork.

        .. note::

            Currently, only the dask backend supports distributed

    log_interval : int, optional
        Logging interval after processed `log_interval` samples, by default 10
    skip_unset_output : bool, optional
        Whether skip unset outputs, this happens when internal node raise the
        :py:class:`capbc.workflow.exception.SkipDownStream` exception, by
        default True
    """  # noqa
    def __init__(self, symbol, partition_input_keys,
                 partition_input_as_iterable=True, reducers=None,
                 num_worker=0, backend='mp_spawn', batch_size=1,
                 log_interval=10, skip_unset_output=True):
        super().__init__(symbol)
        self.num_outputs = len(self.symbol)
        if reducers is None:
            reducers = [ConcatenateReducer() for _ in range(self.num_outputs)]
        else:
            reducers = _as_list(reducers)
            assert all([isinstance(reducer_i, Reducer)
                        for reducer_i in reducers])
            assert len(reducers) == self.num_outputs, \
                f'The number of reducers and outputs are not match, {len(reducers)} VS. {self.num_outputs}'  # noqa
        self.reducers = reducers
        self.partition_input_keys = partition_input_keys
        self.partition_input_as_iterable = partition_input_as_iterable
        self.log_interval = log_interval
        self.num_worker = num_worker
        if num_worker == 0 and backend is not None:
            logger.warning(f'Reset backend {backend} to be None since the number of workers is 0')  # noqa
            backend = None
        assert backend in self.VALID_BACKEND, \
            f"Invalid backend: {backend}, all valid backend: {','.join(map(str, self.VALID_BACKEND))}"  # noqa
        self.backend = backend
        self.skip_unset_output = skip_unset_output
        if batch_size < 1:
            raise ValueError(f'required batch_size >= 1, but get {batch_size}')
        if batch_size > 1 and not partition_input_as_iterable:
            raise ValueError(f'partition_input_as_iterable should be True when batch_size > 1')  # noqa
        self.batch_size = batch_size

    def _get_client_and_worker_fn_and_done_callback(self):

        if self.backend is None:
            worker_fn = functools.partial(
                _worker_fn, param=dict(
                    worker_id=0, num_worker=1,
                    executor=SymbolExecutor(self.symbol),
                    resource_manager=ContextResourceManager()))
            return None, worker_fn, None

        elif self.backend in ['mp_spawn', 'concurrent_mp_spawn']:  # noqa

            mpctx_str = self.backend.split('_')[-1]
            mpctx = multiprocessing.get_context(mpctx_str)

            def _get_initargs(idx):
                initargs = [
                    idx, self.num_worker,
                    self.symbol.dumps(),
                    dill.dumps(get_current_resource_manager()) if mpctx_str == 'spawn' else None,  # noqa
                    get_serializer(return_name=True) if mpctx_str == 'spawn' else None,  # noqa
                ]
                return initargs

            if self.backend in ['mp_spawn']:
                worker_pools = []
                for idx in range(self.num_worker):
                    worker_pools.append(mpctx.Pool(
                        1, initializer=_worker_initializer,
                        initargs=_get_initargs(idx)
                    ))
                client = MultiProcessingClient(worker_pools)
                return client, _mp_worker_fn, lambda: close_worker_pools(worker_pools)  # noqa

            else:
                worker_pools = []
                for idx in range(self.num_worker):
                    worker_pools.append(ProcessPoolExecutor(
                        1, mp_context=mpctx,
                        initializer=_worker_initializer,
                        initargs=_get_initargs(idx)
                    ))
                client = ConcurrnetPoolExecutorClient(worker_pools)

                def _callback():
                    for worker_i in worker_pools:
                        worker_i.shutdown()

                return client, _mp_worker_fn, _callback

        elif self.backend in ['dask']:
            assert DaskClient is not None, \
                'Please install the dask, distributed, dask_mpi package'

            import dask

            if dask.config.get("scheduler-address", None) is not None:
                if self.num_worker is not None:
                    logger.info('Ignore num_worker number since scheduler address is set!')  # noqa
                client = DaskClient()
            else:
                client = DaskClient(n_workers=self.num_worker)
            key = 'worker_initializer'

            print("\n\n\n\nRegister Worker\n\n\n\n")

            client.register_worker_plugin(
                _DaskWorkerInitializer(
                    serializer=get_serializer(return_name=True),
                    symbol_buf=self.symbol.dumps(),
                    resource_manager_buf=dill.dumps(get_current_resource_manager()),  # noqa
                    num_worker=client.num_workers,
                    with_lock=True
                ),
                key
            )
            worker_fn = functools.partial(_dask_worker_fn, key=key)

            def _callback():
                # some old version of dask does not have this method
                if hasattr(client, 'unregister_worker_plugin'):
                    client.unregister_worker_plugin(key)
                client.__del__()

            return client, worker_fn, _callback

        else:
            raise ValueError(f'Invalid backend {self.backend}')

    def __call__(self, inputs):
        assert isinstance(inputs, dict)

        input_iter = _InputIter(
            inputs=inputs, partition_input_keys=self.partition_input_keys,
            partition_input_as_iterable=self.partition_input_as_iterable,
            batch_size=self.batch_size)

        results = [[] for _ in range(self.num_outputs)]

        client, worker_fn, done_callback = \
            self._get_client_and_worker_fn_and_done_callback()

        processor_iter = MultiWorkerIter(
            input_iter=input_iter, worker_fn=worker_fn,
            client=client, max_prefetch=None)

        tic = time.time()
        idx = 0
        try:
            for idx, ret in enumerate(processor_iter):
                if self.num_outputs == 1:
                    ret = [ret, ]
                has_unset = any([ret_i is OpSkipFlag for ret_i in ret])
                if has_unset:
                    if self.skip_unset_output:
                        logger.info(f'Skip {idx+1}-th sample, because some outputs is empty')  # noqa
                        continue
                    else:
                        logger.warning(f'{idx+1}-th sample has some outputs not set, please make sure you really want it')  # noqa
                for all_ret_i, ret_i, reducer_i in zip(results, ret, self.reducers):  # noqa
                    all_ret_i = reducer_i(all_ret_i, ret_i)
                if (idx + 1) % self.log_interval == 0:
                    time_cost = time.time() - tic
                    speed = round((idx + 1) / time_cost, 2)
                    logger.info(f'Processed {idx+1}-th samples, time cost {time_cost}s, speed: {speed} samples/s')  # noqa
        except Exception:
            if done_callback is not None:
                done_callback()
            logger.error(f"Failed when executing {idx+1}-th sample")
            raise
        finally:
            if done_callback is not None:
                done_callback()

        return results[0] if self.num_outputs == 1 else results
