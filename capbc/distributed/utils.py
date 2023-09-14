from capbc.utils import _as_list


__all__ = ['close_worker_pools', ]


def close_worker_pools(worker_pools):
    """
    Close multiple :py:class:`multiprocessing.pool.Pool`

    Parameters
    ----------
    worker_pools : list/tuple of :py:class:`multiprocessing.pool.Pool`
    """

    for pool_i in _as_list(worker_pools):

        pool_i.close()
        pool_i.join()
        pool_i.terminate()
