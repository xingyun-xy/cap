import abc
import logging
import atexit
from enum import Enum
import time
from capbc.env import is_launch_by_mpi


__all__ = ['init_dist', 'is_initialized', 'BackendEnum']


logger = logging.getLogger(__name__)


class BackendEnum:
    """Distributed backend enumeration"""
    kDaskMPI = 'dask_mpi'


_dist_env = set()


def _check_initialized(backend):
    assert is_initialized(backend), \
        f'Please initialize backend {backend} via init_dist first'


def is_initialized(backend):
    return backend in _dist_env


def _init_dask_mpi_dist_env(*args, memory_limit=1.0, **kwargs):
    assert is_launch_by_mpi(), 'You can call this api only when the program is launched by mpi'  # noqa
    from dask_mpi import initialize
    from .backend.dask import register_at_exit
    initialize(*args, **kwargs, memory_limit=memory_limit)
    register_at_exit()
    time.sleep(10)


def init_dist(backend, *args, **kwargs):
    """
    Initialize the distributed environment.

    Parameters
    ----------
    backend : str
        Distributed backend, possible values are {dask_mpi, }
    args : tuple
        Args for the backend initializer
    kwargs : dict
        Args for the backend initializer
    """

    global _dist_env

    if backend in _dist_env:
        logger.warning('Distributed environment have been initialized, ignored...')  # noqa
        return True

    if backend == BackendEnum.kDaskMPI:
        _init_dask_mpi_dist_env(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported backend type {backend}')

    _dist_env.add(backend)

    return True
