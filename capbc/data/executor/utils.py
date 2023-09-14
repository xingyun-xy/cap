import logging
from typing import Optional

from capbc.env import is_launch_by_mpi
from capbc.registry import Registry


__all__ = ["get_dask_client", "initialize_ray", "initialize_dask_on_ray"]


logger = logging.getLogger(__name__)


_dask_client = None


def get_dask_client(address: Optional[str] = None,
                    num_workers: Optional[int] = None):

    global _dask_client

    if _dask_client is not None:
        return _dask_client

    from distributed import Client

    if address is not None:
        _dask_client = Client(address)
    elif is_launch_by_mpi():
        logger.info('Ignore num_worker number since is launched by mpi')
        _dask_client = Client()
    else:
        _dask_client = Client(n_workers=num_workers)

    return _dask_client


_ray_init = False


def initialize_ray(address=None, num_workers=None):
    global _ray_init

    if _ray_init:
        return

    import ray

    if address is not None:
        ray.init(address)
    else:
        try:
            ray.init("auto")
        except ConnectionError:
            ray.init(num_cpus=num_workers)

    _ray_init = True


_dask_on_ray_init = False


def initialize_dask_on_ray(address=None, num_workers=None):
    global _dask_on_ray_init

    if _dask_on_ray_init:
        return

    from ray.util.dask import ray_dask_get
    import dask

    initialize_ray(
        address=address,
        num_workers=num_workers
    )
    dask.config.set(scheduler=ray_dask_get)

    _dask_on_ray_init = True
