try:
    from dask.distributed import (
        Client as _DaskClient
    )
except ImportError:
    _DaskClient = None
from capbc.env import is_launch_by_mpi
from .base import ClientMixin


__all__ = ['DaskClient', 'get_worker_rank']


if _DaskClient is not None:

    class DaskClient(_DaskClient, ClientMixin):
        """
        Extend :py:class:`distributed.Client` to have the interface
        `num_workers`
        """

        @property
        def num_workers(self):
            return len(self._scheduler_identity['workers'])

else:

    DaskClient = None


def get_worker_rank(worker):
    """
    Get the dask worker rank.
    """
    if is_launch_by_mpi():
        # rank 0 for scheduler, rank 1 for client
        return int(worker.name) - 2
    else:
        return int(worker.name)
