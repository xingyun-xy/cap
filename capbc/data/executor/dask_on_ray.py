from capbc.data.backend import Backend
from capbc.data.executor.base import EXECUTOR_REGISTRY
from capbc.data.executor.dask import DaskExecutor
from capbc.data.executor.utils import initialize_dask_on_ray


@EXECUTOR_REGISTRY.register(name=Backend.DaskOnRay)
class DaskOnRayExecutor(DaskExecutor):

    def _post_init(self):

        initialize_dask_on_ray(
            address=self.address,
            num_workers=self.num_workers,
        )

        super()._post_init()
