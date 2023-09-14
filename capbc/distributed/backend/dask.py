import atexit
from tornado import gen
from dask.distributed import Client
from dask_mpi.core import send_close_signal as _send_close_signal


async def stop(dask_scheduler):
    await dask_scheduler.close()
    await gen.sleep(0.1)
    local_loop = dask_scheduler.loop
    local_loop.add_callback(local_loop.stop)


def send_close_signal():
    with Client() as c:
        c.run_on_scheduler(stop, wait=False)


def register_at_exit():
    atexit.unregister(_send_close_signal)
    atexit.register(send_close_signal)
