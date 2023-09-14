from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Dask, "zip"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.DaskOnRay, "zip"))
def dask_op(*data):

    import dask.bag

    assert all([isinstance(data_i, dask.bag.Bag) for data_i in data])

    return dask.bag.zip(*data)
