from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Dask, "map"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.DaskOnRay, "map"))
def dask_op(data, func):

    import dask.bag

    assert isinstance(data, dask.bag.Bag), bag

    return data.map(func)
