from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Dask, "filter"
))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.DaskOnRay, "filter"
))
def dask_op(data, func):

    import dask.bag

    assert isinstance(data, dask.bag.Bag)

    return data.filter(func)
