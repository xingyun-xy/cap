from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


@BACKEND_OP_REGISTRY.register(name=(
    get_backend_op_register_name(Backend.Dask, "reduction")))
@BACKEND_OP_REGISTRY.register(name=(
    get_backend_op_register_name(Backend.DaskOnRay, "reduction")))
def dask_op(data, func, aggregate_func=None):

    import dask
    import dask.bag

    assert isinstance(data, dask.bag.Bag)

    if not aggregate_func:
        aggregate_func = func

    return data.reduction(func, aggregate_func)
