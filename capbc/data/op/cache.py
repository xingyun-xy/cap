from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Dask, "cache"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.DaskOnRay, "cache"))
def dask_op(data):
    return data.persist()
