import operator
from functools import partial

from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


def is_indexable(data):
    return hasattr(data, "__len__") and hasattr(data, "__getitem__")


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Dask, "from_sequence"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.DaskOnRay, "from_sequence"))
def dask_op(data, npartitions=1):

    import dask.bag

    if isinstance(data, (list, tuple)):
        return dask.bag.from_sequence(data, npartitions=npartitions)
    elif is_indexable(data):
        bag = dask.bag.range(len(data), npartitions)
        return bag.map(partial(operator.getitem, data))
    else:
        return dask.bag.from_sequence(data, npartitions=npartitions)
