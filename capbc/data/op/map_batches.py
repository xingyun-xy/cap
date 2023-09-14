from functools import partial

from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


def _op_impl(data, func, batch_size):

    outputs = []

    batches = []
    for data_i in data:
        batches.append(data_i)
        if len(batches) == batch_size:
            outputs.append(func(batches))
            batches = []

    if batches:
        outputs.append(func(batches))

    return outputs


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Dask, "map_batches"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.DaskOnRay, "map_batches"))
def dask_op(data, func, batch_size):
    import dask.bag

    assert batch_size >= 1

    assert isinstance(data, dask.bag.Bag)

    return data.map_partitions(partial(
        _op_impl, func=func, batch_size=batch_size))


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Spark, "map_batches"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.SparkOnRay, "map_batches"))
def spark_op(data, func, batch_size):

    import pyspark

    assert batch_size >= 1

    assert isinstance(data, pyspark.RDD)

    return data.mapPartitions(partial(
        _op_impl, func=func, batch_size=batch_size))
