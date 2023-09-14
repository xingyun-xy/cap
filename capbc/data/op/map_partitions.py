from capbc.data.op.base import (
    Backend,
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Dask, "map_partitions"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.DaskOnRay, "map_partitions"))
def dask_op(data, func):

    import dask.bag

    assert isinstance(data, dask.bag.Bag)

    return data.map_partitions(func)


@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.Spark, "map_partitions"))
@BACKEND_OP_REGISTRY.register(name=get_backend_op_register_name(
    Backend.SparkOnRay, "map_partitions"))
def spark_op(self, data, func):

    import pyspark

    assert isinstance(data, pyspark.RDD)

    return data.mapPartitions(identity)
