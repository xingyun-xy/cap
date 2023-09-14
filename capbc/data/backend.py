from enum import Enum, unique


__all__ = ["Backend", ]


@unique
class Backend(Enum):
    Spark = "spark"
    SparkOnRay = "spark_on_ray"
    Dask = "dask"
    DaskOnRay = "dask_on_ray"
    Ray = "ray"
