import os


__all__ = ['is_launch_by_mpi']


def is_launch_by_mpi():
    flag = 'OMPI_COMM_WORLD_SIZE' in os.environ and 'OMPI_COMM_WORLD_RANK' in os.environ  # noqa
    if not flag:
        flag = 'MV2_COMM_WORLD_SIZE' in os.environ and 'MV2_COMM_WORLD_RANK' in os.environ  # noqa
    return flag
