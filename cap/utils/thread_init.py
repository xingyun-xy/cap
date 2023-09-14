import logging
import os

import cv2
import torch

__all__ = ["init_num_threads"]
logger = logging.getLogger(__name__)


def init_num_threads():  # noqa: D205,D400
    """Init num threads to cv2, mkl, openmp, openblas
    and torch according to environment variables.
    """

    module_name_lists = [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
    ]

    # init module num threads
    for module_name in module_name_lists:
        num_threads = os.environ.get(module_name)
        if num_threads is not None:
            os.environ[module_name] = num_threads

    # init cv2 num threads
    opencv_num_threads = os.environ.get("OPENCV_NUM_THREADS")
    if opencv_num_threads is not None:
        cv2.setNumThreads(int(opencv_num_threads))

    # init torch num threads
    torch_num_threads = os.environ.get("TORCH_NUM_THREADS")
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))

    logger.info(
        f"init torch_num_thread is `{torch.get_num_threads()}`,"
        f"opencv_num_thread is `{cv2.getNumThreads()}`,"
        f"openblas_num_thread is `{os.environ.get('OPENBLAS_NUM_THREADS')}`,"
        f"mkl_num_thread is `{os.environ.get('MKL_NUM_THREADS')}`,"
        f"omp_num_thread is `{os.environ.get('OMP_NUM_THREADS')}`,"
    )
