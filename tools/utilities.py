import logging
import os
import time
from typing import Optional

from cap.utils.logger import init_logger

__all__ = ["set_cap_env", "init_rank_logger"]


def set_cap_env(step, pipeline_test=False, val_only=False):
    os.environ["CAP_TRAINING_STEP"] = step
    os.environ["CAP_PIPELINE_TEST"] = str(int(pipeline_test))
    os.environ["CAP_VAL_ONLY"] = str(int(val_only))
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_NUM_THREADS"] = os.environ.get("TORCH_NUM_THREADS", "12")
    os.environ["OPENCV_NUM_THREADS"] = os.environ.get(
        "OPENCV_NUM_THREADS", "12"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get(
        "OPENBLAS_NUM_THREADS", "12"
    )
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "12")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "12")
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def init_rank_logger(
    rank: int,
    save_dir: str,
    cfg_file: str,
    step: str,
    prefix: Optional[str] = "",
) -> logging.Logger:
    """Init logger of specific rank.

    Args:
        rank: rank id.
        cfg_file: Config file used to build log file name.
        step: Current training step used to build log file name.
        save_dir: Directory to save log file.
        prefix: Prefix of log file.

    Returns:
        Logger.
    """
    time_stamp = time.strftime(
        "%Y%m%d%H%M%S", time.localtime(int(time.time()))
    )
    cfg_name = os.path.splitext(os.path.basename(cfg_file))[0]
    log_file = os.path.join(
        save_dir, "%s%s-%s-%s" % (prefix, cfg_name, step, time_stamp)
    )
    init_logger(log_file=log_file, rank=rank, clean_handlers=True)

    logger = logging.getLogger()
    return logger


LOG_DIR = ".cap_logs"
