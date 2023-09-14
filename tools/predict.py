"""predict tools."""
import sys
sys.path.append('/workspace/cap_develop/')

import argparse
import os
import warnings
from typing import Sequence, Union
import time


import changan_plugin_pytorch as changan
import torch

from cap.engine import build_launcher
from cap.registry import RegistryContext, build_from_registry
from cap.utils.config import ConfigVersion
from cap.utils.config_v2 import Config
from cap.utils.distributed import get_dist_info
from cap.utils.logger import (
    DisableLogger,
    MSGColor,
    format_msg,
    rank_zero_info,
)
from utilities import LOG_DIR, init_rank_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        required=True,
        help=(
            "the predict stage, you should define "
            "{stage}_predictor in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default=None,
        help="GPU device ids like '0,1,2,3', "
        "will override `device_ids` in config",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="NCCL",
        choices=["NCCL", "GLOO"],
        help="dist url for init process",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    return parser.parse_args()


def predict_entrance(
    device: Union[None, int, Sequence[int]],
    stage: str,
    cfg_file: str,
):
    cfg = Config.fromfile(cfg_file)
    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step="pred",
        prefix="predict-",
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)

    rank_zero_info("=" * 50 + "BEGIN PREDICT" + "=" * 50)
    changan.march.set_march(cfg.get("march", changan.march.March.BAYES))

    # build model
    assert hasattr(
        cfg, f"{stage}_predictor"
    ), f"you should define {stage}_predictor in the config file"
    predictor = cfg[f"{stage}_predictor"]

    with DisableLogger(disable_logger), RegistryContext():
        predictor["device"] = device
        predictor = build_from_registry(predictor)
        start_time = time.time()
        predictor.fit()
        end_time = time.time()

    rank_zero_info("=" * 50 + "END PREDICT" + "=" * 50)
    rank_zero_info("=" * 39 + "PREDICT TIME = " + str((end_time - start_time) * 1000) + "ms" + "=" * 39)


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)
    # check config version
    config_version = config.get("VERSION", None)
    if config_version is not None:
        assert (
            config_version == ConfigVersion.v2
        ), "{} only support config with version 2, not version {}".format(
            os.path.basename(__file__), config_version.value
        )
    else:
        warnings.warn(
            "VERSION will must set in config in the future."
            "You can refer to configs/classification/resnet18.py,"
            "and configs/classification/bernoulli/mobilenetv1.py."
        )
    if args.device_ids is not None:
        ids = list(map(int, args.device_ids.split(",")))
    else:
        ids = config.device_ids
    num_processes = config.get("num_processes", None)

    predictor_cfg = config[f"{args.stage}_predictor"]
    launch = build_launcher(predictor_cfg)
    launch(
        predict_entrance,
        ids,
        dist_url=args.dist_url,
        dist_launcher=args.launcher,
        num_processes=num_processes,
        backend=args.backend,
        args=(args.stage, args.config),
    )
