# Copyright (c) Changan Auto. All rights reserved.
"""perf dataloader speed tool.

Usage:

>> python tools/perf_dataloader.py --config tests/data/toy_multitask/multitask.py --step float --iter-nums 100 --allow-all-rank

"""  # noqa
import argparse
import logging
import pprint
from typing import Sequence, Union

import torch
import torch.backends.cudnn as cudnn

from cap.engine import build_launcher
from cap.profiler.dataloader_speed_perf import DataloaderSpeedPerf
from cap.registry import RegistryContext, build_from_registry
from cap.utils.config import Config, filter_configs
from cap.utils.distributed import get_dist_info, rank_zero_only
from cap.utils.logger import DisableLogger, MSGColor, format_msg
from cap.utils.seed import seed_everything
from trainer_wrapper import TRAINING_STEPS
from utilities import init_rank_logger, set_cap_env

LOG_DIR = ".cap_logs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        help="training step in: %s" % TRAINING_STEPS,
        choices=TRAINING_STEPS,
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
        help="dist url for init process",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["torch", "mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
        help="export CAP_PIPELINE_TEST=1, which used in config",
    )
    parser.add_argument(
        "--iter-nums",
        default=2000,
        type=int,
        help="perf dataloader iteration nums",
    )
    parser.add_argument(
        "--frequent", default=10, type=int, help="perf dataloader frequent"
    )
    parser.add_argument(
        "--allow-all-rank",
        action="store_true",
        default=False,
        help="allow all rank process to show perf result",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        default=False,
        help="skip transform when perf dataloader",
    )

    return parser.parse_args()


def seed_training(seed: int, logger: logging.Logger):  # noqa: D205,D400
    """
    Set seed for pseudo-random number generators in:
    pytorch, numpy, python.random, set cudnn state as well.

    Args:
        seed: the integer value seed for global random state.
        logger: Logger instance.
    """
    seed_everything(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    rank_zero_only(
        logger.warning(
            format_msg(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic and turn off CUDNN "
                "benchmark, which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints.",
                MSGColor.RED,
            )
        )
    )


def remove_transform(cfg, logger):
    """Remove transform in cfg."""
    if cfg["type"] == "MultitaskLoader":
        for _task, loader in cfg["loaders"].items():
            assert loader["type"] == torch.utils.data.dataloader.DataLoader
            remove_transform(loader, logger)
    elif cfg["type"] == torch.utils.data.dataloader.DataLoader:
        dataset = cfg["dataset"]
        remove_transform(dataset, logger)
    elif cfg["type"] == "ConcatDataset":
        assert "datasets" in cfg
        for dataset_i in cfg["datasets"]:
            remove_transform(dataset_i, logger)
    elif cfg["type"] == "ComposeDataset":
        assert "datasets" in cfg
        for dataset_i in cfg["datasets"]:
            remove_transform(dataset_i, logger)
    elif "transform" in cfg:
        logger.info(
            format_msg("remove transform: ", MSGColor.GREEN)
            + f"{cfg['transform']}"
        )  # noqa
        cfg.pop("transform")
    elif "transforms" in cfg:
        logger.info(
            format_msg("remove transform: ", MSGColor.GREEN)
            + f"{cfg['transforms']}"
        )  # noqa
        cfg.pop("transforms")
    else:
        print("no transform found", cfg)


def main(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    step: str,
    iter_nums: int,
    frequent: int,
    allow_all_rank: bool,
    skip_transform: bool,
):  # noqa: D205,D400
    """
    Args:
        device: run on cpu (if None), or gpu (list of gpu ids)
        cfg_file: Config file used to build log file name.
        step: Current training step used to build log file name.
        iter_nums: Iteration nums to perf dataloader.
        frequent: Log frequent.
        allow_all_rank: Allow all rank process to show perf result.
        skip_transform: Skip transform when perf dataloader.
    """
    cfg = Config.fromfile(cfg_file)

    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)
    if allow_all_rank:
        disable_logger = False

    # 1. init logger
    logger = init_rank_logger(
        rank, save_dir=LOG_DIR, cfg_file=cfg_file, step=step, prefix="train-"
    )
    logger.info("=" * 50 + "BEGIN %s STAGE" % step.upper() + "=" * 50)

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`disable_current_rank_logger` in config if you don't want this.",  # noqa: E501
                MSGColor.GREEN,
            )
        )
    else:
        logger.info(pprint.pformat(filter_configs(cfg)))

    # 2. seed training
    cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        seed_training(cfg.seed, logger)

    # 3. build and run trainer
    with DisableLogger(disable_logger), RegistryContext():
        trainer_cfg = cfg.step2solver[step].trainer
        assert "data_loader" in trainer_cfg

        if skip_transform:
            remove_transform(trainer_cfg["data_loader"], logger)
            logger.info(
                format_msg("perf dataloader without transform", MSGColor.GREEN)
            )
        data_loader = build_from_registry(trainer_cfg["data_loader"])

        msg = "Run in perf dataloader speed mode"
        logger.info(format_msg(msg, MSGColor.GREEN))

        DataloaderSpeedPerf(
            data_loader, iter_nums=iter_nums, frequent=frequent
        ).run()

    logger.info("=" * 50 + "END Perf Dataloader" + "=" * 50)


if __name__ == "__main__":
    args = parse_args()
    set_cap_env(args.step, args.pipeline_test)

    config = Config.fromfile(args.config)

    if args.device_ids is not None:
        ids = list(map(int, args.device_ids.split(",")))
    else:
        ids = config.device_ids

    trainer_config = config.step2solver[args.step]["trainer"]
    launch = build_launcher(trainer_config)
    launch(
        main,
        ids,
        dist_url=args.dist_url,
        dist_launcher=args.launcher,
        args=(
            args.config,
            args.step,
            args.iter_nums,
            args.frequent,
            args.allow_all_rank,
            args.skip_transform,
        ),
    )
