"""train tools."""
import sys
sys.path.append('/workspace/cap_develop/')

import argparse
import ast
import json
import logging
import os
import pprint
import warnings
from typing import List, Sequence, Union
import sys
sys.path.append("/root/cap-xy/")

import changan_plugin_pytorch as changan
import torch.backends.cudnn as cudnn

from cap.engine import build_launcher
from cap.registry import RegistryContext, build_from_registry
from cap.utils.config import ConfigVersion
from cap.utils.config_v2 import Config, filter_configs
from cap.utils.distributed import get_dist_info, rank_zero_only
from cap.utils.logger import (
    DisableLogger,
    MSGColor,
    format_msg,
    rank_zero_info,
)
from cap.utils.seed import seed_training
from cap.utils.thread_init import init_num_threads
from utilities import LOG_DIR, init_rank_logger, set_cap_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        required=True,
        help=(
            "the training stage, you should define "
            "{stage}_trainer in your config"
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
        "--opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--opts-overwrite",
        type=ast.literal_eval,
        default=True,
        help="True or False, default True, "
        "Weather to overwrite existing (keys, values) in configs",
    )

    return parser.parse_args()


def train_entrance(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    cfg_opts: List,
    cfg_opts_overwrite: bool,
    stage: str,
):
    """Training entrance function for launcher.

    Args:
        device: run on cpu (if None), or gpu (list of gpu ids)
        cfg_file: Config file used to build log file name.
        cfg_opts: Custom config options from command-line.
        cfg_opts_overwrite: Weather to overwrite existing {k: v} in configs.
        stage: Current training stage used to build log file name.
    """
    cfg = Config.fromfile(cfg_file)
    if cfg_opts is not None:
        cfg.merge_from_list_or_dict(cfg_opts, overwrite=cfg_opts_overwrite)

    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step=stage,
        prefix="train-",
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    if (
        "redirect_config_logging_path" in cfg
        and cfg["redirect_config_logging_path"]
        and rank == 0
    ):
        with open(cfg["redirect_config_logging_path"], "w") as fid:
            fid.write(pprint.pformat(filter_configs(cfg)))
        rank_zero_info(
            "save config logging output to %s"
            % cfg["redirect_config_logging_path"]
        )
    else:
        rank_zero_info(pprint.pformat(filter_configs(cfg)))

    rank_zero_info("=" * 50 + "BEGIN %s STAGE" % stage.upper() + "=" * 50)
    # 2. init num threads
    init_num_threads()
    # 3. seed training
    cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        seed_training(cfg.seed)

    if "march" not in cfg:
        rank_zero_only(
            logger.warning(
                format_msg(
                    f"Please make sure the march is provided in configs. "
                    f"Defaultly use {changan.march.March.BAYES}",
                    MSGColor.RED,
                )
            )
        )
    changan.march.set_march(cfg.get("march", changan.march.March.BAYES))

    # 4. build and run trainer
    with DisableLogger(disable_logger), RegistryContext():
        trainer = getattr(cfg, f"{stage}_trainer")
        trainer["device"] = device
        trainer = build_from_registry(trainer)
        trainer.fit()

    rank_zero_info("=" * 50 + "END %s STAGE" % stage.upper() + "=" * 50)


def main():
    args = parse_args()
    set_cap_env(args.stage, args.pipeline_test, False)
    
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
    if args.opts is not None:
        config.merge_from_list_or_dict(
            args.opts, overwrite=args.opts_overwrite
        )

    if args.device_ids is not None:
        ids = list(map(int, args.device_ids.split(",")))
    else:
        ids = config.device_ids

    assert hasattr(
        config, f"{args.stage}_trainer"
    ), f"There are not {args.stage}_trainer in config"
    trainer_config = getattr(config, f"{args.stage}_trainer")
    try:
        launch = build_launcher(trainer_config)
        launch(
            train_entrance,
            ids,
            dist_url=args.dist_url, #多机多卡时用
            dist_launcher=args.launcher, #多机多卡时用
            args=(
                args.config,
                args.opts,
                args.opts_overwrite,
                args.stage,
            ),
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
