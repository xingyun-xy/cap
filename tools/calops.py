"""predict tools."""
import argparse
import os
import copy
import warnings
from typing import Sequence, Union

import changan_plugin_pytorch as changan
from changan_plugin_pytorch.utils.onnx_helper import export_to_onnx
import torch
import torch.onnx
import torch.nn as nn
from onnxsim import simplify
import onnx

from cap.engine import build_launcher
from cap.registry import RegistryContext, build_from_registry
from cap.utils.config import ConfigVersion
from cap.utils.config_v2 import Config
from cap.utils.statistics import cal_ops
from cap.utils.distributed import get_dist_info
from cap.utils.logger import (
    DisableLogger,
    MSGColor,
    format_msg,
    rank_zero_info,
)
from capbc.utils import deprecated_warning
from utilities import LOG_DIR, init_rank_logger
from cap.models.model_convert.converters import LoadCheckpoint
from cap.models.model_convert.pipelines import ModelConvertPipeline


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


def calops(
    device: Union[None, int, Sequence[int]],
    stage: str,
    cfg_file: str,
):
    cfg = Config.fromfile(cfg_file)
    # 1. build model
    model_cfg = copy.deepcopy(cfg.onnx_cfg['export_model'])
    model = build_from_registry(model_cfg)
    model.eval()

    # 2. calculate ops
    dummy_input = cfg.onnx_cfg['dummy_input']
    total_ops, total_params = cal_ops(model, dummy_input)

    print("Params: %.6f M" % (total_params / (1000 ** 2)))
    print("FLOPs: %.6f G" % (total_ops / (1000 ** 3)))

    


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
        calops,
        ids,
        dist_url=args.dist_url,
        dist_launcher=args.launcher,
        num_processes=num_processes,
        backend=args.backend,
        args=(args.stage, args.config),
    )
