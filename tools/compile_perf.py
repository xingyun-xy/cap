# Copyright (c) Changan Auto. All rights reserved.
import argparse
import copy
import logging
import os
import pprint
from typing import Optional

import changan_plugin_pytorch as changan
from easydict import EasyDict
from hbdk import hbir_base
from changan_plugin_pytorch.quantization import perf_model

from cap.registry import RegistryContext
from cap.utils.apply_func import _as_list
from cap.utils.config import Config
from cap.utils.distributed import rank_zero_only
from cap.utils.hash import generate_sha256_file
from cap.utils.logger import MSGColor, format_msg
from trainer_wrapper import INT_INFERENCE_STEP, TrainerWrapper
from utilities import set_cap_env

logger = logging.getLogger(__file__)


def compile_then_perf(
    cfg_file: str,
    random_params: bool,
    out_dir: Optional[str] = None,
    opt: Optional[int] = None,
    jobs: Optional[int] = 0,
):  # noqa: D205,D400
    """Compile deploy_model of step `int_infer` then test performance of it,
    `.hbm` and other performance file like `.json` will save.

    Args:
        cfg_file: Config file name.
        random_params: Use random params for compile.
        out_dir: Directory to hold performance files.
            If not None, will override `config.compile_cfg["out_dir"]`.
            If None, no-op.
        opt:
            If not None, will override `config.compile_cfg["opt"]`.
            If None, no-op.
        jobs: Number of threads launched during compiler optimization.
            Default 0 means to use all available hardware concurrency.
    """
    cfg = Config.fromfile(cfg_file)

    # check BPU march
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

    with RegistryContext():
        trainer = TrainerWrapper(
            cfg=cfg, train_step=INT_INFERENCE_STEP, logger=logger
        )

    # qat (quantization-aware training) model is not int model, only use for
    # training
    logger.info("Building deploy_model may take some time, be patient.")
    int_deploy_model = trainer.deploy_model
    deploy_inputs = trainer.deploy_inputs
    # int_deploy_model.load_state_dict(trainer.model.state_dict(), strict=True)
    if not random_params:
        int_deploy_model.load_state_dict(
            trainer.model.state_dict(), strict=True
        )

    # have a test
    int_deploy_model(deploy_inputs)

    # override default compile config
    compile_cfg = EasyDict(
        copy.deepcopy(trainer.compile_cfg)
        if trainer.compile_cfg is not None
        else {}
    )

    if out_dir is not None:
        compile_cfg["out_dir"] = out_dir
        compile_cfg["hbm"] = os.path.join(compile_cfg["out_dir"], "model.hbm")
    if opt is not None:
        compile_cfg["opt"] = opt

    if compile_cfg["out_dir"] is None:
        compile_cfg["out_dir"] = "."
        compile_cfg["hbm"] = os.path.join(compile_cfg["out_dir"], "model.hbm")
    if compile_cfg["hbm"] is None:
        cfg_name = os.path.splitext(os.path.basename(cfg_file))[0]
        compile_cfg["hbm"] = os.path.join(
            compile_cfg["out_dir"], "%s-deploy_model.hbm" % cfg_name
        )
    compile_cfg["jobs"] = jobs

    if not os.path.exists(compile_cfg["out_dir"]):
        os.makedirs(compile_cfg["out_dir"])

    logger.info("Compile config:\n" + pprint.pformat(compile_cfg))

    # compile, perf
    # wrap dict, tensor as list
    example_inputs = tuple(_as_list(deploy_inputs))
    # TODO(xuefang.wang, 0.2), show trace fail reason #
    hbir_base.CleanUpContext()
    result = perf_model(
        module=int_deploy_model.eval(),
        example_inputs=example_inputs,
        **compile_cfg,
    )
    hashed_hbm_file = generate_sha256_file(compile_cfg["hbm"], remove_old=True)
    if isinstance(result, dict):
        logger.info("Perf details:\n" + pprint.pformat(result))
        logger.info(
            format_msg("Compiled model: %s" % hashed_hbm_file, MSGColor.GREEN)
        )
        logger.info(
            format_msg(
                "Performance results saved at: %s" % compile_cfg["out_dir"],
                MSGColor.GREEN,
            )
        )
    else:
        assert result == 0, (
            "Compile or perf deploy_model fail in step %s" % INT_INFERENCE_STEP
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--random-params",
        action="store_true",
        default=False,
        help="use random params for compile",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        default=None,
        help="directory to hold perf results like `.hbm`, `.json`, will "
        'override config.compile_cfg["out_dir"]',
    )
    parser.add_argument(
        "--opt",
        type=int,
        required=False,
        default=None,
        help='optimization options, will override config.compile_cfg["opt"], '
        "the bigger the compile speed is the slower, "
        "set `0` to quickly debug. ",
        choices=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--jobs",
        type=int,
        required=False,
        default=4,
        help="number of threads launched during compiler optimization."
        " Default 0 means to use all available hardware concurrency.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cap_env(step=INT_INFERENCE_STEP)
    compile_then_perf(
        cfg_file=args.config,
        random_params=args.random_params,
        out_dir=args.out_dir,
        opt=args.opt,
        jobs=args.jobs,
    )
