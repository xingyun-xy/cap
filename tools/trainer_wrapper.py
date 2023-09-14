import logging
from typing import List, Optional, Sequence, Union

import changan_plugin_pytorch as changan
import torch
import torch.nn as nn
from capbc.utils import deprecated_warning
from changan_plugin_pytorch.quantization import check_model

from cap.models.transforms import qat_fuse_bn_by_patterns
from cap.registry import build_from_registry
from cap.utils import qconfig_manager
from cap.utils.checkpoint import (
    load_checkpoint,
    load_state_dict,
    update_state_dict_by_strip_prefix,
)
from cap.utils.config import Config
from cap.utils.distributed import get_device_count, rank_zero_only
from cap.utils.logger import MSGColor, format_msg

__all__ = [
    "TrainerWrapper",
    "FLOAT_STEPS",
    "CALIBRATION_STEP",
    "QAT_STEPS",
    "INT_INFERENCE_STEP",
    "TRAINING_STEPS",
]

FLOAT_STEPS = ["float"]
FLOAT_FREEZE_BN_STEPS = [
    "with_bn",
    "float_freeze_bn",
    "freeze_bn_1",
    "freeze_bn_2",
    "freeze_bn_3",
    "sparse_3d_freeze_bn_1",
    "sparse_3d_freeze_bn_2",
]
CALIBRATION_STEP = ["calibration"]
QAT_STEPS = ["qat"]
INT_INFERENCE_STEP = "int_infer"
TRAINING_STEPS = (FLOAT_STEPS + FLOAT_FREEZE_BN_STEPS + CALIBRATION_STEP +
                  QAT_STEPS + [INT_INFERENCE_STEP])


def build_quantize_check(
    model: nn.Module,
    logger: logging.Logger,
    quantize: Optional[bool] = False,
    trace_then_check: Optional[bool] = False,
    build_calibration: Optional[bool] = False,
    trace_inputs: Optional[Union[torch.Tensor, tuple, dict]] = None,
    qconfig_params: Optional[dict] = None,
    qat_fuse_patterns: Optional[List] = None,
) -> nn.Module:
    """Build model, then quantize and check whether it can run on BPU.

    Args:
        model: Model config.
        logger: Logger instance.
        quantize: Whether convert float model to qat
            (quantization-aware training) model.
        build_calibration: Weather to build calibration model.
        trace_then_check: Whether check the traced quantized model.
        trace_inputs: Example inputs used in :func:`torch.jit.trace`.
            Refer :func:`torch.jit.trace` for more.

    Returns:
        The qat (quantization-aware training) model, only used for training,
        for bpu deploy, you have to convert it by:
        `changan.quantization.convert(qat_model.eval(), inplace=False)`
    """
    if not quantize:
        trace_then_check = False

    # is `float` model now
    model = build_from_registry(model)

    if build_calibration:
        assert (not quantize
                ), "`quantize` should be False when build calibration model"

        model.fuse_model()
        qconfig_manager.set_default_qconfig(**qconfig_params)
        model.qconfig = qconfig_manager.get_default_calibration_qconfig()
        if hasattr(model, "set_calibration_qconfig"):
            model.set_calibration_qconfig()
        changan.quantization.prepare_calibration(model, inplace=True)

    if quantize:
        # convert `float` to `qat` (quantization-aware training) model,
        # but not int model
        model.fuse_model()
        qconfig_manager.set_default_qconfig(**qconfig_params)
        model.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(model, "set_qconfig"):
            model.set_qconfig()
        else:
            raise RuntimeError("`model` should implement `set_qconfig()`")
        changan.quantization.prepare_qat(model, inplace=True)
        if qat_fuse_patterns:
            qat_fuse_bn_by_patterns(model,
                                    qat_fuse_patterns,
                                    regex=True,
                                    strict=False)

    @rank_zero_only
    def _check():
        assert trace_inputs is not None
        # convert `qat` to `int` model
        int_model = changan.quantization.convert(model.eval(), inplace=False)
        flag = check_model(int_model, trace_inputs, advice=10)
        if flag != 0:
            raise AssertionError("Failed to pass hbdk checker")

    # trace model, then check whether can run on BPU
    if trace_then_check:
        # checking is slow, so only do check in rank 0
        _check()

    return model


class TrainerWrapper(object):
    """TrainerWrapper connects `Config` with `Trainer`.

    it does following:
    (1) Parse config to build trainer.
    (2) Wrap trainer functions like `fit`.
    (3) Provide access to config objects like `model`, `deploy_model`.

    Args:
        cfg: Config instance.
        train_step: Training step in `TRAINING_STEPS`.
        logger: Logger instance.
        device: One or a sequence of GPU device ids.
            If None, Trainer run on cpu.
    """

    def __init__(
        self,
        cfg: Config,
        train_step: str,
        logger: logging.Logger,
        device: Optional[Union[int, Sequence[int]]] = None,
    ):
        assert (
            train_step in TRAINING_STEPS
        ), "step `%s` not in %s,  register in TRAINING_STEPS first." % (
            train_step,
            TRAINING_STEPS,
        )

        if train_step == INT_INFERENCE_STEP:
            assert cfg.step2solver[train_step]["quantize"], (
                "models of step `%s` should be quantized" % train_step)

        self.cfg = cfg
        self.device = device
        self.train_step = train_step
        self.logger = logger

        self.step2solver = self.cfg.step2solver
        self.solver = self.step2solver[train_step]
        self.trainer_cfg = self.solver["trainer"]
        assert self.trainer_cfg is not None, ("trainer of step `%s` is None" %
                                              self.train_step)

        self.quantize = self.solver.get("quantize", False)
        self.check_quantize_model = self.solver.get("check_quantize_model",
                                                    False)
        self.compile_cfg = self.cfg.get("compile_cfg", None)
        self.is_calibration = self.train_step in CALIBRATION_STEP
        self.pre_fuse_patterns = self.solver.get("pre_fuse_patterns", [])
        self.qat_fuse_patterns = self.solver.get("qat_fuse_patterns", [])
        self.qconfig_params = self.solver.get("qconfig_params", {})

        if self.is_calibration and get_device_count() > 1:
            msg = "`Calibrator` only support running one GPU while using MPI."
            self.logger.warning(format_msg(msg, MSGColor.RED))

        self._model = None
        self._val_model = None
        self._deploy_model = None
        self._deploy_inputs = None
        self._trainer = None

    def _modify_optimizer_cfg(self, optimizer, model=None):
        if isinstance(optimizer, dict) and "model" not in optimizer:
            assert model is not None
            optimizer["model"] = model
        return optimizer

    def prepare_fit(
        self,
        export_ckpt_only: Optional[bool] = False,
        val_only: Optional[bool] = False,
        val_ckpt: Optional[str] = None,
    ):
        """Init trainer.

        Manually run `prepare_fit`, because sometimes we don't need trainer,
        i.e. not need `fit()`.

        Args:
            export_ckpt_only: Whether Skip training and export checkpoint only
                (by running `Checkpoint.on_epoch_end()`).
            val_only: Whether skip training and do validation only (by running
                `Validation.on_epoch_end()`)
            val_ckpt: Validation checkpoint, default is None.
        """
        assert self._trainer is None, "trainer has initialized"

        skip_training = export_ckpt_only or val_only
        if skip_training:
            deprecated_warning(
                "The parameter of export_ckpt_only and val_only will be "
                "removed from tools/trainer_warpper.py and be achieved in "
                "tool/predict.py in the future.")

        # 1. override trainer config
        trainer_cfg = self.trainer_cfg
        trainer_cfg["device"] = self.device

        # 2. build model/val_model/deploy_model
        trainer_cfg["model"] = self.model

        # 5. build trainer
        if "optimizer" in trainer_cfg:
            trainer_cfg["optimizer"] = self._modify_optimizer_cfg(
                trainer_cfg["optimizer"], trainer_cfg["model"])

        # 6. build task sampler
        if ("task_sampler" in trainer_cfg
                and trainer_cfg["task_sampler"] is not None):
            trainer_cfg["task_sampler"] = build_from_registry(
                trainer_cfg["task_sampler"])

        if "data_loader" not in trainer_cfg:
            trainer_cfg["data_loader"] = None

        self._trainer = build_from_registry(trainer_cfg)

    def fit(self):
        assert self._trainer is not None, "run `prepare_fit` first"
        self._trainer.fit()

    def build_model(self):
        model = self.model_cfg
        if model is None:
            return None

        pre_step_ckpt = self.solver.get("pre_step_checkpoint", None)
        resume_ckpt = self.solver.get("resume_checkpoint", None)
        pretrain_ckpt = self.solver.get("pretrain_checkpoint", None)
        allow_miss = self.solver.get("allow_miss", False)
        ignore_extra = self.solver.get("ignore_extra", False)
        verbose = self.solver.get("verbose", 0)
        state_dict_update_func = self.solver.get(
            "state_dict_update_func", update_state_dict_by_strip_prefix)
        check_ckpt_hash = self.solver.get("check_hash", True)
        # TODO(xuefang.wang): remove in v1.0 #
        if "load_checkpoint_func" in self.solver:
            self.logger.error("load_checkpoint_func is deprecarted, "
                              "use state_dict_update_func instead")

        # 1. build model
        self.logger.info("building train model ...")

        qat_fuse_patterns = self.qat_fuse_patterns + self.pre_fuse_patterns
        # 2. init
        # NOTE: resume_checkpoint > pre_step_checkpoint > pretrain
        if resume_ckpt is not None:
            model = build_quantize_check(
                model,
                logger=self.logger,
                quantize=self.quantize,
                # TODO(xuefang.wang, 0.2) whether check model?
                trace_then_check=False,
                build_calibration=self.is_calibration,
                qconfig_params=self.qconfig_params,
                qat_fuse_patterns=qat_fuse_patterns,
            )

            resume_optimizer = self.solver.get("resume_optimizer", True)
            # deal with, `resume_epoch` and `resume_step`
            # TODO (xuefang.wang): modify next version
            if "resume_epoch" in self.solver and "resume_step" in self.solver:
                resume_epoch = self.solver.get("resume_epoch")
                resume_step = self.solver.get("resume_step")
                deprecated_warning(
                    "The keys `resume_epoch` and `resume_step` for resume "
                    "training will be deprecated, please use "
                    "`resume_epoch_or_step` instead, and see details from: ")
                if resume_epoch or resume_step:
                    resume_epoch_or_step = True
                else:
                    resume_epoch_or_step = False
            else:
                resume_epoch_or_step = self.solver.get("resume_epoch_or_step",
                                                       True)

            if resume_optimizer and not resume_epoch_or_step:
                raise ValueError("`resume_epoch_or_step` should be False when "
                                 "`resume_optimizer=False`, but get True.")

            if pre_step_ckpt is not None:
                self.logger.info(
                    format_msg(
                        "`pre_step_checkpoint` is unused when "
                        "`resume_checkpoint` is specific",
                        MSGColor.RED,
                    ))

            self.logger.info(
                format_msg(
                    "init train model with checkpoint: %s" % resume_ckpt,
                    MSGColor.GREEN,
                ))

            ckpt_dict = load_checkpoint(
                resume_ckpt,
                map_location="cpu",
                state_dict_update_func=state_dict_update_func,
                check_hash=check_ckpt_hash,
            )

            model = load_state_dict(
                model,
                ckpt_dict["state_dict"],
                allow_miss=False,
                ignore_extra=False,
                verbose=verbose,
            )

            if resume_optimizer:
                # check gpu device
                # TODO (xuefang.wang): modify next version.
                previous_gpu_num = ckpt_dict.get("devices", None)
                device_num = get_device_count()
                if previous_gpu_num is None:
                    # for old checkpoint
                    self.logger.warning(
                        format_msg(
                            "The number of devices is not found in checkpoint."
                            " please ensure that the number of gpu devices is "
                            "the same as before when resuming",
                            MSGColor.RED,
                        ))
                else:
                    assert previous_gpu_num == device_num, (
                        f"The number of gpu devices should be "
                        f"{previous_gpu_num}, but get {device_num}.")

                # resume optimizer
                optimizer = self.trainer_cfg["optimizer"]
                self.logger.info(
                    format_msg(
                        "resume optimizer state from checkpoint %s" %
                        resume_ckpt,
                        MSGColor.GREEN,
                    ))
                assert (
                    optimizer is not None
                ), "resume_optimizer is True, but optimizer config is None"
                optimizer = self._modify_optimizer_cfg(optimizer, model)

                optimizer = build_from_registry(optimizer)
                # override optimizer cfg
                if "optimizer" in ckpt_dict:
                    optimizer.load_state_dict(ckpt_dict["optimizer"])

                if resume_epoch_or_step is False:
                    # for resume_optimizer only
                    for group in optimizer.param_groups:
                        assert ("lr" in group
                                ), "Not found `lr` in a optimizer.param_groups"
                        group["initial_lr"] = group["lr"]

                self.trainer_cfg["optimizer"] = optimizer

            if resume_epoch_or_step:
                stop_by = self.trainer_cfg.get("stop_by", "epoch")
                assert stop_by in [
                    "epoch",
                    "step",
                ], f"stop_by should be 'epoch' or 'step', but get {stop_by}"

                # override start_epoch and start_step
                self.trainer_cfg["start_epoch"] = (
                    ckpt_dict["epoch"] +
                    1 if stop_by.lower == "epoch" else ckpt_dict["epoch"])
                self.trainer_cfg["start_step"] = (
                    ckpt_dict["step"] +
                    1 if stop_by.lower() == "step" else ckpt_dict["step"])

                self.logger.info(
                    format_msg(
                        "reset training `start_epoch` to %d and `start_step` to %d"  # noqa: E501
                        % (
                            self.trainer_cfg["start_epoch"],
                            self.trainer_cfg["start_step"],
                        ),
                        MSGColor.GREEN,
                    ))

        elif pre_step_ckpt is not None:
            # build pre step model
            pre_step = self.solver["pre_step"]
            self.logger.info("building pre step train model ...")
            if pre_step in self.step2solver:
                pre_solver = self.step2solver[pre_step]
            else:
                pre_solver = self.solver["pre_solver"]
            is_pre_step_calibration = pre_step in CALIBRATION_STEP
            model = build_quantize_check(
                model,
                logger=self.logger,
                quantize=pre_solver["quantize"] or is_pre_step_calibration,
                trace_then_check=False,
                qconfig_params=self.qconfig_params,
                qat_fuse_patterns=self.pre_fuse_patterns,
            )
            self.logger.info(
                format_msg(
                    "init pre step (%s) train model with checkpoint: %s" %
                    (pre_step, pre_step_ckpt),
                    MSGColor.GREEN,
                ))

            pre_ckpt_dict = load_checkpoint(
                pre_step_ckpt,
                map_location="cpu",
                state_dict_update_func=state_dict_update_func,
                check_hash=check_ckpt_hash,
            )

            # for torchvision checkpoint
            if "state_dict" in pre_ckpt_dict:
                pre_ckpt_state_dict = pre_ckpt_dict["state_dict"]
            else:
                pre_ckpt_state_dict = pre_ckpt_dict

            model = load_state_dict(
                model,
                pre_ckpt_state_dict,
                allow_miss=allow_miss,
                ignore_extra=ignore_extra,
                verbose=verbose,
            )

            self.logger.info(
                format_msg(
                    "init current step (%s) train model with pre step (%s) train "  # noqa: E501
                    "model" % (self.train_step, pre_step),
                    MSGColor.GREEN,
                ))

            if self.is_calibration:
                model = build_quantize_check(
                    model,
                    logger=self.logger,
                    quantize=False,
                    build_calibration=True,
                    qconfig_params=self.qconfig_params,
                )

            if (self.quantize and not pre_solver["quantize"]
                    and not is_pre_step_calibration):
                # quantize pre_model, so that param names of `pre_model`
                # match that of `model`
                model = build_quantize_check(
                    model,
                    logger=self.logger,
                    quantize=True,
                    qconfig_params=self.qconfig_params,
                )

            if self.qat_fuse_patterns:
                model = qat_fuse_bn_by_patterns(model,
                                                self.qat_fuse_patterns,
                                                regex=True,
                                                strict=False)

        elif pretrain_ckpt is not None:
            model = build_quantize_check(
                model,
                logger=self.logger,
                quantize=self.quantize,
                trace_then_check=False,
                build_calibration=self.is_calibration,
                qconfig_params=self.qconfig_params,
                qat_fuse_patterns=qat_fuse_patterns,
            )

            self.logger.info(
                format_msg(
                    "init train model with checkpoint: %s" % pretrain_ckpt,
                    MSGColor.GREEN,
                ))

            ckpt_dict = load_checkpoint(
                pretrain_ckpt,
                map_location="cpu",
                state_dict_update_func=state_dict_update_func,
                check_hash=check_ckpt_hash,
            )

            model = load_state_dict(
                model,
                ckpt_dict["state_dict"],
                allow_miss=allow_miss,
                ignore_extra=ignore_extra,
                verbose=verbose,
            )

        else:
            model = build_quantize_check(
                model,
                logger=self.logger,
                quantize=self.quantize,
                trace_then_check=False,
                qconfig_params=self.qconfig_params,
                qat_fuse_patterns=qat_fuse_patterns,
            )
            if self.train_step not in FLOAT_STEPS:
                allow_not_init = self.solver.get("allow_not_init", False)
                assert allow_not_init, (
                    "both `pre_step_checkpoint` and `resume_checkpoint` of "
                    "step `%s` are None, if so, train model will not "
                    "initialized by any checkpoint, set allow_not_init=True "
                    "to skip this check" % self.train_step)

        if self.train_step == INT_INFERENCE_STEP:
            self.logger.info("convert qat model to int model...")
            model = changan.quantization.convert(model.eval(), inplace=True)

        return model

    def build_val_model(self):
        val_model = self.val_model_cfg
        if val_model is None:
            return None

        # Note: The val_model for the calibration step
        # will be build as a QAT.
        val_model = build_quantize_check(
            val_model,
            logger=self.logger,
            quantize=self.quantize or self.is_calibration,
            trace_then_check=False,
            qconfig_params=self.qconfig_params,
        )

        if self.train_step == INT_INFERENCE_STEP:
            self.logger.info("convert qat val_model to int val_model...")
            val_model = changan.quantization.convert(val_model.eval(),
                                                     inplace=True)

        return val_model

    def build_deploy_model(self):
        deploy_model = self.deploy_model_cfg
        if deploy_model is not None:
            check_deploy_model = self.check_quantize_model
            if check_deploy_model:
                assert self.deploy_inputs is not None, (
                    "specific `Checkpoint.deploy_inputs` in config when "
                    "check_deploy_model is True in step %s" % self.train_step)

            # self.logger.info("building deploy_model ...")
            # convert `float` model to `qat` model
            # Note: The val_model for the calibration step
            # will be build as a QAT.
            deploy_model = build_quantize_check(
                deploy_model,
                logger=self.logger,
                quantize=self.quantize or self.is_calibration,
                trace_then_check=check_deploy_model,
                trace_inputs=self.deploy_inputs,
                qconfig_params=self.qconfig_params,
            )

        if self.train_step == INT_INFERENCE_STEP:
            assert deploy_model is not None, (
                "deploy_model of step %s can't be None" % INT_INFERENCE_STEP)
            self.logger.info(
                "convert qat deploy_model to int deploy_model ...")
            deploy_model = changan.quantization.convert(deploy_model.eval(),
                                                        inplace=True)

        return deploy_model

    @property
    def model_cfg(self):
        return self.trainer_cfg["model"]

    @property
    def val_model_cfg(self):
        val_callback = self.cfg.get("val_callback", None)
        if val_callback is None:
            return None
        else:
            # val may be unnecessary, so use get
            return val_callback.get("val_model", None)

    @property
    def deploy_model_cfg(self):
        ckpt_callback = self.cfg.get("ckpt_callback", None)
        if ckpt_callback is None:
            deploy_model = None
        else:
            deploy_model = ckpt_callback.get("test_model", None)
            if deploy_model is not None:
                deprecated_warning(
                    "The key `test_model` of ckpt_callback will be "
                    "deprecated in the future, Use deploy_model instead.")
            else:
                deploy_model = ckpt_callback.get("deploy_model", None)

        # deploy_model of other steps can be None, but not int step
        if self.train_step == INT_INFERENCE_STEP:
            assert deploy_model is not None, (
                "please set `deploy_model` in callback `Checkpoint`'s config "
                "of step %s" % INT_INFERENCE_STEP)
        return deploy_model

    @property
    def model(self):
        if self._model is None:
            self.logger.info("building train model ...")
            self._model = self.build_model()
        return self._model

    @property
    def val_model(self):
        if self._val_model is None:
            self._val_model = self.build_val_model()
        return self._val_model

    @property
    def deploy_model(self):
        if self._deploy_model is None:
            self.logger.info("building deploy_model ...")
            self._deploy_model = self.build_deploy_model()
        return self._deploy_model

    @property
    def deploy_inputs(self):
        if self._deploy_inputs is None:
            ckpt_callback = self.cfg.get("ckpt_callback", None)
            if ckpt_callback is None:
                deploy_inputs = None
            else:
                deploy_inputs = ckpt_callback.get("test_inputs", None)
                if deploy_inputs is not None:
                    deprecated_warning(
                        "The key `test_inputs` of ckpt_callback will be "
                        "deprecated in the future. Use deploy_inputs instead.")
                else:
                    deploy_inputs = ckpt_callback.get("deploy_inputs", None)

            # deploy_inputs of other steps can be None, but not int step
            if self.train_step == INT_INFERENCE_STEP:
                assert deploy_inputs is not None, (
                    "please set deploy_model's tracing example inputs "
                    "`deploy_inputs` in config of step `%s`" % self.train_step)
            self._deploy_inputs = deploy_inputs
        return self._deploy_inputs
