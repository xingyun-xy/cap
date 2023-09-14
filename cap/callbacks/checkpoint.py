# Copyright (c) Changan Auto. All rights reserved.
import copy
import logging
import os
import pprint
import shutil
from typing import Dict, List, Optional, Tuple, Union

import changan_plugin_pytorch as changan
import numpy as np
import torch
import torch.nn as nn
from capbc.utils import deprecated_warning
from torch import Tensor

from cap.metrics import EvalMetric
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from cap.utils.distributed import (
    all_gather_object,
    get_device_count,
    get_dist_info,
    rank_zero_only,
)
from cap.utils.hash import generate_sha256_file
from .callbacks import CallbackMixin

__all__ = ["Checkpoint"]

logger = logging.getLogger(__name__)

TRAIN_CHECKPOINT_FORMAT = "%scheckpoint-%s.pth.tar"
DEPLOY_CHECKPOINT_FORMAT = "%sdeploy-checkpoint-%s.pt"


def get_valid_state_dict(model: nn.Module) -> dict:
    """Remove 'module' prefix in param name if model is ddp or dp model.

    Args:
        model: Module that need to get state dict.

    Returns:
        Dict of param names and values
    """
    if isinstance(
        model, (nn.parallel.DistributedDataParallel, nn.DataParallel)
    ):
        return model.module.state_dict()
    elif isinstance(model, nn.Module):
        ddp_module_dict = {}
        for name, module in model.named_children():
            if isinstance(module, nn.parallel.DistributedDataParallel):
                ddp_module_dict[name] = module

        if ddp_module_dict:
            # support submodule is ddp.
            model_bak = copy.deepcopy(model)
            for name, module in ddp_module_dict.items():
                setattr(model_bak, name, module.module)
            return model_bak.state_dict()
        else:
            return model.state_dict()
    else:
        raise NotImplementedError("unknown model type: %s" % type(model))


def _get_average_tensor(tensors: List[Tensor]):
    """Get average tensor of a list of tensors.

    When compute average for low precision tensor, using high precision dtype.
    """
    with torch.no_grad():
        origin_type = tensors[0].dtype
        for i, t in enumerate(tensors):
            assert (
                origin_type == t.dtype
            ), f"{i}th tensor type is {t.dtype} not {origin_type}"

        if origin_type == torch.float16:
            tmp = tensors[0].to(torch.float32)
        elif origin_type == torch.uint8:
            tmp = tensors[0].to(torch.int32)
        elif origin_type == torch.int8:
            tmp = tensors[0].to(torch.int32)
        else:
            tmp = tensors[0]

        for t in tensors[1:]:
            tmp = tmp.add(t.to(tmp.device))
        tmp = tmp / len(tensors)
        return tmp.to(origin_type)


@OBJECT_REGISTRY.register
class Checkpoint(CallbackMixin):  # noqa: D205,D400
    """
    Checkpoint Callback is used for saving model after training
    and resume model before training as the same times.

    Args:
        save_dir: Directory to save checkpoints.
        name_prefix: Checkpoint name prefix.
        save_interval: Save checkpoint every `save_interval` epoch or step.
        interval_by: Set `save_interval` unit to step or epoch.
            Default is epoch.
        save_on_train_end: Whether save checkpoint when `on_loop_end` is
            triggered.
        test_model: Test model that initialized with current train model.
            It will be trace and the checkpoints are 'test_model.pt' and
            'test_model-best.pt' (if mode is not None).
        test_inputs: Example inputs for tracing test_model.
        deploy_model: Model that initialized with current train model and
            then used to export pt file. It will be traced and the checkpoints
            are `deploy_model.pt` and `deploy_model-best.pt`.
        deploy_inputs: Example inputs for tracing deploy_model.
        allow_anno_miss: Whether allow output tensors of test_model missing
            annotation attr.
            When test_model not enable, no-op. Default True.
        strict_match: Whether to strictly enforce that the keys in
            `model.state_dict()` (train model) match the keys in
            `test_model.state_dict()`. Default: ``False``
        mode: State of monitor for saving model.
        monitor_metric_key: Monitor metric for saving best checkpoint.
        best_refer_metric: Metric that evaluate which epoch is the best.
        save_hash: Whether to save the hash value to the name of the
             Checkpoint file. Default is True.
    """

    SUPPORTED_MODES = ["min", "max"]

    def __init__(
        self,
        save_dir: str,
        name_prefix: Optional[str] = "",
        save_interval: Optional[int] = 1,
        interval_by: Optional[str] = "epoch",
        save_on_train_end: Optional[bool] = True,
        test_model: Optional[bool] = None,
        test_inputs: Optional[tuple] = None,
        deploy_model: Optional[nn.Module] = None,
        deploy_inputs: Optional[Union[Tuple, Dict]] = None,
        allow_anno_miss: Optional[bool] = True,
        strict_match: Optional[bool] = False,
        mode: Optional[str] = None,
        monitor_metric_key: Optional[str] = None,
        best_refer_metric: Optional[Union[dict, EvalMetric]] = None,
        task_sampler=None,
        save_hash: bool = True,
    ):
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.save_interval = save_interval
        self.interval_by = interval_by
        self.save_on_train_end = save_on_train_end
        self.allow_anno_miss = allow_anno_miss
        self.strict_match = strict_match
        self.task_sampler = task_sampler
        self.save_hash = save_hash

        assert self.interval_by in ("epoch", "step")

        if test_model is not None:
            deprecated_warning(
                "The parameters of test_model and test_inputs from class "
                "Checkpoint will be deprecated in the future, "
                "use deploy_model and deploy_inputs instead."
            )
            assert test_inputs is not None
            self.deploy_model = test_model
            self.deploy_inputs = test_inputs
        else:
            if deploy_model is not None:
                assert deploy_inputs is not None
                self.deploy_model = deploy_model
                self.deploy_inputs = deploy_inputs
            else:
                self.deploy_model = None
                self.deploy_inputs = None

        self.mode = mode
        self.monitor_metric_key = monitor_metric_key
        self.best_refer_metric = best_refer_metric
        if self.mode is not None:
            if self.mode == "min":
                self.monitor_op = np.less
                self.best = np.Inf
            elif self.mode == "max":
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                raise ValueError(
                    "Supported modes: %s, but get %s"
                    % (self.SUPPORTED_MODES, self.mode)
                )

    @rank_zero_only
    def save_to_file(
        self,
        state,
        epoch_id,
        save_best,
        save_epoch_or_step=True,
        save_type="epoch",
        save_last=True,
        step_id=None,
    ):

        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass

        assert save_type in [
            "epoch",
            "step",
        ], f"save_type should be one of ['epoch', 'step'], but get {save_type}"
        if save_epoch_or_step:
            ckpt_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT
                % (
                    self.name_prefix,
                    "step-%d" % step_id
                    if save_type == "step"
                    else "epoch-%04d" % epoch_id,
                ),
            )
            torch.save(state, ckpt_file)

            if self.save_hash:
                ckpt_file = generate_sha256_file(ckpt_file)
            logger.info("Save model checkpoint: %s" % ckpt_file)

        if save_last:
            last_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT % (self.name_prefix, "last"),
            )
            torch.save(state, last_file)
            if self.save_hash:
                last_file = generate_sha256_file(last_file, remove_old=True)
            logger.info("Save last model checkpoint: %s" % last_file)

        if save_best:
            best_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT % (self.name_prefix, "best"),
            )
            torch.save(state, best_file)
            if self.save_hash:
                best_file = generate_sha256_file(best_file, remove_old=True)
            logger.info("Save best model checkpoint: %s" % best_file)

    def _sync_model(self, model):
        """
        Sync model's buffers that distributed on diff host.

        Gather all buffers from diff host, then merge them together.
        When the buffer of diff host has diff values, compute average of them.
        """
        if self.task_sampler is None or not self.task_sampler.is_parallel():
            return

        if not isinstance(model, nn.parallel.DistributedDataParallel):
            return
        else:
            model = model.module
            if not hasattr(model, "named_parameters_by_outname"):
                return

        with torch.no_grad():
            tasks = self.task_sampler.tasks
            # get current task params and buffers
            states = {}
            for n, b in model.named_buffers_by_outname(tasks):
                states[n] = b

            # gather all state from every ranks
            rank, world_size = get_dist_info()
            global_states = [None for _ in range(world_size)]
            all_gather_object(global_states, states)

            merged_states = {}  # {p_name => [buffer, cnt]}
            for s in global_states:
                for k, v in s.items():
                    if k not in merged_states:
                        merged_states[k] = [v]
                    else:
                        merged_states[k].append(v)

            # check and average same tensor on different ranks
            def _merge_tensor(key, tensor):
                # check all buffer in model state_dict
                if k not in merged_states:
                    logger.warning(f"tensor: {k} not updated on training.")
                    return

                v = merged_states[k]
                if len(v) == 1:
                    # set into model state dict directly
                    v = v[0]
                else:
                    v = _get_average_tensor(v)
                if tensor.shape != v.shape:
                    tensor.resize_(v.shape)
                tensor.copy_(v)

            # check and average same buffer on different ranks
            for k, buff in model.named_buffers():
                _merge_tensor(k, buff)

    def save_checkpoint(
        self,
        model,
        epoch_id,
        optimizer,
        save_best,
        save_epoch_or_step=True,
        save_type="epoch",
        save_last=True,
        step_id=None,
    ):
        self._sync_model(model)
        state = {
            "epoch": epoch_id,
            "step": step_id,
            "devices": get_device_count(),
            "state_dict": get_valid_state_dict(model),
            "optimizer": optimizer.state_dict()
            if optimizer is not None
            else None,
        }

        self.save_to_file(
            state,
            epoch_id,
            save_best,
            save_epoch_or_step,
            save_type,
            save_last,
            step_id,
        )

    @rank_zero_only
    def trace_and_save_deploy_model(self, train_model, save_best):
        deprecated_warning(
            "trace_and_save will be deleted in the near future. "
            "Please use SaveTraced callback to save traced model "
            "instead."
        )
        # init with train model
        self.deploy_model.load_state_dict(
            get_valid_state_dict(train_model), strict=self.strict_match
        )

        self.deploy_model.cpu()
        script_module = torch.jit.trace(
            func=self.deploy_model.eval(),
            example_inputs=self.deploy_inputs,
        )

        # TODO (?), get_output_annotation supports more complicated
        #  output format.
        per_tensor_anno = changan.get_output_annotation(script_module)
        logger.info(
            "annotation str of each output tensor:\n%s"
            % pprint.pformat(per_tensor_anno)
        )
        if not self.allow_anno_miss:
            for i, anno in enumerate(per_tensor_anno):
                assert anno is not None, (
                    f"annotation of the {i}th output "
                    f"tensor is None, two reasons may cause this error:"
                    f"(1) have not set annotation for this tensor. "
                    f"(2) bug of changan.get_output_annotation(). "
                    f"you can set allow_anno_miss=True to skip this check"
                )

        pt_file = os.path.join(
            self.save_dir,
            DEPLOY_CHECKPOINT_FORMAT % (self.name_prefix, "last"),
        )

        # may override
        torch.jit.save(script_module, pt_file)
        if self.save_hash:
            pt_file = generate_sha256_file(pt_file, remove_old=True)
        logger.info("Save last traced deploy_model checkpoint: %s" % pt_file)

        if save_best:
            best_file = os.path.join(
                self.save_dir,
                DEPLOY_CHECKPOINT_FORMAT % (self.name_prefix, "best"),
            )
            shutil.copyfile(pt_file, best_file)
            if self.save_hash:
                best_file = generate_sha256_file(best_file, remove_old=True)
            logger.info("Save best deploy_model checkpoint: %s" % best_file)

    def _get_ckp_model(self, model, ema_model):
        ckp_model = model
        if ema_model is not None:
            ckp_model = ema_model
        return ckp_model

    def on_step_end(
        self,
        epoch_id,
        step_id,
        global_step_id,
        val_metrics,
        model=None,
        ema_model=None,
        optimizer=None,
        num_steps=None,
        **kwargs,
    ):
        ckp_model = self._get_ckp_model(model, ema_model)

        if self.interval_by == "step" and (
            (global_step_id + 1) % self.save_interval == 0
            or (num_steps is not None and (global_step_id + 1) == num_steps)
        ):
            self.do_checkpoint(
                ckp_model,
                optimizer,
                epoch_id,
                save_type="step",
                step_id=global_step_id,
                val_metrics=val_metrics,
            )

    def on_epoch_end(
        self,
        epoch_id,
        global_step_id,
        model,
        optimizer,
        num_epochs,
        val_metrics,
        ema_model=None,
        **kwargs,
    ):
        ckp_model = self._get_ckp_model(model, ema_model)

        if self.interval_by == "epoch" and (
            (epoch_id + 1) % self.save_interval == 0
            or (num_epochs is not None and (epoch_id + 1) == num_epochs)
        ):
            self.do_checkpoint(
                ckp_model,
                optimizer,
                epoch_id,
                save_type="epoch",
                step_id=global_step_id,
                val_metrics=val_metrics,
            )

    def do_checkpoint(
        self,
        model,
        optimizer,
        epoch_id,
        save_type="epoch",
        step_id=None,
        val_metrics=None,
    ):
        if self.mode is not None:
            if self.best_refer_metric is not None:
                metrics = _as_list(self.best_refer_metric)
            else:
                assert val_metrics is not None
                metrics = val_metrics
            value = None
            for val_metric in metrics:
                names, values = val_metric.get()
                if self.monitor_metric_key is None:
                    values = _as_list(values)
                    if len(values) != 1:
                        raise KeyError(
                            "Cannot resolve more than one metric values"
                            + "while monitor_metric_key is None."
                        )
                    value = values[0]
                    break
                else:
                    if self.monitor_metric_key in names:
                        index = _as_list(names).index(self.monitor_metric_key)
                        value = _as_list(values)[index]
                        break
            assert value is not None

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            is_best = self.monitor_op(value, self.best)
            if is_best:
                logger.info("former best:%f, current:%f" % (self.best, value))
                self.best = value
        else:
            is_best = False

        # save train model
        self.save_checkpoint(
            model,
            epoch_id=epoch_id,
            optimizer=optimizer,
            save_best=is_best,
            save_type=save_type,
            step_id=step_id,
        )

        # save deploy_model
        if self.deploy_model is not None:
            self.trace_and_save_deploy_model(
                train_model=model, save_best=is_best
            )

    def on_loop_end(self, model, optimizer, ema_model=None, **kwargs):
        if not self.save_on_train_end:
            return

        ckp_model = self._get_ckp_model(model, ema_model)

        # sometimes, we want to skip training and just export model checkpoint,
        # when skip training, `on_epoch_end` will not trigger, but
        # `on_loop_end` will.
        self.save_checkpoint(
            ckp_model,
            epoch_id=None,
            optimizer=optimizer,
            save_best=False,
            save_epoch_or_step=False,
            save_last=True,
        )

        if self.deploy_model is not None:
            self.trace_and_save_deploy_model(
                train_model=ckp_model, save_best=False
            )
