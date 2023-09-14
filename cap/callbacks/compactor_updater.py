import copy
import logging
from collections import defaultdict

import numpy as np
from torch import nn
from torch.nn import Conv2d
from torch.nn.modules.batchnorm import _BatchNorm

from cap.callbacks import CallbackMixin
from cap.models.base_modules.conv_compactor import ConvCompactor2d
from cap.registry import OBJECT_REGISTRY
from cap.utils.model_helpers import get_binding_module

__all__ = ["CompactorUpdater"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class CompactorUpdater(CallbackMixin):
    """Callback used to do filter selection.

    When doing filter selection, this callback will add compactor
    after the conv layers. During training, compactor will be
    updated like normal parameters, but they have additional
    penalty gradients. After training, this callback calculates
    l2-norm for every row in every compactor, one row represents
    one filters' importance. We will selection pruned filters
    accordingto the value whether less than the given threshold.

    Args:
        before_mask_iters (int): skip some iterations before mask
        mask_interval (int): the interval iterations between masks
        pruned_epsilon (float): the threshold to select filters
        modules (list): the pruned modules
    """

    def __init__(
        self, before_mask_iters, mask_interval, pruned_epsilon, modules
    ):
        self.before_mask_iters = before_mask_iters
        self.mask_interval = mask_interval
        self.modules = modules
        self.conv_idx = 0
        self.pruned_epsilon = pruned_epsilon
        self.origin_channels = []
        self.conv2d_change = False
        self.kernel_nums = 0

    def on_loop_begin(self, model, optimizer, **kwargs):
        logger.info("Add compactor in {}".format(self.modules))
        model = get_binding_module(model)
        for block in self.modules:
            node = getattr(model, block)
            node_compactor = self._add_compactor_recursively(node, optimizer)
            setattr(model, block, node_compactor)

    def on_step_begin(self, step_id, model, **kwargs):
        if step_id > self.before_mask_iters:
            total_iters_in_compactor_phase = step_id - self.before_mask_iters
            if total_iters_in_compactor_phase > 0 and (
                total_iters_in_compactor_phase % self.mask_interval == 0
            ):
                self._resrep_mask_model(model=model)

    def on_optimizer_step_begin(self, model, **kwargs):
        for child_module in model.modules():
            if isinstance(child_module, ConvCompactor2d):
                child_module.add_penalty_gradients()

    def on_loop_end(self, model, **kwargs):
        self._pruning_compactor(
            model=model,
            thresh=self.pruned_epsilon,
        )

    def _add_compactor_recursively(self, model, optimizer):
        for module_name, _ in model.named_children():
            if not model._modules[module_name]:
                continue
            self.conv2d_change = False
            model._modules[module_name] = self._add_compactor(
                model._modules[module_name],
                optimizer,
            )
            if (
                len(model._modules[module_name]._modules) > 0
                and not self.conv2d_change
                and not isinstance(
                    model._modules[module_name], ConvCompactor2d
                )
            ):
                self._add_compactor_recursively(
                    model._modules[module_name],
                    optimizer,
                )
        return model

    def _add_compactor(self, block, optimizer):
        if not isinstance(block, (nn.Sequential, nn.ModuleList, nn.Conv2d)):
            return block
        stack = []
        if isinstance(block, (nn.Sequential, nn.ModuleList)):
            for m in block.children():
                if (
                    isinstance(m, _BatchNorm)
                    and stack
                    and isinstance(stack[-1], Conv2d)
                ):
                    out_channels = stack[-1].out_channels
                    device = stack[-1].weight.device
                    self.origin_channels.append(out_channels)
                    compactor = ConvCompactor2d(
                        out_channels, self.conv_idx
                    ).to(device)
                    self.conv_idx += 1
                    self.conv2d_change = True
                    if isinstance(m, _BatchNorm):
                        stack.append(copy.deepcopy(m))
                        stack.append(compactor)
                    elif isinstance(m, Conv2d):
                        stack.append(copy.deepcopy(m))
                        stack.append(compactor)
                    optimizer.add_param_group(
                        {
                            "params": filter(
                                lambda p: p.requires_grad,
                                compactor.parameters(),
                            )
                        }
                    )
                    self.kernel_nums += 1
                else:
                    stack.append(copy.deepcopy(m))
        elif isinstance(block, Conv2d):
            stack = []
            stack.append(copy.deepcopy(block))
            out_channels = block.out_channels
            device = stack[-1].weight.device
            compactor = ConvCompactor2d(out_channels, self.conv_idx).to(device)
            self.origin_channels.append(out_channels)
            stack.append(compactor)
            self.conv_idx += 1
            optimizer.add_param_group(
                {
                    "params": filter(
                        lambda p: p.requires_grad, compactor.parameters()
                    )
                }
            )
            self.conv2d_change = True
            self.kernel_nums += 1
        else:
            stack.append(copy.deepcopy(block))

        if isinstance(block, nn.Sequential) or isinstance(block, nn.Conv2d):
            return nn.Sequential(*stack)
        elif isinstance(block, nn.ModuleList):
            return nn.ModuleList(stack)
        else:
            raise ValueError(
                "Only support nn.Sequential, nn.ModuleList and nn.Conv2d"
            )

    def _resrep_get_layer_mask_ones_and_metric_dict(self, model):
        layer_mask_ones = {}
        layer_metric_dict = {}
        for child_module in model.modules():
            if isinstance(child_module, ConvCompactor2d) and hasattr(
                child_module, "conv_idx"
            ):
                layer_mask_ones[
                    child_module.conv_idx
                ] = child_module.get_num_mask_ones()
                metric_vector = child_module.get_metric_vector()
                for i in range(len(metric_vector)):
                    layer_metric_dict[
                        (child_module.conv_idx, i)
                    ] = metric_vector[i]
        return layer_mask_ones, layer_metric_dict

    def _set_model_masks(self, model, layer_masked_out_filters):
        for child_module in model.modules():
            if (
                isinstance(child_module, ConvCompactor2d)
                and hasattr(child_module, "conv_idx")
                and child_module.conv_idx in layer_masked_out_filters
            ):
                child_module.set_mask(
                    layer_masked_out_filters[child_module.conv_idx]
                )

    def _resrep_get_deps_and_metric_dict(self, model):
        new_deps = np.array(self.origin_channels)
        (
            layer_ones,
            metric_dict,
        ) = self._resrep_get_layer_mask_ones_and_metric_dict(model)
        for idx, ones in layer_ones.items():
            assert ones <= self.origin_channels[idx]
            new_deps[idx] = ones
        return new_deps, metric_dict

    def _get_cur_num_deactivated_filters(self, cur_deps):
        assert len(self.origin_channels) == len(cur_deps)
        diff = self.origin_channels - cur_deps
        assert np.sum(diff < 0) == 0

        result = 0
        for i in range(len(self.origin_channels)):
            result += self.origin_channels[i] - cur_deps[i]

        return result

    def _resrep_mask_model(self, model):
        cur_deps, metric_dict = self._resrep_get_deps_and_metric_dict(model)

        sorted_metric_dict = sorted(metric_dict, key=metric_dict.get)

        cur_deactivated = self._get_cur_num_deactivated_filters(cur_deps)

        next_deactivated_max = cur_deactivated + 4

        assert next_deactivated_max > 0
        attempt_deps = np.array(self.origin_channels)
        i = 0
        skip_idx = []
        while True:
            attempt_layer_filter = sorted_metric_dict[i]
            if attempt_deps[attempt_layer_filter[0]] <= 1:
                skip_idx.append(i)
                i += 1
                continue
            attempt_deps[attempt_layer_filter[0]] -= 1

            i += 1
            if i >= next_deactivated_max:
                break

        layer_masked_out_filters = defaultdict(list)
        for k in range(i):
            if k not in skip_idx:
                layer_masked_out_filters[sorted_metric_dict[k][0]].append(
                    sorted_metric_dict[k][1]
                )

        self._set_model_masks(model, layer_masked_out_filters)

    def _pruning_compactor(self, model, thresh):
        compactor_mats = {}
        for child_module in model.modules():
            if isinstance(child_module, ConvCompactor2d) and hasattr(
                child_module, "conv_idx"
            ):
                compactor_mats[child_module.conv_idx] = (
                    child_module.pwc.weight.detach().cpu().numpy()
                )

        for cur_conv_idx in range(self.kernel_nums + 1):
            fold_direct = cur_conv_idx in compactor_mats
            if fold_direct:
                fm = compactor_mats[cur_conv_idx]
                pruned_ids = sorted(self._get_pruned_ids(thresh, fm))
                logger.info("pruned ids: {}".format(pruned_ids))

    def _get_pruned_ids(self, thresh, compactor_mat):
        metric_vec = np.sqrt(np.sum(compactor_mat ** 2, axis=(1, 2, 3)))
        filter_ids_below_thresh = np.where(metric_vec < thresh)[0]
        if len(filter_ids_below_thresh) == len(metric_vec):
            sortd_ids = np.argsort(metric_vec)
            filter_ids_below_thresh = sortd_ids[:-3]
        return filter_ids_below_thresh
