from copy import deepcopy
from typing import Tuple

import torch.nn as nn

from cap.models.base_modules import ExtSequential
from cap.registry import OBJECT_REGISTRY

__all__ = ["get_update_state_dict", "ZeroPad2DPatcher"]


def get_update_state_dict(prefix):
    def update_func(old_state_dict):
        quant_prefix = ".".join([prefix, "quant"])
        q0_prefix = quant_prefix.replace("quant", "quant.0")
        q1_prefix = quant_prefix.replace("quant", "quant.1")

        new_state_dict = deepcopy(old_state_dict)
        for k in old_state_dict:
            if not k.startswith(prefix):
                continue

            if quant_prefix in k:
                k0 = k.replace(quant_prefix, q0_prefix)
                k1 = k.replace(quant_prefix, q1_prefix)
                new_state_dict[k0] = old_state_dict[k]
                new_state_dict[k1] = old_state_dict[k]

        return new_state_dict

    return update_func


@OBJECT_REGISTRY.register
class ZeroPad2DPatcher(nn.Module):
    """ZeroPad2D Patcher.

    Args:
        backbone: Backbone module.
        input_padding: Input padding, in (w_left, w_right, h_top, h_bottom) format.  # noqa E501

    """

    def __new__(cls, backbone: nn.Module, input_padding: Tuple[int]):

        assert len(input_padding) == 4
        if not all([p == 0 for p in input_padding]):
            pad = nn.ZeroPad2d(input_padding)
            if hasattr(backbone, "quant"):
                backbone.quant = ExtSequential(modules=[backbone.quant, pad])
            else:
                raise ValueError(
                    "`backbone` has no `quant` module, please check your `backbone`."  # noqa E501
                )
        return backbone
