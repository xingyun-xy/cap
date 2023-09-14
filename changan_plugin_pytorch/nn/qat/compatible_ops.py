# Copyright (c) 2022 by Contributors
# file: compatible_ops.py
# date: 2022-06-01
# brief: compatible ops
# =============================================================================

import torch
import changan_plugin_pytorch as hz


def relu(out):
    if hz.get_march() == hz.March.BERNOULLI:
        return torch.nn.functional.relu(out, inplace=True)
    if hz.qat_mode.tricks.relu6:
        return torch.nn.functional.relu6(out, inplace=True)
    return torch.nn.functional.relu(out, inplace=True)
