# Copyright (c) Changan Auto. All rights reserved.
from torch import nn

from .basic_vargnet_module import BasicVarGBlockV2

BLOCK_CONFIG = {
    "varg_k3": {"kernel_size": 3, "padding": 1},
    "varg_k5": {"kernel_size": 5, "padding": 2},
    "varg_k3f1": {"kernel_size": 3, "padding": 1, "factor": 1},
    "vargr_k3": {"kernel_size": 3, "padding": 1, "merge_branch": True},
    "vargr_k5": {"kernel_size": 5, "padding": 2, "merge_branch": True},
}


class VargNASNetBlock(nn.Module):
    """
    A block for VargNASNetBlock.

    Args:
        in_ch (int): The in_channels for the block.
        block_ch (int): The out_channels for the block.
        head_op (str): One key of the BLOCK_CONFIG.
        stack_ops (list): a list consisting the keys of the
            BLOCK_CONFIG, or be None.
        stride (int): Stride of basic block.
        bias (bool): Whether to use bias in basic block.
        bn_kwargs (dict): Dict for BN layer.
    """

    def __init__(
        self, in_ch, block_ch, head_op, stack_ops, stride, bias, bn_kwargs
    ):
        super(VargNASNetBlock, self).__init__()
        self.head_layer = BasicVarGBlockV2(
            in_channels=in_ch,
            mid_channels=block_ch,
            out_channels=block_ch,
            stride=stride,
            bias=bias,
            bn_kwargs=bn_kwargs,
            **BLOCK_CONFIG[head_op],
        )

        modules = []
        for stack_op in stack_ops:
            modules.append(
                BasicVarGBlockV2(
                    in_channels=block_ch,
                    mid_channels=block_ch,
                    out_channels=block_ch,
                    stride=1,
                    bias=bias,
                    bn_kwargs=bn_kwargs,
                    **BLOCK_CONFIG[stack_op],
                )
            )
        self.stack_layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.head_layer(x)
        x = self.stack_layers(x)
        return x

    def fuse_model(self):
        self.head_layer.fuse_model()
        for module in self.stack_layers:
            module.fuse_model()
