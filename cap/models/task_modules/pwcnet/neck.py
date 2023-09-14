# Copyright (c) Changan Auto. All rights reserved.

import torch.nn as nn
from changan_plugin_pytorch.quantization import QuantStub

from cap.models.base_modules import ConvModule2d
from cap.models.weight_init import kaiming_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["PwcNetNeck"]


@OBJECT_REGISTRY.register
class PwcNetNeck(nn.Module):
    """
    A extra features module of PWCNet.

    Args:
        out_channels: Channels for each block.
        use_bn: Whether to use BN in module.
        bn_kwargs: Dict for BN layer.
        bias: Whether to use bias in module.
        pyr_lvls: Number of feature levels in the flow pyramid.
        flow_pred_lvl: Which level to upsample to\
        generate the final optical flow prediction.
        act_type: Activation layer.
    """

    def __init__(
        self,
        out_channels: list,
        use_bn: bool,
        bn_kwargs: dict,
        bias: bool = True,
        pyr_lvls: int = 6,
        flow_pred_lvl: int = 2,
        act_type=None,
    ):
        super(PwcNetNeck, self).__init__()

        assert (
            len(out_channels) == pyr_lvls
        ), "The length of out_channels must equal pyr_lvls."
        assert (
            pyr_lvls >= flow_pred_lvl
        ), "The pyr_lvls must be greater than or equal to flow_pred_lvl."

        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.quant = QuantStub()
        self.pyr_lvls = pyr_lvls
        self.flow_pred_lvl = flow_pred_lvl
        self.use_bn = use_bn
        self.out_channels = out_channels

        self.mod = nn.ModuleList()
        self.mod.append(self._make_stage(3, self.out_channels[0], act_type))
        for idx in range(self.pyr_lvls - 1):
            self.mod.append(
                self._make_stage(
                    self.out_channels[idx],
                    self.out_channels[idx + 1],
                    act_type,
                )
            )
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of pwcnet module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(
                    m,
                    mode="fan_in",
                    nonlinearity="relu",
                    bias=0,
                    distribution="normal",
                )

    def _make_stage(self, in_channels, out_channels, act_type):
        layers = []
        layers.append(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs)
                if self.use_bn
                else None,
                act_layer=act_type,
            ),
        )
        for _ in range(2):
            layers.append(
                ConvModule2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs)
                    if self.use_bn
                    else None,
                    act_layer=act_type,
                ),
            )
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.quant(x)
        output1 = []
        output2 = []
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]

        for module in self.mod:
            x1 = module(x1)
            x2 = module(x2)
            output1.append(x1)
            output2.append(x2)
        return [output1, output2]

    def fuse_model(self):
        for module in self.mod:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
