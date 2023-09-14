# Copyright (c) Changan Auto. All rights reserved.
from typing import List

import changan_plugin_pytorch as hpp
import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, ConvTransposeModule2d
from cap.models.weight_init import kaiming_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["PwcNetHead"]


class PredFlow(nn.Module):
    """
    A basic module of predict flow.

    Args:
        in_channel: Number of channel in the input feature map.
        out_channles: Numbers of channel in the block of this
                                module.
        bn_kwargs: Dict for BN layer.
        use_bn: Whether to use BN in module.
        bias: Whether to use bias in module.
        use_dense: Whether to use dense connections.
        use_res: Whether to use residual connections.
        act_type: Activation layer.
        is_pred_lvl: Whether this module is Final prediction module.

    """

    def __init__(
        self,
        in_channel: int,
        out_channles: List[int],
        bn_kwargs: dict,
        use_bn: bool = False,
        bias: bool = True,
        use_dense: bool = True,
        use_res: bool = False,
        act_type=None,
        is_pred_lvl: bool = False,
    ):

        super(PredFlow, self).__init__()
        self.in_channel = in_channel
        self.out_channles = out_channles
        self.use_bn = use_bn
        self.bn_kwargs = bn_kwargs
        self.bias = bias
        self.act_type = act_type
        self.use_dense = use_dense
        self.is_pred_lvl = is_pred_lvl

        self.mod = self._make_mod(self.in_channel, self.out_channles)
        out_mod_channels = self.out_channles[-1]
        if self.use_dense:
            out_mod_channels = sum(self.out_channles) + self.in_channel
            self.concat = nn.ModuleList()
            for _ in range(len(self.mod)):
                self.concat.append(nn.quantized.FloatFunctional())
        self.pred_flow = nn.Sequential(
            ConvModule2d(
                in_channels=out_mod_channels,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.bias,
                norm_layer=None,
                act_layer=None,
            ),
        )
        self.use_res = use_res
        self.res_cxt = None
        if self.use_res:
            res_out_channels = [128, 128, 128, 96, 64, 32, 2]
            res_dilations = [1, 2, 4, 8, 16, 1, 1]
            self.res_cxt = self._make_res_cxt(
                out_mod_channels, res_out_channels, res_dilations
            )
            self.short_add = nn.quantized.FloatFunctional()
        self.deconv_flow = ConvTransposeModule2d(
            in_channels=2,
            out_channels=2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=self.bias,
        )
        self.deconv_feat = ConvTransposeModule2d(
            in_channels=out_mod_channels,
            out_channels=2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=self.bias,
        )

    def _make_mod(self, in_channel, out_channles):
        mod = nn.ModuleList()
        for idx in range(len(out_channles)):
            mod.append(
                nn.Sequential(
                    ConvModule2d(
                        in_channels=in_channel,
                        out_channels=out_channles[idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=self.bias,
                        norm_layer=nn.BatchNorm2d(
                            out_channles[idx], **self.bn_kwargs
                        )
                        if self.use_bn
                        else None,
                        act_layer=self.act_type,
                    ),
                )
            )
            if self.use_dense:
                in_channel = out_channles[idx] + in_channel
            else:
                in_channel = out_channles[idx]
        return mod

    def _make_res_cxt(self, in_channel, out_channels, dilations):
        layers = []
        for idx in range(len(out_channels)):
            layers.append(
                ConvModule2d(
                    in_channels=in_channel,
                    out_channels=out_channels[idx],
                    kernel_size=3,
                    stride=1,
                    padding=dilations[idx],
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(
                        out_channels[idx], **self.bn_kwargs
                    )
                    if self.use_bn and idx != len(out_channels) - 1
                    else None,
                    act_layer=self.act_type
                    if idx != len(out_channels) - 1
                    else None,
                    dilation=dilations[idx],
                ),
            )
            in_channel = out_channels[idx]
        return nn.Sequential(*layers)

    def fuse_model(self):
        for module in self.mod:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
        if self.res_cxt:
            for module in self.res_cxt:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()
            from changan_plugin_pytorch import quantization

            torch.quantization.fuse_modules(
                self,
                ["pred_flow.0.0", "short_add"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        for idx, module in enumerate(self.mod):
            x1 = module(x)
            if self.use_dense:
                x = self.concat[idx].cap((x1, x), 1)
            else:
                x = x1
        flow = self.pred_flow(x)
        if self.use_res:
            flow = self.short_add.add(flow, self.res_cxt(x))
        if self.is_pred_lvl:
            return flow
        up_feat = self.deconv_feat(x)
        up_flow = self.deconv_flow(flow)
        return [flow, up_feat, up_flow]

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        if self.is_pred_lvl:
            if self.use_res:
                self.short_add.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
                self.res_cxt[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_qconfig()
            else:
                self.pred_flow[
                    0
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        else:
            if self.use_res:
                self.res_cxt[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_qconfig()
                self.short_add.qconfig = (
                    qconfig_manager.get_default_qat_qconfig()
                )
            else:
                self.pred_flow[
                    0
                ].qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class PwcNetHead(nn.Module):
    """
    A basic head of PWCNet.

    Args:
        in_channels: Number of channels in the input feature map.
        bn_kwargs: Dict for BN layer.
        use_bn: Whether to use BN in module.
        md: search range of Correlation module.
        use_res: Whether to use residual connections.
        use_dense: Whether to use dense connections.
        flow_pred_lvl: Which level to upsample to
            generate the final optical flow prediction.
        pyr_lvls: Number of feature levels in the flow pyramid.
        bias: Whether to use bias in module.
        act_type: Activation layer.
    """

    def __init__(
        self,
        in_channels: List[int],
        bn_kwargs: dict,
        use_bn: bool = False,
        md: int = 4,
        use_res: bool = True,
        use_dense: bool = True,
        flow_pred_lvl: int = 2,
        pyr_lvls: int = 6,
        bias: bool = True,
        act_type=None,
    ):

        super(PwcNetHead, self).__init__()

        self.bn_kwargs = bn_kwargs
        self.use_bn = use_bn
        self.use_dense = use_dense
        self.use_res = use_res
        self.bias = bias
        self.in_channels = in_channels
        self.pyr_lvls = pyr_lvls
        self.flow_pred_lvl = flow_pred_lvl
        self.act_type = act_type
        predict_flow_channels = [128, 128, 96, 64, 32]
        corr_out_channel = (md * 2 + 1) ** 2

        self.predict_flow = nn.ModuleList()
        self.warp_scale = []
        self.concat = nn.ModuleList()
        for idx in range(pyr_lvls, flow_pred_lvl - 1, -1):
            pred_in_channel = (
                corr_out_channel + 2 + 2 + self.in_channels[idx - 1]
            )
            use_res_flag = self.use_res
            pred_lvl = False
            if idx == pyr_lvls:
                pred_in_channel = corr_out_channel
            if idx == flow_pred_lvl:
                use_res_flag = True
                pred_lvl = True
            self.predict_flow.append(
                PredFlow(
                    in_channel=pred_in_channel,
                    out_channles=predict_flow_channels,
                    bn_kwargs=self.bn_kwargs,
                    use_bn=self.use_bn,
                    use_dense=self.use_dense,
                    use_res=use_res_flag,
                    act_type=self.act_type,
                    bias=self.bias,
                    is_pred_lvl=pred_lvl,
                )
            )
            self.warp_scale.append(20.0 / 2 ** idx)

        self.dequant = DeQuantStub()
        self.corr = hpp.nn.Correlation(max_displacement=md, pad_size=md)
        self.grid_sample = hpp.nn.GridSample()
        for _ in range(pyr_lvls, flow_pred_lvl, -1):
            self.concat.append(nn.quantized.FloatFunctional())
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of head module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(
                    m,
                    mode="fan_in",
                    nonlinearity="relu",
                    bias=0,
                    distribution="normal",
                )

    def warp(self, x: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
        """
        Warp an image/tensor (im2) back to im1, according to the optical flow.

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """

        vgrid = flo.permute(0, 2, 3, 1)
        output = self.grid_sample(x, vgrid)

        return output

    def forward(
        self, features: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        feat0 = features[0]
        feat1 = features[1]

        flow_list = []
        for idx in range(self.pyr_lvls, self.flow_pred_lvl - 1, -1):
            if idx == self.pyr_lvls:
                corr = self.corr(feat0[idx - 1], feat1[idx - 1])
                corr = self.act_type(corr)
                flow, up_feat, up_flow = self.predict_flow[
                    self.pyr_lvls - idx
                ](corr)
                flow_list.append(flow)
            else:
                warp_flow = up_flow.mul(self.warp_scale[self.pyr_lvls - idx])
                feat1_warp = self.warp(feat1[idx - 1], warp_flow)
                corr = self.corr(feat0[idx - 1], feat1_warp)
                corr = self.act_type(corr)
                corr = self.concat[self.pyr_lvls - idx - 1].cap(
                    (corr, feat0[idx - 1], up_flow, up_feat), 1
                )
                if idx == self.flow_pred_lvl:
                    flow = self.predict_flow[self.pyr_lvls - idx](corr)
                    flow_list.append(flow)
                else:
                    flow, up_feat, up_flow = self.predict_flow[
                        self.pyr_lvls - idx
                    ](corr)
                    flow_list.append(flow)
        if self.training:
            flow_list = [self.dequant(flow) for flow in flow_list]
            return flow_list
        else:
            flow = self.dequant(flow_list[-1])
            return flow

    def fuse_model(self):
        for module in self.predict_flow:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        for module in self.predict_flow:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
