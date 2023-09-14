# Copyright (c) Changan Auto. All rights reserved.
from typing import Dict, List, Optional

import changan_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn

from cap.models.base_modules import BasicVarGBlock, SeparableGroupConvModule2d
from cap.models.utils import _check_strides
from cap.registry import OBJECT_REGISTRY

__all__ = ["Unet"]

default_bn_kwargs = {"eps": 1e-5, "momentum": 0.1}


class FusionBlock(nn.Module):
    """
    Fusion block to fusion two level feature map.

    Parameters
    ----------
    up_c : int
        the channel of high level feature map.
    bottom_c : int
        the channel of high level feature map.

    """

    def __init__(
        self,
        up_c,
        bottom_c,
        use_bias=False,
        bn_kwargs=None,
        factor=2,
        group_base=8,
    ):
        super(FusionBlock, self).__init__()
        bn_kwargs = default_bn_kwargs if bn_kwargs is None else bn_kwargs

        self.bottom_proj = BasicVarGBlock(
            in_channels=bottom_c,
            mid_channels=bottom_c,
            out_channels=bottom_c,
            stride=1,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            factor=factor,
            group_base=group_base,
            merge_branch=False,
        )

        self.upsampling = hnn.Interpolate(
            scale_factor=2, recompute_scale_factor=True
        )

        group_num = int(up_c / group_base)
        self.fusion = SeparableGroupConvModule2d(
            in_channels=up_c,
            out_channels=bottom_c,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            groups=group_num,
            factor=factor,
            dw_norm_layer=nn.BatchNorm2d(
                int(up_c * factor),
                **bn_kwargs,
            ),
            dw_act_layer=nn.ReLU(inplace=True),
            pw_norm_layer=nn.BatchNorm2d(bottom_c, **bn_kwargs),
        )
        self.skip_add = nn.quantized.FloatFunctional()
        self.act = nn.ReLU(inplace=True)

    def forward(self, up, bottom):
        upscale = self.upsampling(up)
        bottom_proj = self.bottom_proj(bottom)
        fusion = self.fusion(upscale)
        fusion = self.skip_add.add(fusion, bottom_proj)
        fusion = self.act(fusion)
        return fusion

    def fuse_model(self):
        from changan_plugin_pytorch import quantization

        self.bottom_proj.fuse_model()
        getattr(self.fusion, "0").fuse_model()
        torch.quantization.fuse_modules(
            self,
            ["fusion.1.0", "fusion.1.1", "skip_add", "act"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


@OBJECT_REGISTRY.register
class Unet(nn.Module):
    """
    Unet neck module.

    Args:
        in_strides: contains the strides of feature maps from backbone.
        out_strides: contains the strides of feature maps the neck output.
        stride2channels: stride to channel dict

    """

    valid_strides = [2, 4, 8, 16, 32, 64, 128, 256]

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        stride2channels: Dict[int, int],
        factor: int = 2,
        use_bias: bool = False,
        bn_kwargs: Optional[Dict] = None,
        group_base: int = 8,
    ):
        super(Unet, self).__init__()
        bn_kwargs = default_bn_kwargs if bn_kwargs is None else bn_kwargs

        for stride_i in stride2channels:
            assert stride_i in self.valid_strides, (
                "stride %s not in stride2channels" % stride_i
            )

        self.in_strides = _check_strides(in_strides, self.valid_strides)
        min_idx = self.valid_strides.index(self.in_strides[0])
        max_idx = self.valid_strides.index(self.in_strides[-1])

        assert tuple(self.in_strides) == tuple(
            self.valid_strides[min_idx : max_idx + 1]
        ), "Input stride must be continuous"

        self.out_strides = _check_strides(out_strides, self.valid_strides)
        for stride in out_strides:
            assert (
                stride in in_strides
            ), "all stride of output stride must be in input stride"

        self.src_min_stride_idx = self.in_strides.index(self.out_strides[0])
        self.src_min_stride = self.in_strides[self.src_min_stride_idx]

        self.fusion_blocks = nn.ModuleList()
        for stride in self.in_strides[self.src_min_stride_idx :][::-1]:
            if stride == self.in_strides[-1]:
                continue
            else:
                top_stride = stride * 2
                bottom_stride = stride
                block = FusionBlock(
                    up_c=stride2channels[top_stride],
                    bottom_c=stride2channels[bottom_stride],
                    group_base=group_base,
                    bn_kwargs=bn_kwargs,
                    factor=factor,
                    use_bias=False,
                )
                self.fusion_blocks.append(block)

    def forward(self, features):
        assert len(features) == len(self.in_strides), "%d vs. %d" % (
            len(features),
            len(self.in_strides),
        )
        features = features[self.src_min_stride_idx :]

        fusion_features = {self.in_strides[-1]: features[-1]}

        for bottom_feat, stride, block in zip(
            features[:-1][::-1],
            self.in_strides[self.src_min_stride_idx : -1][::-1],
            self.fusion_blocks,
        ):

            fusion_features[stride] = block(
                fusion_features[stride * 2], bottom_feat
            )

        out_features = [
            fusion_features[stride_i] for stride_i in self.out_strides
        ]

        return out_features

    def fuse_model(self):
        for m in self.fusion_blocks:
            m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
