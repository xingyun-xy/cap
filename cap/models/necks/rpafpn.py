import logging

import changan_plugin_pytorch.nn as nnF
import torch.nn as nn
import torch.nn.functional as F

from cap.models.base_modules import ConvModule2d
from cap.models.weight_init import normal_init
from cap.registry import OBJECT_REGISTRY

__all__ = ["RPAFPN"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class RPAFPN(nn.Module):
    """The implementation of the `RPAFPN` in Rotate Path Aggregation Network.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to\
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to\
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv\
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed:

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra\
            conv. Default: False.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
    ):
        super(RPAFPN, self).__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule2d(
                in_channels=in_channels[i],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                padding=0,
            )
            fpn_conv = ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                groups=1,
                padding=1,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for _ in range(self.start_level, self.backbone_end_level):
            d_conv = ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                groups=1,
                padding=1,
            )
            pafpn_conv = ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                groups=1,
                padding=1,
            )
            if len(self.downsample_convs) < 3:
                self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    groups=1,
                    padding=1,
                )
                self.pafpn_convs.append(extra_fpn_conv)
        # nearest may get higher map than bilinear in mmdetection
        self.upsampling = nnF.Interpolate(
            scale_factor=2, mode="bilinear", recompute_scale_factor=True
        )
        self.add = nn.quantized.FloatFunctional()

        self._init_weights()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build bottom-up path
        used_backbone_levels = len(laterals)
        for i in range(0, used_backbone_levels - 1):
            laterals[i + 1] = self.add.add(
                self.downsample_convs[i](laterals[i]), laterals[i + 1]
            )

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.add.add(self.pafpn_convs[i](laterals[i]), laterals[i])
            for i in range(used_backbone_levels)
        ]

        # part 2: add top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = inter_outs[i - 1].shape[2:]
            inter_outs[i - 1] = self.add.add(
                self.upsampling(inter_outs[i]), inter_outs[i - 1]
            )

        # outs.append(inter_outs[0])
        outs = [
            self.add.add(self.fpn_convs[i](inter_outs[i]), inter_outs[i])
            for i in range(0, used_backbone_levels)
        ]

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == "on_input":
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.pafpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == "on_lateral":
                    outs.append(
                        self.pafpn_convs[used_backbone_levels](laterals[-1])
                    )
                elif self.add_extra_convs == "on_output":
                    # add from bottom-up features
                    outs.append(
                        self.pafpn_convs[used_backbone_levels](outs[-1])
                    )
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.pafpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.pafpn_convs[i](outs[-1]))
        return tuple(outs)

    def _init_weights(self):
        """Initialize the weights of RPAFPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def fuse_model(self):
        for m in self.lateral_convs:
            m.fuse_model()
        for m in self.downsample_convs:
            m.fuse_model()
        for m in self.pafpn_convs:
            m.fuse_model()
        for m in self.fpn_convs:
            m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
