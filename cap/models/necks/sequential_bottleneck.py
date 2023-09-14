# Copyright (c) Changan Auto. All rights reserved.

# Sequential bottleneck module

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from cap.models.base_modules import (
    ConvModule2d,
    ConvTransposeModule2d,
    ConvUpsample2d,
)
from cap.models.weight_init import xavier_init
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register_module
class SequentialBottleNeck(nn.Module):
    def __init__(
        self,
        layer_nums: List[int],
        ds_layer_strides: List[int],
        ds_num_filters: List[int],
        us_layer_strides: List[int],
        us_num_filters: List[int],
        num_input_features: int,
        bn_kwargs: Optional[Dict[str, Any]] = None,
        quantize: bool = False,
        return_stride_features: bool = True,
        use_relu6: bool = True,
        dilation_kernel: int = 1,
        conv_kernel: int = 3,
        ds_maxpool: bool = False,
        ds_avgpool: bool = False,
        use_iden: bool = False,
        use_scnet: bool = False,
        use_res2net: bool = False,
        use_repvgg: bool = False,
        deploy: bool = False,
        use_tconv: bool = False,
    ):
        """Sequenence of bottlneck modules.

        Implments the network structure of PointPillars:
        <https://arxiv.org/abs/1812.05784>

        Although the strucure is called backbone in the original paper, we
        follow the publicly available code structure and use it as a neck
        module.

        Adapted from GitHub Det3D:
        <https://github.com/poodarchu/Det3D>

        Args:
            layer_nums (List[int]): number of layers for each stage.
            ds_layer_strides (List[int]): stride for each down-sampling stage.
            ds_num_filters (List[int]): number of filters for each down-
                sampling stage.
            us_layer_strides (List[int]): stride for each up-sampling stage.
            us_num_filters (List[int]): number of filters for each up-sampling
                stage.
            num_input_features (int): number of input feature channels.
            bn_kwargs (dict, optional): batch norm kwargs. Defaults to None.
            quantize (bool, optional): whether to quantize the module.
                Defaults to False.
            return_stride_features: return middle features.
            use_relu6: use activation funcation relu6.
            dilation_kernel:dilation convolution kernel size.
            conv_kernel: convolution kernel size.
            use_iden: use identity modules.
            ds_maxpool:use maxpool after conv block.
            ds_avgpool:use avgpool after conv block.
            use_scnet: whether to build scnet block.
            use_res2net: whether to build res2net block.
            use_repvgg: whether to build repvgg block.
            use_tconv: whether to use transposed convolution in upsampling.
        """
        super(SequentialBottleNeck, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features
        self._dilation_kernel = [dilation_kernel] * len(self._layer_nums)
        self._conv_kernel = [conv_kernel] * len(self._layer_nums)
        self.ds_maxpool = ds_maxpool
        self.ds_avgpool = ds_avgpool
        # ds_maxpool and ds_avgpool cannot be True together
        assert not (self.ds_maxpool and self.ds_avgpool)
        self.use_iden = use_iden
        self.return_stride_features = return_stride_features
        self.quantize = quantize

        if bn_kwargs is None:
            bn_kwargs = {"eps": 1e-3, "momentum": 0.01}
        self._bn_kwargs = bn_kwargs

        self.act_layer = (
            nn.ReLU6(inplace=True) if use_relu6 else nn.ReLU(inplace=True)
        )

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(
            self._upsample_strides
        )

        # Make sure that downsample scale and upsample scale match
        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(
                    self._layer_strides[: i + self._upsample_start_idx + 1]
                )
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        # Select different types of blocks
        if use_scnet:
            make_layer = self._make_scnet_layer
        elif use_res2net:
            make_layer = self._make_res2net_layer
        elif use_repvgg:
            make_layer = self._make_repvgg_layer
            self.deploy = deploy
        else:
            make_layer = self._make_layer

        self.cap = nn.quantized.FloatFunctional()

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
                dilation=self._dilation_kernel[i],
                kernel=self._conv_kernel[i],
                ds_maxpool=self.ds_maxpool,
                ds_avgpool=self.ds_avgpool,
                use_iden=self.use_iden,
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = self._upsample_strides[i - self._upsample_start_idx]
                if stride > 1:
                    deblock = self._make_deblock(
                        num_out_filters, i, stride, True, use_tconv
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = self._make_deblock(
                        num_out_filters, i, stride, False, use_tconv
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self) -> int:
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(
        self,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        dilation: int = 1,
        kernel: int = 3,
        ds_maxpool: bool = False,
        ds_avgpool: bool = False,
        use_iden: bool = False,
    ):
        """Create original layer structure."""
        block_list = []

        if use_iden:
            block_list.append(nn.Identity())
        block_list.append(
            ConvModule2d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=kernel,
                dilation=dilation,
                stride=1,
                bias=False,
                padding=kernel // 2,
                norm_layer=nn.BatchNorm2d(planes, **self._bn_kwargs),
                act_layer=self.act_layer,
            )
        )
        if ds_maxpool:
            block_list.append(nn.MaxPool2d(stride))
        elif ds_avgpool:
            block_list.append(nn.AvgPool2d(stride))

        block = nn.Sequential()
        for i in range(len(block_list)):
            block.add_module(str(i), block_list[i])

        for j in range(num_blocks):
            block.add_module(
                str(j + len(block_list)),
                ConvModule2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    dilation=dilation,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(planes, **self._bn_kwargs),
                    act_layer=self.act_layer,
                ),
            )

        return block, planes

    def _make_res2net_layer(
        self,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        **kwargs
    ):
        """Create Res2net layer structure.

        NOTE: This is still and experimental feature, so it only supports
        floating-point functionalities.
        """
        layers = []
        block = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            nn.BatchNorm2d(planes, **self._bn_kwargs),
            nn.ReLU(),
        )
        layers.append(block)

        for _ in range(num_blocks):
            layers.append(Bottle2neck(inplanes=planes, planes=planes))

        blocks = nn.Sequential(*layers)
        return blocks, planes

    def _make_repvgg_layer(
        self,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        **kwargs
    ):
        """Create repvgg layer structure.

        NOTE: This is still and experimental feature, so it only supports
        floating-point functionalities.
        """
        # works better, use normal downsample and repvgg block
        layers = []
        block = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            nn.BatchNorm2d(planes, **self._bn_kwargs),
            nn.ReLU(),
        )
        layers.append(block)
        for _ in range(num_blocks):
            layers.append(
                RepVGGBlock(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3,
                    padding=1,
                    deploy=self.deploy,
                )
            )
        blocks = nn.Sequential(*layers)
        return blocks, planes

    def _make_scnet_layer(
        self,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        **kwargs
    ):
        """Create Self-calibrated Conv Net layer structure.

        NOTE: This is still and experimental feature, so it only supports
        floating-point functionalities.
        """
        layers = []
        block = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            nn.BatchNorm2d(planes, **self._bn_kwargs),
            nn.ReLU(),
        )
        layers.append(block)

        for _ in range(num_blocks):
            layers.append(SCBottleneck(inplanes=planes, planes=planes))

        blocks = nn.Sequential(*layers)
        return blocks, planes

    def _make_deblock(
        self,
        num_out_filters: int,
        i: int,
        stride: int,
        deconv: bool = True,
        use_tconv: bool = False,
    ):
        """Create deconvolution layer structure."""
        norm_layer = nn.BatchNorm2d(
            self._num_upsample_filters[i - self._upsample_start_idx],
            **self._bn_kwargs
        )
        act_layer = self.act_layer
        if deconv:
            if use_tconv:
                return ConvTransposeModule2d(
                    in_channels=num_out_filters,
                    out_channels=self._num_upsample_filters[
                        i - self._upsample_start_idx
                    ],
                    kernel_size=stride,
                    stride=stride,
                    bias=False,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            else:
                kernel_size = stride + 1
                padding = kernel_size // 2
                return ConvUpsample2d(
                    in_channels=num_out_filters,
                    out_channels=self._num_upsample_filters[
                        i - self._upsample_start_idx
                    ],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
        else:
            return ConvModule2d(
                num_out_filters,
                self._num_upsample_filters[i - self._upsample_start_idx],
                stride,
                stride,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x: torch.Tensor):

        ups = []
        stride_feats = []

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride_feats.append(x)
            ups.append(x)

        # Use deconv to upsample them
        for i in range(len(ups)):
            if i - self._upsample_start_idx >= 0:
                ups[i] = self.deblocks[i - self._upsample_start_idx](ups[i])

        ups = ups[self._upsample_start_idx :]

        if len(ups) > 0:
            if self.quantize:
                x = self.cap.cap(ups, dim=1)
            else:
                x = torch.cat(ups, dim=1)
        if self.return_stride_features:
            return x, stride_feats
        else:
            return x, None

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

    def fuse_model(self):
        for m in self.modules():
            if isinstance(
                m, (ConvModule2d, ConvUpsample2d, ConvTransposeModule2d)
            ):
                m.fuse_model()


class SCBottleneck(nn.Module):
    """Create Self-calibrated Convolution Network Bottleneck.

    NOTE: This module is an experimental feature. Future work includes
    supporting quantification and making it reusable for different tasks, such
    as image-based detections.
    """

    # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.
    pooling_r = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        cardinality: int = 1,
        bottleneck_width: int = 32,
        avd: bool = False,
        dilation: int = 1,
        is_first: int = False,
    ):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1_a = nn.Conv2d(
            inplanes, group_width, kernel_size=1, bias=False
        )
        self.bn1_a = nn.BatchNorm2d(group_width)
        self.conv1_b = nn.Conv2d(
            inplanes, group_width, kernel_size=1, bias=False
        )
        self.bn1_b = nn.BatchNorm2d(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
            ),
            nn.BatchNorm2d(group_width),
        )

        self.scconv = SCConv(
            group_width,
            group_width,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=cardinality,
            pooling_r=self.pooling_r,
        )

        self.conv3 = nn.Conv2d(
            group_width * 2, planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        baseWidth: int = 24,
        scale: int = 4,
        stype: str = "normal",
    ):
        """Bottle2neck module.

        NOTE: This module is an experimental feature. Future work includes
        supporting quantification and making it reusable for different tasks,
        such as image-based detections.

        Args:
            inplanes (int): input channels.
            planes (int): output channels.
            stride (int, optional): conv stride. Defaults to 1.
            baseWidth (int, optional): base channel width. Defaults to 24.
            scale (int, optional): scale that is used to control the splitting
                of conv kernel portions. Defaults to 4.
            stype (str, optional): scale type. Use "normal" or "stage".
                Defaults to "normal".
        """

        super(Bottle2neck, self).__init__()
        assert stype in ("normal", "stage")
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = nn.Conv2d(
            inplanes, width * scale, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(
            width * scale, planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)

        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                ),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SCConv(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        pooling_r: int,
    ):
        """Self-calibrated convolution module.

        <http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf>

        NOTE: This module is an experimental feature. Future work includes
        supporting quantification and making it reusable for different tasks,
        such as image-based detections.

        Args:
            inplanes (int): input number of channels.
            planes (int): output number of channels.
            stride (int): convolution stride.
            padding (int): convolution padding.
            dilation (int): convolution dilation.
            groups (int): convolution groups.
            pooling_r (int): pooling size in the average pooling layer.
        """
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))
        )  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False,
    ):
        """Rep-vgg block module.

        This module implements structure in the following paper:

        RepVGG: Making VGG-style ConvNets Great Again
        <https://arxiv.org/pdf/2101.03697.pdf>

        NOTE: This module is an experimental feature. Future work includes
        supporting quantification and making it reusable for different tasks,
        such as image-based detections.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            kernel_size (int): conv kernel size.
            stride (int, optional): conv stride. Defaults to 1.
            padding (int, optional): conv padding. Defaults to 0.
            dilation (int, optional): conv dilation. Defaults to 1.
            groups (int, optional): conv groups. Defaults to 1.
            padding_mode (str, optional): padding mode. Defaults to 'zeros'.
            deploy (bool, optional): whether for training or deploy.
                Defaults to False.
        """
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
                norm_layer=nn.BatchNorm2d(out_channels),
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                norm_layer=nn.BatchNorm2d(out_channels),
            )
            self.rbr_1x1 = ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                norm_layer=nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    def get_equivalent_kernel_bias(self):
        """Derive the equivalent kernel and bias in a differentiable way.

        You can get the equivalent kernel and bias at any time and do
        whatever you want, for example, apply some penalties or constraints
        during training, just like you do to the other models.
        May be useful for quantization or pruning.
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: torch.Tensor):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device
                )
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu(), bias.detach().cpu()
