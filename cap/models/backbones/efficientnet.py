# Copyright (c) Changan Auto. All rights reserved.

import collections
import copy
import functools
import math
from typing import Dict, Sequence

import torch.nn as nn
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d, MBConvBlock
from cap.registry import OBJECT_REGISTRY, build_from_registry

__all__ = ["EfficientNet", "efficientnet", "efficientnet_lite"]

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "in_filters",
        "out_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
    ],
)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = (
    BlockArgs(
        kernel_size=3,
        num_repeat=1,
        in_filters=32,
        out_filters=16,
        expand_ratio=1,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=2,
        in_filters=16,
        out_filters=24,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=2,
        in_filters=24,
        out_filters=40,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=3,
        in_filters=40,
        out_filters=80,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=3,
        in_filters=80,
        out_filters=112,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=4,
        in_filters=112,
        out_filters=192,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=1,
        in_filters=192,
        out_filters=320,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
)


def round_filters(filters, width_coefficient, depth_division):
    """Round number of filters based on width coefficient."""

    filters *= width_coefficient
    new_filters = max(
        int((filters + depth_division / 2) // depth_division * depth_division),
        depth_division,
    )
    # Make sure that round down does not go down by more than 10%
    if new_filters < 0.9 * filters:
        new_filters += depth_division
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth coefficient."""

    return int(math.ceil(depth_coefficient * repeats))


class EfficientNet(nn.Module):
    """
    A module of EfficientNet.

    Args:
        model_type (str): Select to use which EfficientNet(B0-B7 or lite0-4), \
            for EfficientNet model, model_type must be one of: \
              ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], \
            for EfficientNet-lite model, model_type must be one of: \
              ['lite0', 'lite1', 'lite2', 'lite3', 'lite4'].
        coefficient_params (tuple): Parameter coefficients of EfficientNet, \
            include: \
              width_coefficient(float): scaling coefficient for net width. \
              depth_coefficient(float): scaling coefficient for net depth. \
              default_resolution(int): default input image size. \
              dropout_rate(float): dropout rate for final classifier layer. \
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for Bn layer.
        bias (bool): Whether to use bias in module.
        drop_connect_rate (float): Dropout rate at skip connections.
        depth_division (int): Depth division, Defaults to 8.
        activation (str): Activation layer, defaults to 'relu'.
        use_se_block (bool): Whether to use SEBlock in module.
        blocks_args (list): A list of BlockArgs to MBConvBlock modules.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
    """

    def __init__(
        self,
        model_type: str,
        coefficient_params: tuple,
        num_classes: int,
        bn_kwargs: dict = None,
        bias: bool = False,
        drop_connect_rate: float = None,
        depth_division: int = 8,
        activation: str = "relu",
        use_se_block: bool = False,
        blocks_args: Sequence[Dict] = DEFAULT_BLOCKS_ARGS,
        include_top: bool = True,
        flat_output: bool = True,
    ):

        super().__init__()

        assert activation in ["relu", "relu6", "swish"], (
            f'activation must be one of ["relu", "relu6", "swish"], but '
            f"get {activation}"
        )

        self.model_type = model_type
        self.use_lite = True if "lite" in self.model_type else False
        if self.use_lite:
            assert (
                use_se_block is False and not activation == "swish"
            ), '"Swish" activation and "Squeeze-and-excitation" block \
                    cannot be used in EfficientNet-lite model'

        (
            self.width_coefficient,
            self.depth_coefficient,
            self.default_resolution,
            self.dropout_rate,
        ) = coefficient_params
        self.drop_connect_rate = drop_connect_rate
        self.depth_division = depth_division
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.blocks_args = [
            BlockArgs(**block_args)
            if isinstance(block_args, dict)
            else block_args
            for block_args in blocks_args
        ]
        self.activation = activation
        act_layer = build_from_registry(
            dict(type=activation, inplace=True)  # noqa
        )

        if bn_kwargs is not None:
            self.bn_kwargs = bn_kwargs
        else:
            batch_norm_momentum = 0.99
            batch_norm_epsilon = 1e-3
            self.bn_kwargs = {
                "momentum": 1.0 - batch_norm_momentum,
                "eps": batch_norm_epsilon,
            }
        self.use_se_block = use_se_block

        if self.use_lite:
            out_planes = 32
        else:
            out_planes = round_filters(
                32, self.width_coefficient, self.depth_division
            )

        self.mod1 = nn.Sequential(
            ConvModule2d(
                in_channels=3,
                out_channels=out_planes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_planes, **self.bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
        )

        self.mod2 = self._make_stage(act_layer)

        in_planes = round_filters(
            self.blocks_args[-1].out_filters,
            self.width_coefficient,
            self.depth_division,
        )
        if self.use_lite:
            out_planes = 1280
        else:
            out_planes = round_filters(
                1280, self.width_coefficient, self.depth_division
            )

        if self.include_top:
            self.mod3 = nn.Sequential(
                ConvModule2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_planes, **self.bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
            )

            self.output = nn.Sequential(
                nn.AvgPool2d(self.default_resolution // (2 ** 5)),
                nn.Dropout2d(self.dropout_rate),
                ConvModule2d(
                    in_channels=out_planes,
                    out_channels=num_classes,
                    kernel_size=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **self.bn_kwargs),
                ),
            )
        else:
            self.output = None

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()

    def _make_stage(self, act_layer):
        self._blocks_conv = nn.ModuleList([])
        self.stage_divider = []
        for idx, block_args in enumerate(self.blocks_args):
            if self.use_lite and idx == 0:
                in_filters = block_args.in_filters
            else:
                in_filters = round_filters(
                    block_args.in_filters,
                    self.width_coefficient,
                    self.depth_division,
                )
            out_filters = round_filters(
                block_args.out_filters,
                self.width_coefficient,
                self.depth_division,
            )
            if self.use_lite and (
                idx == 0 or idx == len(self.blocks_args) - 1
            ):
                num_repeat = block_args.num_repeat
            else:
                num_repeat = round_repeats(
                    block_args.num_repeat, self.depth_coefficient
                )
            block_args = block_args._replace(
                in_filters=in_filters,
                out_filters=out_filters,
                num_repeat=num_repeat,
            )

            self._blocks_conv.append(
                MBConvBlock(
                    block_args,
                    bn_kwargs=self.bn_kwargs,
                    act_layer=copy.deepcopy(act_layer),
                    use_se_block=self.use_se_block,
                )
            )
            if block_args.strides == 2:
                self.stage_divider.append(len(self._blocks_conv) - 1)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    in_filters=block_args.out_filters,
                    strides=1,
                )

            for _ in range(block_args.num_repeat - 1):
                self._blocks_conv.append(
                    MBConvBlock(
                        block_args,
                        bn_kwargs=self.bn_kwargs,
                        act_layer=copy.deepcopy(act_layer),
                        use_se_block=self.use_se_block,
                    )
                )
        self.stage_divider.append(len(self._blocks_conv))
        return self._blocks_conv

    def forward(self, inputs):
        output = []
        x = inputs
        x = self.quant(x)
        x = self.mod1(x)
        for idx, block in enumerate(self._blocks_conv):
            x = block(x)
            if idx + 1 in self.stage_divider:
                output.append(x)
        if not self.include_top:
            return output

        x = self.mod3(x)
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2]
        if self.include_top:
            modules += [self.mod3, self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "2"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


def _efficient(mode_type, coefficient_params, **kwargs):
    efficient = functools.partial(EfficientNet, mode_type, coefficient_params)
    return efficient(**kwargs)


@OBJECT_REGISTRY.register
def efficientnet(model_type, **kwargs):  # noqa: D200, D401
    """
    A module of efficientnet.

    """
    coefficient_params_dict = {
        "b0tiny": (0.5, 0.5, 224, 0.2),
        "b0small": (0.5, 1.0, 224, 0.2),
        "b0": (1.0, 1.0, 224, 0.2),
        "b1": (1.0, 1.1, 240, 0.2),
        "b2": (1.1, 1.2, 260, 0.3),
        "b3": (1.2, 1.4, 300, 0.3),
        "b4": (1.4, 1.8, 380, 0.4),
        "b5": (1.6, 2.2, 456, 0.4),
        "b6": (1.8, 2.6, 528, 0.5),
        "b7": (2.0, 3.1, 600, 0.5),
    }
    assert model_type in list(coefficient_params_dict.keys()), (
        f"model_type must be one of {list(coefficient_params_dict.keys())}, "
        f"but get {model_type}"
    )

    return _efficient(
        model_type, coefficient_params_dict[model_type], **kwargs
    )


@OBJECT_REGISTRY.register
def efficientnet_lite(model_type, **kwargs):  # noqa: D200, D401
    """
    A module of efficientnet_lite.

    """
    coefficient_params_dict = {
        "lite0": (1.0, 1.0, 224, 0.2),
        "lite1": (1.0, 1.1, 240, 0.2),
        "lite2": (1.1, 1.2, 260, 0.3),
        "lite3": (1.2, 1.4, 280, 0.3),
        "lite4": (1.4, 1.8, 300, 0.3),
    }
    assert model_type in list(coefficient_params_dict.keys()), (
        f"model_type must be one of {list(coefficient_params_dict.keys())}, "
        f"but get {model_type}"
    )

    return _efficient(
        model_type, coefficient_params_dict[model_type], **kwargs
    )
