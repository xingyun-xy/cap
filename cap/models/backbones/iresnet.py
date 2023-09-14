import torch
import torch.nn as nn
from changan_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from cap.models.base_modules import ConvModule2d
from cap.registry import OBJECT_REGISTRY

__all__ = ["IResNet100", "IResNet180"]


class IBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: dict,
        act_type: str = "relu",
        stride: int = 1,
        bias: bool = False,
        inplace: bool = True,
    ):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, **bn_kwargs)
        if act_type == "relu":
            act_layer = nn.ReLU(inplace=inplace)
        else:
            raise TypeError("only support relu act")
        self.conv1 = ConvModule2d(
            in_channels,
            out_channels,
            3,
            padding=1,
            stride=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=act_layer,
        )
        self.conv2 = ConvModule2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            stride=stride,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=act_layer,
        )
        self.downsample = None
        if not (stride == 1 and in_channels == out_channels):
            self.downsample = ConvModule2d(
                in_channels,
                out_channels,
                3 if stride == 2 else 1,
                padding=1 if stride == 2 else 0,
                stride=stride,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                bias=bias,
            )

        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out


class IResNet(nn.Module):
    """
    A backbone module of FaceID.

    Refer to insightface residual_unit_v3,
    https://github.com/deepinsight/insightface/blob/master/src/
    symbols/fresnet.py

    Exclusively for face recognition :)

    Args:
        basic_block : Basic block for resnet.
        expansion : expansion of channels in basic_block.
        unit : Unit num for each block.
        channels_list : Channels for each block.
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        act_type : Nonlinear method.
        dropout : Dropout rate.
        embedding_size : FaceID output embedding_size.
        use_fp16 : Whether to use mixed precision training.
        inplace : Whether to use inplace op.
    """

    fc_scale = 7 * 7

    def __init__(
        self,
        basic_block: nn.Module,
        expansion: int,
        unit: list,
        channels_list: list,
        bn_kwargs: dict,
        bias: bool = False,
        act_type: str = "relu",
        dropout: float = 0.0,
        embedding_size: int = 256,
        use_fp16: bool = False,
        inplace: bool = True,
    ):
        super(IResNet, self).__init__()
        self.basic_block = basic_block
        self.expansion = expansion
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.in_channels = channels_list[0]
        self.act_type = act_type
        if self.act_type == "relu":
            act_layer = nn.ReLU(inplace=inplace)
        else:
            raise TypeError("only support relu act")
        self.embedding_size = embedding_size
        self.use_fp16 = use_fp16
        self.inplace = inplace

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.mod1 = nn.Sequential(
            ConvModule2d(
                3,
                channels_list[0],
                3,
                stride=1,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels_list[0], **bn_kwargs),
                act_layer=act_layer,
            ),
            ConvModule2d(
                channels_list[0],
                channels_list[0],
                3,
                stride=1,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels_list[0], **bn_kwargs),
                act_layer=act_layer,
            ),
        )

        self.mod2 = self._make_stage(channels_list[1], unit[0], 2)
        self.mod3 = self._make_stage(channels_list[2], unit[1], 2)
        self.mod4 = self._make_stage(channels_list[3], unit[2], 2)
        self.mod5 = self._make_stage(channels_list[4], unit[3], 2)
        self.bn = nn.BatchNorm2d(channels_list[4] * expansion, **bn_kwargs)

        self.dropout = nn.Dropout(p=dropout, inplace=self.inplace)
        self.fc = nn.Linear(
            channels_list[4] * expansion * self.fc_scale, self.embedding_size
        )
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, channels, repeats, stride):
        layers = []
        layers.append(
            self.basic_block(
                self.in_channels,
                channels,
                self.bn_kwargs,
                self.act_type,
                stride,
                bias=self.bias,
                inplace=self.inplace,
            )
        )
        self.in_channels = channels * self.expansion
        for _ in range(1, repeats):
            layers.append(
                self.basic_block(
                    self.in_channels,
                    channels,
                    self.bn_kwargs,
                    self.act_type,
                    bias=self.bias,
                    inplace=self.inplace,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        output = []
        with torch.cuda.amp.autocast(self.use_fp16):
            x = self.quant(x)
            for module in [
                self.mod1,
                self.mod2,
                self.mod3,
                self.mod4,
                self.mod5,
            ]:
                x = module(x)
                output.append(x)
            x = self.bn(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)

        x = self.fc(x.float() if self.use_fp16 else x)
        x = self.features(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]
        modules += [self.bn]
        modules += [self.fc]
        modules += [self.features]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        getattr(
            self.features, "1"
        ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class IResNet50(IResNet):
    """
    A module of resnet50.

    Args:
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        act_type : Nonlinear method.
        embedding_size : FaceID output embedding_size.
        dropout : Dropout rate.
        use_fp16 : Whether to use mixed precision training.
        inplace : Whether to use inplace op.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        bias: bool = True,
        act_type: str = "relu",
        embedding_size: int = 512,
        dropout: float = 0.0,
        use_fp16: bool = False,
        inplace: bool = True,
    ):
        unit = [3, 4, 6, 3]
        block = IBasicBlock
        expansion = 1
        channels_list = [64, 64, 128, 256, 512]
        super(IResNet50, self).__init__(
            block,
            expansion,
            unit,
            channels_list,
            bn_kwargs,
            bias,
            act_type=act_type,
            dropout=dropout,
            embedding_size=embedding_size,
        )


@OBJECT_REGISTRY.register
class IResNet100(IResNet):
    """
    A module of resnet100.

    Args:
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        act_type : Nonlinear method.
        embedding_size : FaceID output embedding_size.
        dropout : Dropout rate.
        use_fp16 : Whether to use mixed precision training.
        inplace : Whether to use inplace op.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        bias: bool = True,
        act_type: str = "relu",
        embedding_size: int = 256,
        dropout: float = 0.0,
        use_fp16: bool = False,
        inplace: bool = True,
    ):
        unit = [3, 13, 30, 3]
        block = IBasicBlock
        expansion = 1
        channels_list = [64, 64, 128, 256, 512]
        super(IResNet100, self).__init__(
            block,
            expansion,
            unit,
            channels_list,
            bn_kwargs,
            bias,
            act_type=act_type,
            dropout=dropout,
            embedding_size=embedding_size,
            use_fp16=use_fp16,
            inplace=inplace,
        )


@OBJECT_REGISTRY.register
class IResNet180(IResNet):
    """
    A module of resnet180.

    Args:
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        act_type : Nonlinear method.
        embedding_size : FaceID output embedding_size.
        dropout : Dropout rate.
        use_fp16 : Whether to use mixed precision training.
        inplace : Whether to use inplace op.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        bias: bool = True,
        act_type: str = "relu",
        embedding_size: int = 256,
        dropout: float = 0.0,
        use_fp16: bool = False,
        inplace: bool = True,
    ):
        unit = [3, 20, 40, 6]
        block = IBasicBlock
        expansion = 1
        channels_list = [64, 64, 128, 256, 512]
        super(IResNet180, self).__init__(
            block,
            expansion,
            unit,
            channels_list,
            bn_kwargs,
            bias,
            act_type=act_type,
            dropout=dropout,
            embedding_size=embedding_size,
            use_fp16=use_fp16,
            inplace=inplace,
        )
