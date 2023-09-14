import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Conv3d,
    ConvTranspose2d,
    Module,
    ReLU,
    ReLU6,
    Linear,
)
from torch.nn.quantized import FloatFunctional


class ConvAdd2d(Module):
    r"""This is a container which calls the Conv2d and add modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add):
        super(ConvAdd2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert type(conv) == Conv2d and (
            type(add) == FloatFunctional or type(add) == HFloatFunctional
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(add)
        )
        self.conv = conv

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return x1 + x2


class ConvAddReLU2d(Module):
    r"""This is a container which calls the Conv2d and add relu modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add, relu):
        super(ConvAddReLU2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(add), type(relu)
        )
        self.conv = conv
        self.relu = relu

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return self.relu(x1 + x2)


class ConvReLU62d(Module):
    r"""This is a container which calls the Conv2d and ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, relu6):
        super(ConvReLU62d, self).__init__()
        assert (
            type(conv) == Conv2d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(relu6)
        )
        self.conv = conv
        self.relu6 = relu6

    def forward(self, x):
        x1 = self.conv(x)
        return self.relu6(x1)


class ConvAddReLU62d(Module):
    r"""This is a container which calls the Conv2d and add ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add, relu6):
        super(ConvAddReLU62d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(add), type(relu6)
        )
        self.conv = conv
        self.relu6 = relu6

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return self.relu6(x1 + x2)


class ConvTransposeReLU2d(Module):
    r"""This is a container which calls the ConvTranspose2d and relu modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv_transpose2d, relu):
        super(ConvTransposeReLU2d, self).__init__()
        assert (
            type(conv_transpose2d) == ConvTranspose2d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(
            type(conv_transpose2d), type(relu)
        )
        self.conv_transpose2d = conv_transpose2d
        self.relu = relu

    def forward(self, x1):
        x1 = self.conv_transpose2d(x1)
        return self.relu(x1)


class ConvTransposeAdd2d(Module):
    r"""This is a container which calls the ConvTranspose2d and add modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv_transpose2d, add):
        super(ConvTransposeAdd2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert type(conv_transpose2d) == ConvTranspose2d and (
            type(add) == FloatFunctional or type(add) == HFloatFunctional
        ), "Incorrect types for input modules{}{}".format(
            type(conv_transpose2d), type(add)
        )
        self.conv_transpose2d = conv_transpose2d

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv_transpose2d(x1)
        return x1 + x2


class ConvTransposeAddReLU2d(Module):
    r"""This is a container which calls the ConvTranspose2d and add relu
    modules. During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv_transpose2d, add, relu):
        super(ConvTransposeAddReLU2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv_transpose2d) == ConvTranspose2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv_transpose2d), type(add), type(relu)
        )
        self.conv_transpose2d = conv_transpose2d
        self.relu = relu

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv_transpose2d(x1)
        return self.relu(x1 + x2)


class ConvTransposeReLU62d(Module):
    r"""This is a container which calls the ConvTranspose2d and ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv_transpose2d, relu6):
        super(ConvTransposeReLU62d, self).__init__()
        assert (
            type(conv_transpose2d) == ConvTranspose2d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(conv_transpose2d), type(relu6)
        )
        self.conv_transpose2d = conv_transpose2d
        self.relu6 = relu6

    def forward(self, x):
        x1 = self.conv_transpose2d(x)
        return self.relu6(x)


class ConvTransposeAddReLU62d(Module):
    r"""This is a container which calls the ConvTranspose2d and add relu6
    modules. During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv_transpose2d, add, relu6):
        super(ConvTransposeAddReLU62d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv_transpose2d) == ConvTranspose2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv_transpose2d), type(add), type(relu)
        )
        self.conv_transpose2d = conv_transpose2d
        self.relu6 = relu6

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv_transpose2d(x1)
        return self.relu6(x1 + x2)


class ConvAdd3d(Module):
    r"""This is a container which calls the Conv3d and add modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add):
        super(ConvAdd3d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert type(conv) == Conv3d and (
            type(add) == FloatFunctional or type(add) == HFloatFunctional
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(add)
        )
        self.conv = conv

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return x1 + x2


class ConvAddReLU3d(Module):
    r"""This is a container which calls the Conv3d and add relu modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add, relu):
        super(ConvAddReLU3d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv3d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(add), type(relu)
        )
        self.conv = conv
        self.relu = relu

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return self.relu(x1 + x2)


class ConvReLU63d(Module):
    r"""This is a container which calls the Conv3d and ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, relu6):
        super(ConvReLU63d, self).__init__()
        assert (
            type(conv) == Conv3d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(relu6)
        )
        self.conv = conv
        self.relu6 = relu6

    def forward(self, x):
        x1 = self.conv(x)
        return self.relu6(x1)


class ConvAddReLU63d(Module):
    r"""This is a container which calls the Conv3d and add ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add, relu6):
        super(ConvAddReLU63d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv3d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(add), type(relu6)
        )
        self.conv = conv
        self.relu6 = relu6

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return self.relu6(x1 + x2)


# with bn
class ConvBN2d(Module):
    r"""This is a container which calls the Conv2d and BatchNorm2d modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, bn):
        super(ConvBN2d, self).__init__()

        assert type(conv) == Conv2d and isinstance(
            bn, torch.nn.modules.batchnorm._BatchNorm
        ), "Incorrect types for input modules{}{}".format(type(conv), type(bn))
        self.conv = conv
        self.bn = bn

    def forward(self, x1):
        x1 = self.conv(x1)
        return self.bn(x1)


class ConvBNAdd2d(Module):
    r"""This is a container which calls the Conv2d, BatchNormand add modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, bn, add):
        super(ConvBNAdd2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(add), type(bn)
        )
        self.conv = conv
        self.bn = bn

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        return x1 + x2


class ConvBNAddReLU2d(Module):
    r"""This is a container which calls the Conv2d, BatchNorm and add relu
    modules. During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv, bn, add, relu):
        super(ConvBNAddReLU2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(add), type(relu)
        )
        self.conv = conv
        self.relu = relu
        self.bn = bn

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        return self.relu(x1 + x2)


class ConvBNReLU2d(Module):
    r"""This is a container which calls the Conv2d, BatchNorm2d and ReLU
    modules. During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv, bn, relu):
        super(ConvBNReLU2d, self).__init__()
        assert (
            type(conv) == Conv2d
            and isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(bn), type(relu)
        )
        self.conv = conv
        self.relu = relu
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ConvBNReLU62d(Module):
    r"""This is a container which calls the Conv2d, BatchNorm2d and ReLU6
    modules. During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv, bn, relu6):
        super(ConvBNReLU62d, self).__init__()
        assert (
            type(conv) == Conv2d
            and isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(bn), type(relu6)
        )
        self.conv = conv
        self.relu6 = relu6
        self.bn = bn

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x)
        return self.relu6(x1)


class ConvBNAddReLU62d(Module):
    r"""This is a container which calls the Conv2d and add ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, bn, add, relu6):
        super(ConvBNAddReLU62d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(add), type(relu6)
        )
        self.conv = conv
        self.bn = bn
        self.relu6 = relu6

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        return self.relu6(x1 + x2)


class LinearAdd(Module):
    r"""This is a container which calls the Linear and add modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, add):
        super(LinearAdd, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert type(linear) == Linear and (
            type(add) == FloatFunctional or type(add) == HFloatFunctional
        ), "Incorrect types for input modules{}{}".format(
            type(linear), type(add)
        )
        self.linear = linear

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.linear(x1)
        return x1 + x2


class LinearAddReLU(Module):
    r"""This is a container which calls the Linear and add and relu modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, add, relu):
        super(LinearAddReLU, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(linear) == Linear
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(linear), type(add), type(relu)
        )
        self.linear = linear
        self.relu = relu

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.linear(x1)
        return self.relu(x1 + x2)


class LinearReLU(Module):
    r"""This is a container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, relu):
        super(LinearReLU, self).__init__()
        assert (
            type(linear) == Linear and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(
            type(linear), type(relu)
        )
        self.linear = linear
        self.relu = relu

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)


class LinearReLU6(Module):
    r"""This is a container which calls the Linear and ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, relu6):
        super(LinearReLU6, self).__init__()
        assert (
            type(linear) == Linear and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(linear), type(relu6)
        )
        self.linear = linear
        self.relu6 = relu6

    def forward(self, x):
        x = self.linear(x)
        return self.relu6(x)


class LinearAddReLU6(Module):
    r"""This is a container which calls the Linear and add and ReLU6 modules.
    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, add, relu6):
        super(LinearAddReLU6, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(linear) == Linear
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(linear), type(add), type(relu6)
        )
        self.linear = linear
        self.relu6 = relu6

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.linear(x1)
        return self.relu6(x1 + x2)


class DeformConvReLU2d(Module):
    def __init__(self, deform_conv2d, relu):
        super(DeformConvReLU2d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.relu = relu

    def forward(self, input, offset, mask=None):
        out = self.deform_conv2d(input, offset, mask)
        return self.relu(out)


class DeformConvReLU62d(Module):
    def __init__(self, deform_conv2d, relu6):
        super(DeformConvReLU62d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.relu6 = relu6

    def forward(self, input, offset, mask=None):
        out = self.deform_conv2d(input, offset, mask)
        return self.relu6(out)


class DeformConvAdd2d(Module):
    def __init__(self, deform_conv2d, add):
        super(DeformConvAdd2d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.add = add

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        out = self.deform_conv2d(*x1)
        return self.add.add(out, x2)


class DeformConvAddReLU2d(Module):
    def __init__(self, deform_conv2d, add, relu):
        super(DeformConvAddReLU2d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.add = add
        self.relu = relu

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        out = self.deform_conv2d(*x1)
        out = self.add.add(out, x2)
        return self.relu(out)


class DeformConvAddReLU62d(Module):
    def __init__(self, deform_conv2d, add, relu6):
        super(DeformConvAddReLU62d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.add = add
        self.relu6 = relu6

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        out = self.deform_conv2d(*x1)
        out = self.add.add(out, x2)
        return self.relu6(out)
