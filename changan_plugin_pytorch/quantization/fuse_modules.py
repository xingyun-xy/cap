"""
fuse modules
"""

import copy
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq
from changan_plugin_pytorch.nn.linear import Identity
from changan_plugin_pytorch.nn.quantized import (
    FloatFunctional as HFloatFunctional,
)
from changan_plugin_pytorch.qat_mode import QATMode, get_qat_mode
from torch import nn
from torchvision import ops


def fuse_conv_bn(conv, bn, fuse_bn=True):
    r"""Given the conv and bn modules, fuses them and always returns
        the fused-conv module.

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert (
        conv.training == bn.training
    ), "Conv and BN both must be in the same mode (train or eval)."
    assert (
        bn.num_features == conv.out_channels
    ), "Output channel of Conv must match num_features of BatchNorm"

    if not fuse_bn:
        from ..nn import intrinsic

        return intrinsic.ConvBN2d(conv, bn)

    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = torch.nn.utils.fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )
    return fused_conv


def fuse_conv_relu6(conv, relu6):
    from ..nn import intrinsic

    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvReLU62d,
        nn.Conv3d: intrinsic.ConvReLU63d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)

    return fused_module(conv, relu6)


def fuse_conv_bn_relu(conv, bn, relu, fuse_bn=True):
    assert (
        conv.training == bn.training == relu.training
    ), "Conv and BN and relu must be in the same mode (train or eval)."

    if not fuse_bn:
        from ..nn import intrinsic

        return intrinsic.ConvBNReLU2d(conv, bn, relu)
    fused_conv = fuse_conv_bn(conv, bn)
    map_to_fused_module = {
        nn.Conv2d: nni.ConvReLU2d,
        nn.Conv3d: nni.ConvReLU3d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(fused_conv, relu)


def fuse_conv_bn_add(conv, bn, add, fuse_bn=True):
    assert (
        conv.training == bn.training == add.training
    ), "Conv and BN and add must be in the same mode (train or eval)."
    from ..nn import intrinsic

    if not fuse_bn:
        return intrinsic.ConvBNAdd2d(conv, bn, add)
    fused_conv = fuse_conv_bn(conv, bn)

    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvAdd2d,
        nn.Conv3d: intrinsic.ConvAdd3d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(fused_conv, add)


def fuse_conv_bn_add_relu(conv, bn, add, relu, fuse_bn=True):
    assert (
        conv.training == bn.training == add.training
    ), "Conv and BN and add must be in the same mode (train or eval)."
    from ..nn import intrinsic

    if not fuse_bn:
        return intrinsic.ConvBNAddReLU2d(conv, bn, add, relu)

    fused_conv = fuse_conv_bn(conv, bn)
    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvAddReLU2d,
        nn.Conv3d: intrinsic.ConvAddReLU3d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(fused_conv, add, relu)


def fuse_conv_add(conv, add):
    from ..nn import intrinsic

    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvAdd2d,
        nn.Conv3d: intrinsic.ConvAdd3d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(conv, add)


def fuse_conv_add_relu(conv, add, relu):
    from ..nn import intrinsic

    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvAddReLU2d,
        nn.Conv3d: intrinsic.ConvAddReLU3d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(conv, add, relu)


def fuse_conv_bn_relu6(conv, bn, relu6, fuse_bn=True):
    from ..nn import intrinsic

    assert (
        conv.training == bn.training == relu6.training
    ), "Conv and BN and relu6 must be in the same mode (train or eval)."
    if not fuse_bn:
        return intrinsic.ConvBNReLU62d(conv, bn, relu6)

    fused_conv = fuse_conv_bn(conv, bn)
    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvReLU62d,
        nn.Conv3d: intrinsic.ConvReLU63d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(fused_conv, relu6)


def fuse_conv_bn_add_relu6(conv, bn, add, relu6, fuse_bn=True):
    assert (
        conv.training == bn.training == add.training
    ), "Conv and BN and add must be in the same mode (train or eval)."
    from ..nn import intrinsic

    if not fuse_bn:
        return intrinsic.ConvBNAddReLU62d(conv, bn, add, relu6)

    fused_conv = fuse_conv_bn(conv, bn)
    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvAddReLU62d,
        nn.Conv3d: intrinsic.ConvAddReLU63d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(fused_conv, add, relu6)


def fuse_conv_add_relu6(conv, add, relu6):
    from ..nn import intrinsic

    map_to_fused_module = {
        nn.Conv2d: intrinsic.ConvAddReLU62d,
        nn.Conv3d: intrinsic.ConvAddReLU63d,
    }
    fused_module = map_to_fused_module.get(type(conv), None)
    return fused_module(conv, add, relu6)


def fuse_conv_transpose2d_relu(conv_transpose2d, relu):
    from ..nn import intrinsic

    return intrinsic.ConvTransposeReLU2d(conv_transpose2d, relu)


def fuse_conv_transpose2d_relu6(conv_transpose2d, relu6):
    from ..nn import intrinsic

    return intrinsic.ConvTransposeReLU62d(conv_transpose2d, relu6)


def fuse_conv_transpose2d_add(conv_transpose2d, add):
    from ..nn import intrinsic

    return intrinsic.ConvTransposeAdd2d(conv_transpose2d, add)


def fuse_conv_transpose2d_add_relu(conv_transpose2d, add, relu):
    from ..nn import intrinsic

    return intrinsic.ConvTransposeAddReLU2d(conv_transpose2d, add, relu)


def fuse_conv_transpose2d_add_relu6(conv_transpose2d, add, relu6):
    from ..nn import intrinsic

    return intrinsic.ConvTransposeAddReLU62d(conv_transpose2d, add, relu6)


def fuse_conv_transpose2d_bn(conv_transpose2d, bn):
    r"""Given the conv_transpose2d and bn modules, fuses them and always
        returns the fused-conv module.

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.ConvTranspose2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_transpose2d_bn(m1, b1)
    """
    assert (
        conv_transpose2d.training == bn.training
    ), "ConvTranspose2d and BN must be in the same mode (train or eval)."
    assert (
        bn.num_features == conv_transpose2d.out_channels
    ), "Output channel of Conv2d must match num_features of BatchNorm2d"
    fused_conv_transpose2d = copy.deepcopy(conv_transpose2d)
    weight = fused_conv_transpose2d.weight
    groups = fused_conv_transpose2d.groups
    wsize = weight.size()
    w = weight.reshape(
        (groups, wsize[0] // groups, wsize[1], wsize[2], wsize[3])
    )
    w = w.transpose(dim0=1, dim1=2)
    w = w.reshape(wsize[1] * groups, wsize[0] // groups, wsize[2], wsize[3])
    fused_conv_transpose2d.weight = nn.parameter.Parameter(w, True)
    (
        fused_conv_transpose2d.weight,
        fused_conv_transpose2d.bias,
    ) = torch.nn.utils.fuse_conv_bn_weights(
        fused_conv_transpose2d.weight,
        fused_conv_transpose2d.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )
    weight = fused_conv_transpose2d.weight
    w = weight.reshape(
        (groups, wsize[1], wsize[0] // groups, wsize[2], wsize[3])
    )
    w = w.transpose(dim0=1, dim1=2)
    fused_conv_transpose2d.weight = nn.parameter.Parameter(
        w.reshape((wsize), True)
    )
    return fused_conv_transpose2d


def fuse_conv_transpose2d_bn_relu(conv_transpose2d, bn, relu):
    assert (
        conv_transpose2d.training == bn.training == relu.training
    ), "ConvTrnaspose2d and BN and relu must be in the same mode \
        (train or eval)."
    fused_conv_transpose2d = fuse_conv_transpose2d_bn(conv_transpose2d, bn)
    from ..nn import intrinsic

    return intrinsic.ConvTransposeReLU2d(fused_conv_transpose2d, relu)


def fuse_conv_transpose2d_bn_relu6(conv_transpose2d, bn, relu6):
    assert (
        conv_transpose2d.training == bn.training == relu6.training
    ), "ConvTrnaspose2d and BN and relu6 must be in the same mode \
        (train or eval)."
    fused_conv_transpose2d = fuse_conv_transpose2d_bn(conv_transpose2d, bn)
    from ..nn import intrinsic

    return intrinsic.ConvTransposeReLU62d(fused_conv_transpose2d, relu6)


def fuse_conv_transpose2d_bn_add(conv_transpose2d, bn, add):
    assert (
        conv_transpose2d.training == bn.training == add.training
    ), "ConvTranspose2d and BN and add must be in the same mode \
        (train or eval)."
    fused_conv_transpose2d = fuse_conv_transpose2d_bn(conv_transpose2d, bn)
    from ..nn import intrinsic

    return intrinsic.ConvTransposeAdd2d(fused_conv_transpose2d, add)


def fuse_conv_transpose2d_bn_add_relu(conv_transpose2d, bn, add, relu):
    assert (
        conv_transpose2d.training == bn.training == relu.training
    ), "ConvTranspose2d and BN and relu must be in the same mode \
        (train or eval)."
    fused_conv_transpose2d = fuse_conv_transpose2d_bn(conv_transpose2d, bn)
    from ..nn import intrinsic

    return intrinsic.ConvTransposeAddReLU2d(fused_conv_transpose2d, add, relu)


def fuse_conv_transpose2d_bn_add_relu6(conv_transpose2d, bn, add, relu6):
    assert (
        conv_transpose2d.training == bn.training == relu6.training
    ), "ConvTranspose2d and BN and relu6 must be in the same mode \
        (train or eval)."
    fused_conv_transpose2d = fuse_conv_transpose2d_bn(conv_transpose2d, bn)
    from ..nn import intrinsic

    return intrinsic.ConvTransposeAddReLU62d(
        fused_conv_transpose2d, add, relu6
    )


def fuse_linear_bn(linear, bn):
    r"""Given the linear and bn modules, fuses them and always returns
        the fused-linear module.

    Args:
        linear: Module instance of type torch.nn.Linear
        bn: Spatial BN instance that needs to be fused with the linear

    Examples::

        >>> m1 = nn.Linear(10, 20)
        >>> b1 = nn.BatchNorm1d(20)
        >>> m2 = fuse_linear_bn(m1, b1)
    """
    assert (
        linear.training == bn.training
    ), "Linear and BN both must be in the same mode (train or eval)."
    assert (
        bn.num_features == linear.out_features
    ), "Out features of Linear must match num_features of BatchNorm"
    fused_linear = copy.deepcopy(linear)
    (
        fused_linear.weight,
        fused_linear.bias,
    ) = torch.nn.utils.fusion.fuse_linear_bn_weights(
        linear.weight,
        linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )
    return fused_linear


def fuse_linear_add(linear, add):
    from ..nn import intrinsic

    assert (
        linear.training == add.training
    ), "Linear and add both must be in the same mode (train or eval)."
    return intrinsic.LinearAdd(linear, add)


def fuse_linear_relu(linear, relu):
    from ..nn import intrinsic

    assert (
        linear.training == relu.training
    ), "Linear and relu both must be in the same mode (train or eval)."
    return nni.LinearReLU(linear, relu)


def fuse_linear_relu6(linear, relu6):
    from ..nn import intrinsic

    assert (
        linear.training == relu6.training
    ), "Linear and BN and relu both must be in the same mode (train or eval)."
    return intrinsic.LinearReLU6(linear, relu6)


def fuse_linear_bn_add(linear, bn, add):
    from ..nn import intrinsic

    assert (
        linear.training == bn.training == add.training
    ), "Linear and BN and add must be in the same mode (train or eval)."
    fused_linear = fuse_linear_bn(linear, bn)
    return intrinsic.LinearAdd(fused_linear, add)


def fuse_linear_bn_relu(linear, bn, relu):
    from ..nn import intrinsic

    assert (
        linear.training == bn.training == relu.training
    ), "Linear and BN and relu must be in the same mode (train or eval)."
    fused_linear = fuse_linear_bn(linear, bn)
    return intrinsic.LinearReLU(fused_linear, relu)


def fuse_linear_bn_relu6(linear, bn, relu6):
    from ..nn import intrinsic

    assert (
        linear.training == bn.training == relu6.training
    ), "Linear and BN and relu6 must be in the same mode (train or eval)."
    fused_linear = fuse_linear_bn(linear, bn)
    return intrinsic.LinearReLU6(fused_linear, relu6)


def fuse_linear_bn_add_relu(linear, bn, add, relu):
    from ..nn import intrinsic

    assert linear.training == bn.training == add.training == relu.training, (
        "Linear and BN and add and relu must be in the same mode (train"
        "or eval)."
    )
    fused_linear = fuse_linear_bn(linear, bn)
    return intrinsic.LinearAddReLU(fused_linear, add, relu)


def fuse_linear_bn_add_relu6(linear, bn, add, relu6):
    from ..nn import intrinsic

    assert linear.training == bn.training == add.training == relu6.training, (
        "Linear and BN and add and relu6 must be in the same mode (train"
        "or eval)."
    )
    fused_linear = fuse_linear_bn(linear, bn)
    return intrinsic.LinearAddReLU6(fused_linear, add, relu6)


def fuse_linear_add_relu(linear, add, relu):
    from ..nn import intrinsic

    assert linear.training == add.training == relu.training, (
        "Linear and BN and add and relu must be in the same mode (train"
        "or eval)."
    )

    return intrinsic.LinearAddReLU(linear, add, relu)


def fuse_linear_add_relu6(linear, add, relu6):
    from ..nn import intrinsic

    assert linear.training == add.training == relu6.training, (
        "Linear and BN and add and relu6 must be in the same mode (train"
        "or eval)."
    )

    return intrinsic.LinearAddReLU6(linear, add, relu6)


def fuse_deform_conv(*mod_list):
    from ..nn import intrinsic

    # ignore fuse_bn param because not supported
    if isinstance(mod_list[-1], bool):
        assert mod_list[-1] is True, "DeformConv do not support with bn"
        mod_list = mod_list[:-1]

    assert len(mod_list) >= 2, "No module to fuse in {}".format(mod_list)
    assert type(mod_list[0]) == ops.DeformConv2d

    # fuse bn
    if type(mod_list[1]) in (nn.BatchNorm2d, nn.SyncBatchNorm):
        deform_conv2dbn = fuse_conv_bn(mod_list[0], mod_list[1])
        mod_list = (deform_conv2dbn,) + mod_list[2:]

    if len(mod_list) == 1:
        return mod_list[0]
    elif len(mod_list) == 2:
        if type(mod_list[1]) in (nnq.FloatFunctional, HFloatFunctional):
            return intrinsic.DeformConvAdd2d(*mod_list)
        else:
            assert type(mod_list[1]) in (nn.ReLU, nn.ReLU6)
            if type(mod_list[1]) == nn.ReLU:
                return intrinsic.DeformConvReLU2d(*mod_list)
            else:
                return intrinsic.DeformConvReLU62d(*mod_list)
    elif len(mod_list) == 3:
        if type(mod_list[2]) == nn.ReLU:
            return intrinsic.DeformConvAddReLU2d(*mod_list)
        else:
            return intrinsic.DeformConvAddReLU62d(*mod_list)
    else:
        raise ValueError("Invalid fuse list: {}".format(mod_list))


OP_LIST_TO_FUSER_MAPPING: Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.SyncBatchNorm): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.SyncBatchNorm, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU6): fuse_conv_bn_relu6,
    (nn.Conv2d, nn.SyncBatchNorm, nn.ReLU6): fuse_conv_bn_relu6,
    (nn.Conv2d, nn.ReLU): nni.ConvReLU2d,
    (nn.Conv2d, nn.ReLU6): fuse_conv_relu6,
    # user guarantee FloatFunctional is add
    (nn.Conv2d, nn.BatchNorm2d, nnq.FloatFunctional): fuse_conv_bn_add,
    (nn.Conv2d, nn.SyncBatchNorm, nnq.FloatFunctional): fuse_conv_bn_add,
    (nn.Conv2d, nn.BatchNorm2d, HFloatFunctional): fuse_conv_bn_add,
    (nn.Conv2d, nn.SyncBatchNorm, HFloatFunctional): fuse_conv_bn_add,
    (
        nn.Conv2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_conv_bn_add_relu,
    (
        nn.Conv2d,
        nn.SyncBatchNorm,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_conv_bn_add_relu,
    (
        nn.Conv2d,
        nn.BatchNorm2d,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_conv_bn_add_relu,
    (
        nn.Conv2d,
        nn.SyncBatchNorm,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_conv_bn_add_relu,
    (
        nn.Conv2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_conv_bn_add_relu6,
    (
        nn.Conv2d,
        nn.SyncBatchNorm,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_conv_bn_add_relu6,
    (
        nn.Conv2d,
        nn.BatchNorm2d,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_conv_bn_add_relu6,
    (
        nn.Conv2d,
        nn.SyncBatchNorm,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_conv_bn_add_relu6,
    (nn.Conv2d, nnq.FloatFunctional): fuse_conv_add,
    (nn.Conv2d, HFloatFunctional): fuse_conv_add,
    (nn.Conv2d, nnq.FloatFunctional, nn.ReLU): fuse_conv_add_relu,
    (nn.Conv2d, HFloatFunctional, nn.ReLU): fuse_conv_add_relu,
    (nn.Conv2d, nnq.FloatFunctional, nn.ReLU6): fuse_conv_add_relu6,
    (nn.Conv2d, HFloatFunctional, nn.ReLU6): fuse_conv_add_relu6,
    (nn.ConvTranspose2d, nn.ReLU): fuse_conv_transpose2d_relu,
    (nn.ConvTranspose2d, nn.ReLU6): fuse_conv_transpose2d_relu6,
    (nn.ConvTranspose2d, nnq.FloatFunctional): fuse_conv_transpose2d_add,
    (nn.ConvTranspose2d, HFloatFunctional): fuse_conv_transpose2d_add,
    (
        nn.ConvTranspose2d,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_conv_transpose2d_add_relu,
    (
        nn.ConvTranspose2d,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_conv_transpose2d_add_relu,
    (
        nn.ConvTranspose2d,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_conv_transpose2d_add_relu6,
    (
        nn.ConvTranspose2d,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_conv_transpose2d_add_relu6,
    (nn.ConvTranspose2d, nn.BatchNorm2d): fuse_conv_transpose2d_bn,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        nn.ReLU,
    ): fuse_conv_transpose2d_bn_relu,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        nn.ReLU6,
    ): fuse_conv_transpose2d_bn_relu6,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
    ): fuse_conv_transpose2d_bn_add,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        HFloatFunctional,
    ): fuse_conv_transpose2d_bn_add,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_conv_transpose2d_bn_add_relu,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_conv_transpose2d_bn_add_relu,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_conv_transpose2d_bn_add_relu6,
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_conv_transpose2d_bn_add_relu6,
    # Conv3d
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.ReLU): nni.ConvReLU3d,
    (nn.Conv3d, nn.ReLU6): fuse_conv_relu6,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d, nnq.FloatFunctional): fuse_conv_bn_add,
    (nn.Conv3d, nn.BatchNorm3d, HFloatFunctional): fuse_conv_bn_add,
    (
        nn.Conv3d,
        nn.BatchNorm3d,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_conv_bn_add_relu,
    (
        nn.Conv3d,
        nn.BatchNorm3d,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_conv_bn_add_relu,
    (nn.Conv3d, nnq.FloatFunctional): fuse_conv_add,
    (nn.Conv3d, HFloatFunctional): fuse_conv_add,
    (nn.Conv3d, nnq.FloatFunctional, nn.ReLU): fuse_conv_add_relu,
    (nn.Conv3d, HFloatFunctional, nn.ReLU): fuse_conv_add_relu,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU6): fuse_conv_bn_relu6,
    (
        nn.Conv3d,
        nn.BatchNorm3d,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_conv_bn_add_relu6,
    (
        nn.Conv3d,
        nn.BatchNorm3d,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_conv_bn_add_relu6,
    (nn.Conv3d, nnq.FloatFunctional, nn.ReLU6): fuse_conv_add_relu6,
    (nn.Conv3d, HFloatFunctional, nn.ReLU6): fuse_conv_add_relu6,
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nnq.FloatFunctional): fuse_linear_add,
    (nn.Linear, HFloatFunctional): fuse_linear_add,
    (nn.Linear, nn.ReLU): fuse_linear_relu,
    (nn.Linear, nn.ReLU6): fuse_linear_relu6,
    (nn.Linear, nn.BatchNorm1d, nnq.FloatFunctional): fuse_linear_bn_add,
    (nn.Linear, nn.BatchNorm1d, HFloatFunctional): fuse_linear_bn_add,
    (nn.Linear, nn.BatchNorm1d, nn.ReLU): fuse_linear_bn_relu,
    (nn.Linear, nn.BatchNorm1d, nn.ReLU6): fuse_linear_bn_relu6,
    (
        nn.Linear,
        nn.BatchNorm1d,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_linear_bn_add_relu,
    (
        nn.Linear,
        nn.BatchNorm1d,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_linear_bn_add_relu,
    (
        nn.Linear,
        nn.BatchNorm1d,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_linear_bn_add_relu6,
    (
        nn.Linear,
        nn.BatchNorm1d,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_linear_bn_add_relu6,
    (nn.Linear, nnq.FloatFunctional, nn.ReLU): fuse_linear_add_relu,
    (nn.Linear, HFloatFunctional, nn.ReLU): fuse_linear_add_relu,
    (nn.Linear, nnq.FloatFunctional, nn.ReLU6): fuse_linear_add_relu6,
    (nn.Linear, HFloatFunctional, nn.ReLU6): fuse_linear_add_relu6,
    # DeformConv2d
    (ops.DeformConv2d, nn.BatchNorm2d): fuse_deform_conv,
    (ops.DeformConv2d, nn.SyncBatchNorm): fuse_deform_conv,
    (ops.DeformConv2d, nn.BatchNorm2d, nn.ReLU): fuse_deform_conv,
    (ops.DeformConv2d, nn.SyncBatchNorm, nn.ReLU): fuse_deform_conv,
    (ops.DeformConv2d, nn.BatchNorm2d, nn.ReLU6): fuse_deform_conv,
    (ops.DeformConv2d, nn.SyncBatchNorm, nn.ReLU6): fuse_deform_conv,
    (ops.DeformConv2d, nn.ReLU): fuse_deform_conv,
    (ops.DeformConv2d, nn.ReLU6): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.SyncBatchNorm,
        nnq.FloatFunctional,
    ): fuse_deform_conv,
    (ops.DeformConv2d, nn.BatchNorm2d, HFloatFunctional): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.SyncBatchNorm,
        HFloatFunctional,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.SyncBatchNorm,
        nnq.FloatFunctional,
        nn.ReLU,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.BatchNorm2d,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.SyncBatchNorm,
        HFloatFunctional,
        nn.ReLU,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.BatchNorm2d,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.SyncBatchNorm,
        nnq.FloatFunctional,
        nn.ReLU6,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.BatchNorm2d,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_deform_conv,
    (
        ops.DeformConv2d,
        nn.SyncBatchNorm,
        HFloatFunctional,
        nn.ReLU6,
    ): fuse_deform_conv,
    (ops.DeformConv2d, nnq.FloatFunctional): fuse_deform_conv,
    (ops.DeformConv2d, HFloatFunctional): fuse_deform_conv,
    (ops.DeformConv2d, nnq.FloatFunctional, nn.ReLU): fuse_deform_conv,
    (ops.DeformConv2d, HFloatFunctional, nn.ReLU): fuse_deform_conv,
    (ops.DeformConv2d, nnq.FloatFunctional, nn.ReLU6): fuse_deform_conv,
    (ops.DeformConv2d, HFloatFunctional, nn.ReLU6): fuse_deform_conv,
}


def get_op_list_to_fuser_mapping():
    return OP_LIST_TO_FUSER_MAPPING


def _get_op_list_to_fuser_method(types):
    return get_op_list_to_fuser_mapping().get(types, None)


def _replace_fused_modules(mod_list, fused, fused_idx=0):
    """
    replace mod list with fused module and Identity
    """

    new_mod: List[Optional[nn.Module]] = [None] * len(mod_list)
    for i in range(0, len(mod_list)):
        if i == fused_idx:
            new_mod[i] = fused
        else:
            identity = Identity()
            identity.training = mod_list[0].training
            new_mod[i] = identity
    return new_mod


def _replace_conv_shared_fused_modules(mod_list, fused, fused_idx=0):
    """
    replace mod list with fused module and Identity, except conv
    """

    new_mod: List[Optional[nn.Module]] = [None] * len(mod_list)
    # keep conv module
    new_mod[0] = mod_list[0]
    for i in range(1, len(mod_list)):
        if i == fused_idx:
            new_mod[i] = fused
        else:
            identity = Identity()
            identity.training = mod_list[0].training
            new_mod[i] = identity
    return new_mod


def _fuse_known_modules(mod_list, shared_conv=False, fuse_bn=True):
    types = tuple(type(m) for m in mod_list)
    fuser_method = _get_op_list_to_fuser_method(types)
    if fuser_method is None:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))
    if (
        torch.nn.ConvTranspose2d in types
        and torch.nn.BatchNorm2d in types
        and not fuse_bn
    ):
        warnings.warn(
            "ConvTranspose2d is not supported for training with bn temporarily"
        )
    if (
        torch.nn.BatchNorm2d in types or torch.nn.SyncBatchNorm in types
    ) and torch.nn.ConvTranspose2d not in types:
        fused = fuser_method(*mod_list, fuse_bn)
    else:
        fused = fuser_method(*mod_list)
    # NOTE: forward hooks not processed in the two following for loops will
    # be lost after the fusion
    # Move pre forward hooks of the base module to resulting fused module
    for handle_id, pre_hook_fn in mod_list[0]._forward_pre_hooks.items():
        fused.register_forward_pre_hook(pre_hook_fn)
        del mod_list[0]._forward_pre_hooks[handle_id]
    # Move post forward hooks of the last module to resulting fused module
    for handle_id, hook_fn in mod_list[-1]._forward_hooks.items():
        fused.register_forward_hook(hook_fn)
        del mod_list[-1]._forward_hooks[handle_id]

    from ..nn import intrinsic

    # normal fused module
    fused_idx = 0
    if (
        type(fused)
        in (
            intrinsic.ConvAdd2d,
            intrinsic.ConvAdd3d,
            intrinsic.ConvTransposeAdd2d,
            intrinsic.ConvBNAdd2d,
            intrinsic.LinearAdd,
            intrinsic.DeformConvAdd2d,
        )
        or shared_conv
    ):
        fused_idx = len(mod_list) - 1
    elif type(fused) in (
        intrinsic.ConvAddReLU2d,
        intrinsic.ConvAddReLU62d,
        intrinsic.ConvTransposeAddReLU2d,
        intrinsic.ConvTransposeAddReLU62d,
        intrinsic.ConvAddReLU3d,
        intrinsic.ConvAddReLU63d,
        intrinsic.ConvBNAddReLU2d,
        intrinsic.ConvBNAddReLU62d,
        intrinsic.LinearAddReLU,
        intrinsic.LinearAddReLU6,
        intrinsic.DeformConvAddReLU2d,
        intrinsic.DeformConvAddReLU62d,
    ):
        fused_idx = len(mod_list) - 2

    if not shared_conv:
        new_mod = _replace_fused_modules(mod_list, fused, fused_idx)
    else:
        # keep conv module
        new_mod = _replace_conv_shared_fused_modules(
            mod_list, fused, fused_idx
        )
    return new_mod


def fuse_known_modules(mod_list, additional_fuser_method_mapping=None):
    """
    Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    conv, bn, add
    conv, bn, add, relu
    conv, add
    conv, add, relu
    linear, bn
    linear, bn, relu
    linear, relu
    linear, bn, add
    linear, bn, add, relu
    linear, add
    linear, add, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
    if (
        get_qat_mode() == QATMode.WithBN
        or get_qat_mode() == QATMode.WithBNReverseFold
    ):
        return _fuse_known_modules(mod_list, shared_conv=False, fuse_bn=False)
    else:
        return _fuse_known_modules(mod_list)


def _fuse_conv_shared_modules(mod_list, additional_fuser_method_mapping=None):
    return _fuse_known_modules(mod_list, shared_conv=True)


def fuse_conv_shared_modules(model, modules_to_fuse, inplace=False):
    r"""Returns a list of modules that fuses the operations specified
     in the input module list when conv module is shared .

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu

    For these sequences, replaces the last item in the list
    with the fused module, replacing the rest of the modules
    with identity.
    Arguments:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse.
        inplace: bool specifying if fusion happens in place on the model,
                 by default a new model is returned
    Returns:
        model with fused modules. A new copy is created if inplace=True.

    Examples::

            >>> m = myModel()
            >>> # m is a module which conv is shared
            >>> modules_to_fuse = [ ['conv', 'bn1', 'relu1'], ['conv', 'bn2']]
            >>> fused_m = changan.quantization.fuse_conv_shared_modules(m, modules_to_fuse) # noqa
            >>> output = fused_m(input)

    """
    from torch.quantization.fuse_modules import _set_module, fuse_modules

    assert not all(
        isinstance(module_element, str) for module_element in modules_to_fuse
    ), "module_to_fuse must be a list of lists"
    # norm fuse modules
    model = fuse_modules(
        model, modules_to_fuse, inplace, _fuse_conv_shared_modules
    )
    # replace conv with Identity
    for module_list in modules_to_fuse:
        _set_module(model, module_list[0], Identity())
    return model
