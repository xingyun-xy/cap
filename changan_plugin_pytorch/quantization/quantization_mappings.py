import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq
from changan_plugin_pytorch import nn as nnf
from changan_plugin_pytorch.nn import intrinsic, qat, quantized
from torch import nn
from torch.quantization import DeQuantStub, QuantStub
from torch.quantization.quantization_mappings import (
    _INCLUDE_QCONFIG_PROPAGATE_LIST,
)
from torchvision import ops as nnv

from . import stubs


# Map for swapping float module to qat modules
QAT_MODULE_MAPPINGS = {
    stubs.QuantStub: qat.QuantStub,
    QuantStub: qat.QuantStub,
    DeQuantStub: qat.DeQuantStub,
    nn.Conv2d: qat.Conv2d,
    # Intrinsic modules:
    nni.ConvReLU2d: qat.ConvReLU2d,
    intrinsic.ConvAdd2d: qat.ConvAdd2d,
    intrinsic.ConvAddReLU2d: qat.ConvAddReLU2d,
    nnq.FloatFunctional: quantized.FloatFunctional,
    quantized.FloatFunctional: quantized.FloatFunctional,
    nn.AvgPool2d: qat.AvgPool2d,
    nn.AdaptiveAvgPool2d: qat.AdaptiveAvgPool2d,
    nn.MaxPool2d: qat.MaxPool2d,
    nnf.Interpolate: qat.Interpolate,
    nn.ConstantPad1d: qat.ConstantPad1d,
    nn.ConstantPad2d: qat.ConstantPad2d,
    nn.ConstantPad3d: qat.ConstantPad3d,
    nn.ZeroPad2d: qat.ZeroPad2d,
    nn.ReplicationPad1d: qat.ReplicationPad1d,
    nn.ReplicationPad2d: qat.ReplicationPad2d,
    nn.ReplicationPad3d: qat.ReplicationPad3d,
    nn.Dropout2d: qat.Dropout2d,
    nn.ReLU: qat.ReLU,
    nn.PReLU: qat.PReLU,
    nnv.RoIAlign: qat.RoIAlign,
    nnf.GridSample: qat.GridSample,
    nnf.LookUpTable: qat.LookUpTable,
    nn.Sigmoid: qat.Sigmoid,
    nnf.BaseGridGenerator: qat.BaseGridGenerator,
    nnf.DetectionPostProcessV1: qat.DetectionPostProcessV1,
    nn.Softmax: qat.Softmax,
    nn.ConvTranspose2d: qat.ConvTranspose2d,
    intrinsic.ConvTransposeAdd2d: qat.ConvTransposeAdd2d,
    intrinsic.ConvTransposeReLU2d: qat.ConvTransposeReLU2d,
    intrinsic.ConvTransposeAddReLU2d: qat.ConvTransposeAddReLU2d,
    nn.SiLU: qat.SiLU,
    nn.Tanh: qat.Tanh,
    nn.Upsample: qat.Upsample,
    nn.UpsamplingNearest2d: qat.UpsamplingNearest2d,
    nn.UpsamplingBilinear2d: qat.UpsamplingBilinear2d,
    nn.Dropout: qat.Dropout,
    intrinsic.ConvTransposeReLU62d: qat.ConvTransposeReLU62d,
    intrinsic.ConvTransposeAddReLU62d: qat.ConvTransposeAddReLU62d,
    intrinsic.ConvReLU62d: qat.ConvReLU62d,
    intrinsic.ConvAddReLU62d: qat.ConvAddReLU62d,
    nnf.MultiScaleRoIAlign: qat.MultiScaleRoIAlign,
    nn.BatchNorm2d: qat.BatchNorm2d,
    nn.GELU: qat.GELU,
    nn.LayerNorm: qat.LayerNorm,
    nn.Conv3d: qat.Conv3d,
    nni.ConvReLU3d: qat.ConvReLU3d,
    intrinsic.ConvAdd3d: qat.ConvAdd3d,
    intrinsic.ConvAddReLU3d: qat.ConvAddReLU3d,
    intrinsic.ConvReLU63d: qat.ConvReLU63d,
    intrinsic.ConvAddReLU63d: qat.ConvAddReLU63d,
    nnf.Correlation: qat.Correlation,
    nnf.SegmentLUT: qat.SegmentLUT,
    nnf.LayerNorm: qat.LayerNorm,
    intrinsic.ConvBN2d: qat.ConvBN2d,
    intrinsic.ConvBNAdd2d: qat.ConvBNAdd2d,
    intrinsic.ConvBNAddReLU2d: qat.ConvBNAddReLU2d,
    intrinsic.ConvBNReLU2d: qat.ConvBNReLU2d,
    intrinsic.ConvBNReLU62d: qat.ConvBNReLU62d,
    intrinsic.ConvBNAddReLU62d: qat.ConvBNAddReLU62d,
    nn.AdaptiveAvgPool1d: qat.AdaptiveAvgPool1d,
    nn.GLU: qat.GLU,
    nn.LeakyReLU: qat.LeakyReLU,
    nnf.Pow: qat.Pow,
    nn.LSTMCell: qat.LSTMCell,
    nn.Linear: qat.Linear,
    intrinsic.LinearAdd: qat.LinearAdd,
    intrinsic.LinearAddReLU: qat.LinearAddReLU,
    intrinsic.LinearReLU: qat.LinearReLU,
    intrinsic.LinearReLU6: qat.LinearReLU6,
    intrinsic.LinearAddReLU6: qat.LinearAddReLU6,
    nn.LSTM: qat.LSTM,
    nnv.DeformConv2d: qat.DeformConv2d,
    intrinsic.DeformConvReLU2d: qat.DeformConvReLU2d,
    intrinsic.DeformConvReLU62d: qat.DeformConvReLU62d,
    intrinsic.DeformConvAdd2d: qat.DeformConvAdd2d,
    intrinsic.DeformConvAddReLU2d: qat.DeformConvAddReLU2d,
    intrinsic.DeformConvAddReLU62d: qat.DeformConvAddReLU62d,
    nnf.Exp: qat.Exp,
    nnf.Div: qat.Div,
}


def get_qat_module_mappings():
    """Get module mapping for quantization aware training"""
    return QAT_MODULE_MAPPINGS


# Map for swapping qat module to quantized ones
QUANT_MODULE_MAPPINGS = {
    qat.QuantStub: quantized.Quantize,
    qat.DeQuantStub: quantized.DeQuantize,
    # add this mapping to make fx treat Identity like quantized op
    nnf.Identity: nnf.Identity,
    # Wrapper Modules:
    quantized.FloatFunctional: quantized.QFunctional,
    # Intrinsic modules:
    qat.ConvReLU2d: quantized.ConvReLU2d,
    qat.ConvAdd2d: quantized.ConvAdd2d,
    qat.ConvAddReLU2d: quantized.ConvAddReLU2d,
    qat.Conv2d: quantized.Conv2d,
    qat.AvgPool2d: quantized.AvgPool2d,
    qat.AdaptiveAvgPool2d: quantized.AdaptiveAvgPool2d,
    qat.MaxPool2d: quantized.MaxPool2d,
    qat.Interpolate: quantized.Interpolate,
    qat.ConstantPad1d: quantized.ConstantPad1d,
    qat.ConstantPad2d: quantized.ConstantPad2d,
    qat.ConstantPad3d: quantized.ConstantPad3d,
    qat.ZeroPad2d: quantized.ZeroPad2d,
    qat.ReplicationPad1d: quantized.ReplicationPad1d,
    qat.ReplicationPad2d: quantized.ReplicationPad2d,
    qat.ReplicationPad3d: quantized.ReplicationPad3d,
    qat.RoIAlign: quantized.RoIAlign,
    qat.GridSample: quantized.GridSample,
    qat.Dropout2d: nnf.Identity,
    qat.ReLU: quantized.ReLU,
    qat.PReLU: quantized.PReLU,
    qat.LookUpTable: quantized.LookUpTable,
    qat.Sigmoid: quantized.Sigmoid,
    qat.BaseGridGenerator: quantized.BaseGridGenerator,
    qat.DetectionPostProcessV1: quantized.DetectionPostProcessV1,
    qat.Softmax: quantized.QuantSoftmax,
    qat.ConvTranspose2d: quantized.ConvTranspose2d,
    qat.ConvTransposeAdd2d: quantized.ConvTransposeAdd2d,
    qat.ConvTransposeReLU2d: quantized.ConvTransposeReLU2d,
    qat.ConvTransposeAddReLU2d: quantized.ConvTransposeAddReLU2d,
    qat.SiLU: quantized.SiLU,
    qat.Tanh: quantized.Tanh,
    qat.Upsample: quantized.Upsample,
    qat.UpsamplingNearest2d: quantized.UpsamplingNearest2d,
    qat.UpsamplingBilinear2d: quantized.UpsamplingBilinear2d,
    qat.Dropout: nnf.Identity,
    qat.ConvReLU62d: quantized.ConvReLU2d,
    qat.ConvAddReLU62d: quantized.ConvAddReLU2d,
    qat.ConvTransposeReLU62d: quantized.ConvTransposeReLU2d,
    qat.ConvTransposeAddReLU62d: quantized.ConvTransposeAddReLU2d,
    qat.MultiScaleRoIAlign: quantized.MultiScaleRoIAlign,
    qat.BatchNorm2d: quantized.BatchNorm2d,
    qat.GELU: quantized.GELU,
    qat.ConvReLU3d: quantized.ConvReLU3d,
    qat.ConvAdd3d: quantized.ConvAdd3d,
    qat.ConvAddReLU3d: quantized.ConvAddReLU3d,
    qat.Conv3d: quantized.Conv3d,
    qat.ConvReLU63d: quantized.ConvReLU3d,
    qat.ConvAddReLU63d: quantized.ConvAddReLU3d,
    qat.Correlation: quantized.Correlation,
    qat.SegmentLUT: quantized.SegmentLUT,
    qat.ConvBN2d: quantized.Conv2d,
    qat.ConvBNAdd2d: quantized.ConvAdd2d,
    qat.ConvBNAddReLU2d: quantized.ConvAddReLU2d,
    qat.ConvBNReLU2d: quantized.ConvReLU2d,
    qat.ConvBNReLU62d: quantized.ConvReLU2d,
    qat.ConvBNAddReLU62d: quantized.ConvAddReLU2d,
    qat.AdaptiveAvgPool1d: quantized.AdaptiveAvgPool1d,
    qat.Linear: quantized.Linear,
    qat.LinearAdd: quantized.LinearAdd,
    qat.LinearReLU: quantized.LinearReLU,
    qat.LinearReLU6: quantized.LinearReLU,
    qat.LinearAddReLU: quantized.LinearAddReLU,
    qat.LinearAddReLU6: quantized.LinearAddReLU,
    qat.DeformConv2d: quantized.DeformConv2d,
    qat.DeformConvReLU2d: quantized.DeformConvReLU2d,
    qat.DeformConvReLU62d: quantized.DeformConvReLU2d,
    qat.DeformConvAdd2d: quantized.DeformConvAdd2d,
    qat.DeformConvAddReLU2d: quantized.DeformConvAddReLU2d,
    qat.DeformConvAddReLU62d: quantized.DeformConvAddReLU2d,
    qat.Exp: quantized.Exp,
    qat.Div: quantized.Div,
}


def get_quantized_operator_mappings():
    """Get the quantized operator mapping"""
    return QUANT_MODULE_MAPPINGS


def get_qconfig_propagation_list():
    """Get the list of module types that we'll attach qconfig
    attribute to in prepare
    """
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST = (
        set(get_qat_module_mappings().keys()) | _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return QCONFIG_PROPAGATE_MODULE_CLASS_LIST


def wrap_qat_modules_for_fx():
    from ..utils.fx_helper import wrap

    qm = list(get_qat_module_mappings().values())
    for m in qm:
        wrap(m)
