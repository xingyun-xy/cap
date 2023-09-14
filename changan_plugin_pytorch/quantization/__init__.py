"""
using 'QuantWrapper', 'QuantStub', 'DeQuantStub'
"""
import warnings

# from hbdk.torch_script.tools import (
#     check_model,
#     compile_model,
#     export_hbir,
#     perf_model,
#     visualize_model,
# )
from changan_plugin_pytorch.utils.mprop import mproperty

from ._learnable_fake_quantize import _LearnableFakeQuantize
from .fake_quantize import (
    FakeQuantize,
    default_4bit_fake_quant,
    default_8bit_fake_quant,
    default_16bit_fake_quant,
    default_calib_fake_quant,
    default_uint4_fake_quant,
    default_weight_4bit_fake_quant,
    default_weight_8bit_fake_quant,
    per_channel_8bit_fake_quant,
)
from .fuse_modules import fuse_conv_shared_modules, fuse_known_modules
from .misc import QATMode, get_qat_mode, set_qat_mode
from .observer import *
from .pact_fake_quantize import PACTFakeQuantize
from .qconfig import (
    get_default_calib_qconfig,
    get_default_qat_out_qconfig,
    get_default_qat_qconfig,
    per_channel_qat_8bit_qconfig,
)
from .quantize import convert, prepare_calibration, prepare_qat
from .quantize_fx import fuse_fx, convert_fx, prepare_qat_fx
from .stubs import QuantStub

__all__ = [
    # qinfo
    # "qint8",
    # "qint16",
    # "qint32",
    # "qinfo",
    # "QTensor",
    # defalut qat qconfig
    "get_default_qat_qconfig",
    "get_default_qat_out_qconfig",
    "get_default_calib_qconfig",
    # fake quantize info
    "FakeQuantize",
    "default_uint4_fake_quant",
    "default_8bit_fake_quant",
    "per_channel_8bit_fake_quant",
    "default_weight_8bit_fake_quant",
    "default_4bit_fake_quant",
    "default_weight_4bit_fake_quant",
    "default_16bit_fake_quant",
    "default_calib_fake_quant",
    "per_channel_qat_8bit_qconfig",
    # lsq
    "_LearnableFakeQuantize",
    # pact
    "PACTFakeQuantize",
    # observer
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "ClipObserver",
    "FixedScaleObserver",
    # fuse modules
    "fuse_known_modules",
    "fuse_conv_shared_modules",
    # prepare and convert
    "prepare_qat",
    "prepare_calibration",
    "convert",
    # march
    # "March",
    # "with_march",
    # hbdk utils
    "export_hbir",
    "check_model",
    "compile_model",
    "perf_model",
    "visualize_model",
    "QuantStub",
    "set_qat_mode",
    "get_qat_mode",
    "QATMode",
]


@mproperty
def March(mod):
    warnings.warn(
        "changan_plugin_pytorch.quantization.March is deprecated, "
        + "please use changan_plugin_pytorch.march.March",
        DeprecationWarning,
        stacklevel=2,
    )
    from changan_plugin_pytorch.march import March

    return March


@mproperty
def march(mod):
    from changan_plugin_pytorch.march import get_march

    warnings.warn(
        "changan_plugin_pytorch.quantization.march is deprecated, "
        + "please use changan_plugin_pytorch.march.get_march",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_march()


@march.setter
def march(mod, value):
    from changan_plugin_pytorch.march import set_march

    warnings.warn(
        "changan_plugin_pytorch.quantization.march is deprecated, "
        + "please use changan_plugin_pytorch.march.set_march",
        DeprecationWarning,
        stacklevel=2,
    )
    set_march(value)


@mproperty
def QTensor(mod):
    warnings.warn(
        "changan_plugin_pytorch.quantization.QTensor is deprecated, "
        + "please use changan_plugin_pytorch.qtensor.QTensor",
        DeprecationWarning,
        stacklevel=2,
    )
    from changan_plugin_pytorch.qtensor import QTensor

    return QTensor


@mproperty
def qinfo(mod):
    warnings.warn(
        "changan_plugin_pytorch.quantization.qinfo.qinfo is deprecated, "
        + "please use changan_plugin_pytorch.dtype.qinfo",
        DeprecationWarning,
        stacklevel=2,
    )
    from changan_plugin_pytorch.dtype import qinfo

    return qinfo


@mproperty
def qint8(mod):
    warnings.warn(
        "changan_plugin_pytorch.quantization.qinfo.qint8 is deprecated, "
        + "please use changan_plugin_pytorch.dtype.qint8",
        DeprecationWarning,
        stacklevel=2,
    )
    from changan_plugin_pytorch.dtype import qint8

    return qint8


@mproperty
def qint16(mod):
    warnings.warn(
        "changan_plugin_pytorch.quantization.qinfo.qint16 is deprecated, "
        + "please use changan_plugin_pytorch.dtype.qint16",
        DeprecationWarning,
        stacklevel=2,
    )
    from changan_plugin_pytorch.dtype import qint16

    return qint16


@mproperty
def qint32(mod):
    warnings.warn(
        "changan_plugin_pytorch.quantization.qinfo.qint32 is deprecated, "
        + "please use changan_plugin_pytorch.dtype.qint32",
        DeprecationWarning,
        stacklevel=2,
    )
    from changan_plugin_pytorch.dtype import qint32

    return qint32
