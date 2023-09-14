from functools import partial

import torch
from changan_plugin_pytorch.dtype import qinfo, qint4, qint8, qint16, quint4
from torch.quantization import QConfig

from ._learnable_fake_quantize import (
    default_4bit_lsq_quant,
    default_8bit_lsq_quant,
    default_16bit_lsq_quant,
    default_uint4_lsq_quant,
    default_weight_4bit_lsq_quant,
    default_weight_8bit_lsq_quant,
    default_weight_16bit_lsq_quant,
)
from .fake_quantize import (
    CalibFakeQuantize,
    default_4bit_fake_quant,
    default_8bit_fake_quant,
    default_16bit_fake_quant,
    default_calib_fake_quant,
    default_uint4_fake_quant,
    default_weight_4bit_fake_quant,
    default_weight_8bit_fake_quant,
    default_weight_calib_fake_quant,
    per_channel_8bit_fake_quant,
    default_weight_16bit_fake_quant,
)
from .pact_fake_quantize import (
    default_4bit_pact_quant,
    default_8bit_pact_quant,
    default_16bit_pact_quant,
    default_uint4_pact_quant,
)

# fake_quantize
default_qat_8bit_qconfig = QConfig(
    activation=default_8bit_fake_quant, weight=default_weight_8bit_fake_quant
)

# This qconfig will make the OP OUTPUT per channel quantized activation.
# Please note that only depthwise_conv, interpolate and add support per channel
# quantized INPUT and OUTPUT. Take care of the next OP when using this qconfig!
# Model trained with this qconfig cannot be compiled now.
per_channel_qat_8bit_qconfig = QConfig(
    activation=per_channel_8bit_fake_quant,
    weight=default_weight_8bit_fake_quant,
)

default_qat_4bit_qconfig = QConfig(
    activation=default_4bit_fake_quant, weight=default_weight_4bit_fake_quant
)
default_qat_out_8bit_qconfig = QConfig(
    activation=None, weight=default_weight_8bit_fake_quant
)

default_qat_out_4bit_qconfig = QConfig(
    activation=None, weight=default_weight_4bit_fake_quant
)

default_calib_qconfig = QConfig(
    activation=default_calib_fake_quant, weight=default_weight_calib_fake_quant
)


def _get_fake_quant(dtype, fake_quant_name, fake_quant_mapping, qkwargs):
    if fake_quant_name is None:
        return None
    assert fake_quant_name in fake_quant_mapping.keys(), (
        "unsupport fake_quant_name" + fake_quant_name
    )
    if qkwargs is not None:
        if "dtype" in qkwargs:
            dtype = qkwargs["dtype"]
        if "quant_min" in qkwargs:
            min = qkwargs["quant_min"]
            assert (
                qinfo(dtype).min == min
            ), f"expect quant_min = {qinfo(dtype).min} but get {min}"
        if "quant_max" in qkwargs:
            max = qkwargs["quant_max"]
            assert (
                qinfo(dtype).max == max
            ), f"expect quant_max = {qinfo(dtype).max} but get {max}"

    assert dtype in fake_quant_mapping[fake_quant_name].keys(), (
        "unsupport dtype " + dtype + " for " + fake_quant_name
    )
    fake_quant = fake_quant_mapping[fake_quant_name][dtype]
    if qkwargs is not None:
        fake_quant = fake_quant.with_args(**qkwargs)
    return fake_quant


def _get_custom_qconfig(
    dtype="qint8",
    weight_dtype="qint8",
    activation_fake_quant="fake_quant",
    weight_fake_quant="fake_quant",
    activation_qkwargs=None,
    weight_qkwargs=None,
    backend="",
):
    activation_fake_quant_mapping = {
        "fake_quant": {
            "qint16": default_16bit_fake_quant,
            "qint8": default_8bit_fake_quant,
            "qint4": default_4bit_fake_quant,
            "quint4": default_uint4_fake_quant,
        },
        "lsq": {
            "qint16": default_16bit_lsq_quant,
            "qint8": default_8bit_lsq_quant,
            "qint4": default_4bit_lsq_quant,
            "quint4": default_uint4_lsq_quant,
        },
        "pact": {
            "qint16": default_16bit_pact_quant,
            "qint8": default_8bit_pact_quant,
            "qint4": default_4bit_pact_quant,
            "quint4": default_uint4_pact_quant,
        },
    }
    weight_fake_quant_mapping = {
        "fake_quant": {
            "qint8": default_weight_8bit_fake_quant,
            "qint4": default_weight_4bit_fake_quant,
            "qint16": default_weight_16bit_fake_quant,
        },
        "lsq": {
            "qint8": default_weight_8bit_lsq_quant,
            "qint4": default_weight_4bit_lsq_quant,
            "qint16": default_weight_16bit_lsq_quant,
        },
        "pact": {
            "qint8": default_weight_8bit_lsq_quant,
            "qint4": default_weight_4bit_lsq_quant,
            "qint16": default_weight_16bit_lsq_quant,
        },
    }
    activation = _get_fake_quant(
        dtype,
        activation_fake_quant,
        activation_fake_quant_mapping,
        activation_qkwargs,
    )
    weight = _get_fake_quant(
        weight_dtype,
        weight_fake_quant,
        weight_fake_quant_mapping,
        weight_qkwargs,
    )
    return QConfig(activation=activation, weight=weight)


def get_default_qat_qconfig(
    dtype="qint8",
    weight_dtype="qint8",
    activation_fake_quant="fake_quant",
    weight_fake_quant="fake_quant",
    activation_qkwargs=None,
    weight_qkwargs=None,
    backend="",
):
    """
    get default qat qconfig


    Args:
        dtype (str): Activation quantization type, the allowable values is
                     qint8 and qint16
        weight_dtype (str): Weight quantization type, the allowable values
                     is qint8 and qint16
        activation_fake_quant (str): FakeQuantize type of activation, default
                                     is fake_quant. Avaliable items is
                                     fake_quant, lsq, pact
        weight_fake_quant (str): FakeQuantize type of weight, default is
                                 fake_quant.Avaliable items is fake_quant, lsq
                                 and pact
        activation_qkwargs(dict): A dict contain activation Observer type, args
                                  of activation FakeQuantize and args of
                                  activation Observer.
        weight_qkwargs(dict): A dict contain weight Observer type, args of
                              weight FakeQuantize and args of weight Observer.
        backend (str): backend implementation
    """
    assert dtype in (
        "qint4",
        "qint8",
        "qint16",
        "quint4",
    ), f"unsupported activation dtype: {dtype}"
    assert weight_dtype in (
        "qint4",
        "qint8",
        "qint16",
    ), f"unsupported weight dtype: {dtype}"
    if activation_qkwargs is not None:
        assert isinstance(activation_qkwargs, dict), (
            "activation qkwargs must be a dict, but get a "
            + type(activation_qkwargs).__name__
        )
    if weight_qkwargs is not None:
        assert isinstance(weight_qkwargs, dict), (
            "activation qkwargs must be a dict, but get a "
            + type(weight_qkwargs).__name__
        )
    return _get_custom_qconfig(
        dtype=dtype,
        weight_dtype=weight_dtype,
        activation_fake_quant=activation_fake_quant,
        weight_fake_quant=weight_fake_quant,
        activation_qkwargs=activation_qkwargs,
        weight_qkwargs=weight_qkwargs,
        backend=backend,
    )


def get_default_qat_out_qconfig(
    dtype="qint8",
    weight_fake_quant="fake_quant",
    weight_qkwargs=None,
    backend="",
):
    """default qat out qconfig


    Args:
        dtype (str): quantization type, the allowable value is qint8 and qint16
        weight_fake_quant (str): FakeQuantize type of weight, default is
                                 fake_quant.Avaliable items is fake_quant, lsq
                                 and pact
        weight_qkwargs(dict): A dict contain weight Observer type, args of
                              weight FakeQuantize and args of weight Observer.
        backend (str): backend implementation
    """
    assert dtype in (
        "qint4",
        "qint8",
        "qint16",
        "quint4",
    ), f"unsupported dtype: {dtype}"
    if weight_qkwargs is not None:
        assert isinstance(weight_qkwargs, dict), (
            "weight qkwargs must be a dict, but get a "
            + type(weight_qkwargs).__name__
        )
    return _get_custom_qconfig(
        dtype=dtype,
        activation_fake_quant=None,
        weight_fake_quant=weight_fake_quant,
        weight_qkwargs=weight_qkwargs,
        backend="",
    )


def get_default_calib_qconfig(dtype="qint8", calib_qkwargs=None, backend=""):
    """default calibration qconfig"""
    assert dtype in (
        "qint8",
        "qint16",
    ), f"unsupported dtype: {dtype}"
    if calib_qkwargs is not None:
        assert isinstance(calib_qkwargs, dict)
        calib_qconfig = QConfig(
            activation=CalibFakeQuantize.with_args(
                dtype=dtype,
                **calib_qkwargs,
            ),
            weight=default_weight_calib_fake_quant,
        )
    else:
        calib_qconfig = QConfig(
            activation=default_calib_fake_quant.with_args(
                dtype=dtype,
            ),
            weight=default_weight_calib_fake_quant,
        )
    return calib_qconfig


def is_calib_qconfig(qconfig):
    return isinstance(qconfig.activation(), CalibFakeQuantize)


def replace_qconfig_dtype(qconfig, activation_dtype, weight_dtype=None):
    dtype_bits = {
        qint8: 8,
        qint16: 16,
        qint4: 4,
        quint4: 4,
    }
    if is_calib_qconfig(qconfig):
        new_activation = partial(
            qconfig.activation,
            dtype=activation_dtype,
            num_bits=dtype_bits[activation_dtype],
            unsigned=True if activation_dtype == quint4 else False,
        )
        if weight_dtype is None or qconfig.weight is None:
            new_weight = qconfig.weight
        else:
            new_weight = partial(
                qconfig.weight,
                dtype=weight_dtype,
                num_bits=dtype_bits[weight_dtype],
                unsigned=True if activation_dtype == quint4 else False,
            )
    else:
        new_activation = partial(
            qconfig.activation,
            quant_min=qinfo(activation_dtype).min,
            quant_max=qinfo(activation_dtype).max,
            dtype=activation_dtype,
            qscheme=torch.per_tensor_affine
            if activation_dtype == quint4
            else torch.per_tensor_symmetric,
        )
        if weight_dtype is None or qconfig.weight is None:
            new_weight = qconfig.weight
        else:
            new_weight = partial(
                qconfig.weight,
                quant_min=qinfo(weight_dtype).min,
                quant_max=qinfo(weight_dtype).max,
                dtype=weight_dtype,
                qscheme=torch.per_channel_affine
                if weight_dtype == quint4
                else torch.per_channel_symmetric,
            )

    return QConfig(
        activation=new_activation,
        weight=new_weight,
    )
