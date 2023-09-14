import changan_plugin_pytorch as changan

__all__ = [
    "set_default_qconfig",
    "get_default_qat_qconfig",
    "get_default_qat_out_qconfig",
]

global_qat_qconfig = changan.quantization.get_default_qat_qconfig()
global_qat_out_qconfig = changan.quantization.get_default_qat_out_qconfig()

try:
    global_calibration_qconfig = (
        changan.quantization.get_default_calib_qconfig()
    )  # noqa E501
except Exception:
    global_calibration_qconfig = None


def set_default_qconfig(
    dtype: str = "qint8",
    activation_fake_quant: str = "fake_quant",
    weight_fake_quant: str = "fake_quant",
    activation_qkwargs: dict = None,
    weight_qkwargs: dict = None,
):
    """Set default qat qconfig.

    Args:
        dtype (str): Quantization type, allowable values are "qint4", "qint8",
                     "qint16", "quint4".
                     For Calibration, only support "qint8" and "qint16"

        activation_fake_quant (str): FakeQuantize type of activation, default
                                     is "fake_quant". Avaliable items is
                                     "fake_quant", "lsq", "pact".

        weight_fake_quant (str): FakeQuantize type of weight, default is
                                 fake_quant.Avaliable items is "fake_quant",
                                 "lsq" and "pact".

        activation_qkwargs (dict): A dict contain activation Observer type,
                                  args of activation FakeQuantize and args
                                  of activation Observer.

        weight_qkwargs (dict): A dict contain weight Observer type, args of
                              weight FakeQuantize and args of weight Observer.

    """

    qconfig_params = {
        "dtype": dtype,
        "activation_fake_quant": activation_fake_quant,
        "weight_fake_quant": weight_fake_quant,
        "activation_qkwargs": activation_qkwargs,
        "weight_qkwargs": weight_qkwargs,
    }

    global global_qat_qconfig
    global global_qat_out_qconfig
    global global_calibration_qconfig

    # For horizon_plugin_version >= 0.12.2
    global_qat_qconfig = changan.quantization.get_default_qat_qconfig(
        **qconfig_params
    )
    global_qat_out_qconfig = changan.quantization.get_default_qat_out_qconfig(
        dtype=dtype,
        weight_fake_quant=weight_fake_quant,
        weight_qkwargs=weight_qkwargs,
    )

    if dtype in ["qint8", "qint16"]:
        global_calibration_qconfig = (
            changan.quantization.get_default_calib_qconfig(dtype=dtype)
        )


def get_default_qat_qconfig():
    return global_qat_qconfig


def get_default_qat_out_qconfig():
    return global_qat_out_qconfig


def get_default_calibration_qconfig():
    return global_calibration_qconfig
