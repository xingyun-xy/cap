import warnings

from changan_plugin_pytorch.utils.mprop import mproperty


@mproperty
def QTensor(mod):
    warnings.warn(
        "changan_plugin_pytorch.quantization.qtensor.QTensor is deprecated, "
        + "please use changan_plugin_pytorch.qtensor.QTensor",
        DeprecationWarning,
        stacklevel=2,
    )
    from changan_plugin_pytorch.qtensor import QTensor

    return QTensor
