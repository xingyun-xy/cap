import warnings

from changan_plugin_pytorch.utils.mprop import mproperty


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
