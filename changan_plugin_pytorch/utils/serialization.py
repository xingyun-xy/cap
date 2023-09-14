r"""
This file provide interface to save traced model with changan-plugin-pytorch
version by save_with_version and get changan-plugin-pytorch version from the
saved traced model by get_version_from_scriptmodule
Examples::

    >>> conv = torch.nn.Conv2d(2, 2, 2)
    >>> traced_conv = torch.jit.trace(conv, torch.randn(1, 2, 3, 3))
    >>> save_with_version(traced_conv, "conv.pt")
    >>> changan-plugin-version = get_version_from_scriptmodule("conv.pt")
    >>> print (changan-plugin-version)
"""
import torch
import warnings


__all__ = ["get_version_from_scriptmodule", "save_with_version"]


def get_version_from_scriptmodule(model):
    """
    get version from a script module

    Args:
        model(str): model file name
    """
    extra_file = {"changan-plugin-version": ""}
    torch.jit.load(model, _extra_files=extra_file)
    version = extra_file["changan-plugin-version"]
    if version.decode("utf-8") == "":
        warnings.warn(
            f"The model has not plugin version information, "
            f"please use changan_plugin_pytorch.utils.save_with_version "
            f"to save model first"
        )
        return None
    else:
        return version.decode("utf-8")


def _get_version():
    try:
        from changan_plugin_pytorch import __version__

        return __version__
    except ImportError:
        return None


def save_with_version(model, f, _extra_files=None):
    """
    Save an offline version of this module and add changan-plugin-pytorch
    version into the saved module for use in a separate process. The saved
    module serializes all of the methods, submodules, parameters, and
    attributes of this module. It can be loaded into the C++ API using
    torch::jit::load(filename) or into the Python API with torch.jit.load.
    And using changan_plugin_pytorch.utils.get_version.
    get_version_from_scriptmodule to get corresponding changan-plugin-pytorch
    version of the saved module.

    To be able to save a module, it must not make any calls to native Python
    functions. This means that all submodules must be subclasses of
    ScriptModule as well.

    Args:
        model: A :class:`ScriptModule` to save.
        f: A file-like object (has to implement write and flush)
            or a string containing a file name.
        _extra_files: Map from filename to contents which will be stored
            as part of `f`
    """
    version = _get_version()
    extra_file = {"changan-plugin-version": version}
    extra_file.update(_extra_files)
    torch.jit.save(model, f, _extra_files=extra_file)
