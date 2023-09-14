"""
  load ops from extension c library.
"""


def _register_extensions():
    import importlib
    import os

    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_C")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)


try:
    _register_extensions()
except (ImportError, OSError):
    raise RuntimeError(
        "Cannot find the extension library(_C.so). "
        "Please rebuild the changan_plugin_pytorch."
    )


def _check_cuda_version():
    """
    Make sure that CUDA versions match between the pytorch install and
    changan_plugin_pytorch install
    """
    import torch

    # _version = torch.ops.changan._cuda_version()
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    _version = get_compiling_cuda_version()

    # if _version != -1 and torch.version.cuda is not None:
    #     horizon_version = str(_version)
    #     if int(horizon_version) < 10000:
    #         tv_major = int(horizon_version[0])
    #         tv_minor = int(horizon_version[2])
    #     else:
    #         tv_major = int(horizon_version[0:2])
    #         tv_minor = int(horizon_version[3])
    #     t_version = torch.version.cuda
    #     t_version = t_version.split(".")
    #     t_major = int(t_version[0])
    #     t_minor = int(t_version[1])
    #     if t_major != tv_major or t_minor != tv_minor:
    #         raise RuntimeError(
    #             "Detected that PyTorch and changan_plugin_pytorch were compiled with different CUDA versions. "  # noqa
    #             "PyTorch has CUDA Version={}.{} and changan_plugin_pytorch has CUDA Version={}.{}. "  # noqa
    #             "Please reinstall the changan_plugin_pytorch that matches your PyTorch install.".format(  # noqa
    #                 t_major, t_minor, tv_major, tv_minor
    #             )
    #         )
    pass
    return _version


_check_cuda_version()
