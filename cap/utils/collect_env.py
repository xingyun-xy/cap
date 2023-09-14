import os
import sys
from collections import defaultdict
from importlib import import_module

import numpy as np
import torch
import torchvision
from tabulate import tabulate


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def collect_requirements_env_info(data):
    requirements = [
        # internal
        "changan_plugin_pytorch",
        "hbdk",
        "capbc",
        "sdasdk",
    ]
    for name in requirements:
        try:
            m = import_module(name)
        except ImportError as e:
            raise ImportError(
                "Unable to import dependency {}. {}".format(name, e)
            )
        else:
            try:
                version_info = str(m.__pypi_version__)
            except AttributeError:
                version_info = str(m.__version__)

            data.append(
                (
                    name,
                    version_info + " @" + os.path.dirname(m.__file__),
                )
            )


def collect_env_info():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.
            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - Numpy: Numpy version.
            - PyTorch: PyTorch version.
            - Torchvision: Torchvision version.
            - changan_plugin_pytorch: changan_plugin_pytorch verison.
            - CAP: CAP version.
            - GPU available: Bool, indicating if GPU is available.
            - GPU devices: Device type of each GPU.
            - Driver: Driver version of Nvidia GPU device.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - CAP requirements info: Require libraries version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
    """

    has_gpu = torch.cuda.is_available()
    torch_version = torch.__version__

    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))
    data.append(
        (
            "PyTorch",
            str(torch_version) + " @" + os.path.dirname(torch.__file__),
        )
    )
    data.append(("PyTorch debug build", torch.version.debug))
    data.append(
        (
            "Torchvision",
            str(torchvision.__version__)
            + " @"
            + os.path.dirname(torchvision.__file__),
        )
    )

    try:
        import cap

        data.append(
            (
                "CAP",
                str(cap.__version__) + " @" + os.path.dirname(cap.__file__),
            )
        )
    except ImportError:
        data.append(("CAP", "failed to import"))
    except AttributeError:
        data.append(("CAP", "imported a wrong installation"))

    if has_gpu is False:
        has_gpu_text = "No: torch.cuda.is_available() == False"
    else:
        has_gpu_text = "Yes"
    data.append(("GPU available", has_gpu_text))

    from torch.utils.cpp_extension import CUDA_HOME

    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = ".".join(
                (str(x) for x in torch.cuda.get_device_capability(k))
            )
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        try:
            from torch.utils.collect_env import get_nvidia_driver_version
            from torch.utils.collect_env import run as _run

            data.append(("Driver version", get_nvidia_driver_version(_run)))
        except Exception:
            pass
        msg = (
            " - invalid!"
            if not (CUDA_HOME and os.path.isdir(CUDA_HOME))
            else ""
        )
        data.append(("CUDA_HOME", str(CUDA_HOME) + msg))

        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if cuda_arch_list:
            data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))

    collect_requirements_env_info(data)
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":
    env_info = collect_env_info()
    print(env_info)
