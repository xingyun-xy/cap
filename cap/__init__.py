# Copyright (c) Changan Auto. All rights reserved.

from . import (
    callbacks,
    data,
    engine,
    metrics,
    models,
    optimizers,
    profiler,
    registry,
    utils,
    visualize,
)
from .version import check_deps, get_tmp_version

check_deps()

try:
    from .version import __version__
except Exception:
    __version__ = get_tmp_version()


def get_docker_url(python_version: str = "py3.8") -> str:
    base_url = ""
    import torch

    assert python_version in [
        "py3.6",
        "py3.8",
    ], f"python_version must be py3.6 or py3.8, but you set {python_version}"

    cuda_version = torch.__version__.split("+")[-1]
    torch_version = torch.__version__.split("+")[0]

    if "+" in __version__:
        image_name = "cap:runtime-%s-torch%s-%s" % (
            python_version,
            torch_version,
            cuda_version,
        )
        if __version__.endswith("unknown"):
            tag_name = __version__.split(".dev")[0]
        else:
            tag_name = __version__[-7:]
    else:
        image_name = "cap-release-%s-torch%s-%s" % (
            python_version,
            torch_version,
            cuda_version,
        )
        tag_name = __version__

    return base_url + image_name + "-" + tag_name
