# Copyright (c) Changan Auto. All rights reserved.

import logging
import os
import subprocess
from datetime import datetime
from distutils.version import LooseVersion
from importlib import import_module
from types import ModuleType
from typing import Dict, List

logger = logging.getLogger(__name__)

# current version
__version__ = "1.1.1"

# torch、torchvision and changan_pytorch_plugin's version
DEPS = [
    [
        {"torch": "1.9.1+cu102", "strict": True},
        {"torchvision": "0.10.1+cu102", "strict": True},
        {"torchaudio": "0.9.1", "strict": True},
        {"changan_plugin_pytorch": "0.14.0", "strict": False},
    ],
    [
        {"torch": "1.9.1+cu111", "strict": True},
        {"torchvision": "0.10.1+cu111", "strict": True},
        {"torchaudio": "0.9.1", "strict": True},
        {"changan_plugin_pytorch": "0.14.0", "strict": False},
    ],
    [
        {"torch": "1.10.2+cu102", "strict": True},
        {"torchvision": "0.11.3+cu102", "strict": True},
        {"torchaudio": "0.10.2+cu102", "strict": True},
        {"changan_plugin_pytorch": "0.15.0", "strict": False},
    ],
    [
        {"torch": "1.10.2+cu111", "strict": True},
        {"torchvision": "0.11.3+cu111", "strict": True},
        {"torchaudio": "0.10.2+cu111", "strict": True},
        {"changan_plugin_pytorch": "0.15.0", "strict": False},
    ],
    [
        {"torch": "1.9.1+cu111", "strict": True},
        {"torchvision": "0.10.1+cu111", "strict": True},
        {"torchaudio": "0.9.0", "strict": True},
        {"changan_plugin_pytorch": "0.14.0", "strict": False},
    ],
]


def check_version(
    module: ModuleType, version: str, strict: bool = False
) -> bool:
    """
    Check software version.

    Args:
        module: python module, with `__version__` attribute.
        version: python module version.
        strict: check version in strict mode, target version == version.
            Default is False, target version >= version.

    Raises:
        ImportError

    """
    if not hasattr(module, "__version__"):
        logger.warning(f"{module} has no __version__, skip check version")
        return True

    module_version = module.__version__  # type: ignore
    # TODO(0.1): The release version of plugin starts with 'v', like v0.12.2   # noqa
    if module_version[0].lower() == "v":
        module_version = module_version[1:]
    if strict:
        # fmt: off
        status = LooseVersion(module_version) == LooseVersion(
            version
        )
        # fmt: on
    else:
        # fmt: off
        status = LooseVersion(module_version) >= LooseVersion(
            version
        )
        # fmt: on
    return status


def get_setup_version(version: str = __version__) -> str:
    # for setup in pre release version
    release_version = os.getenv("RELEASE_VERSION")

    if release_version is None:
        cwd = os.path.dirname(os.path.abspath("__file__"))

        # check if in a git repo and if there are uncommitted changes
        try:
            _ = (
                subprocess.check_output(["git", "status", "-s"], cwd=cwd)
                .decode("ascii")
                .strip()
            )
            is_git_repo = True
        except Exception:
            logger.warning("Not in a git repo, get temporary version number")
            is_git_repo = False

        # get last commit time
        if is_git_repo:
            commit_time_unix = int(
                subprocess.check_output(
                    ["git", "log", "-1", "--pretty=format:%ct"], cwd=cwd
                )
                .decode("ascii")
                .strip()
            )
            commit_time = datetime.fromtimestamp(commit_time_unix)

            commit_id = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode("ascii")
                .strip()
            )[:7]
        else:
            commit_time = datetime.now()

            commit_id = "unknown"
        commit_timestamp = commit_time.strftime("%Y%m%d%H%M")

        version += ".dev{}+{}".format(commit_timestamp, commit_id)
    return version


def get_tmp_version(version: str = __version__) -> str:
    # if CAP has not been installed, get a temporary version number
    # with format __version__.dev{timestamp}+unknown
    version += ".dev{}+{}".format(
        datetime.now().strftime("%Y%m%d%H%M"), "unknown"
    )
    return version


def write_version_file(version: str) -> None:
    version_path = os.path.join(os.path.dirname(__file__), "_version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))


def check_deps(deps: List[List[Dict]] = DEPS) -> None:
    status = False
    for dep in deps:
        each_status = True
        for each_lib in dep:
            keys = list(each_lib.keys())
            assert len(keys) == 2
            try:
                m = import_module(keys[0])
            except ImportError as e:
                raise ImportError(
                    "Unable to import dependency {}. {}".format(keys[0], e)
                )
            else:
                result = check_version(m, each_lib[keys[0]], each_lib[keys[1]])
                each_status = each_status and result
            if not each_status:
                break
        status = status or each_status
        if status:
            break
    if not status:
        raise ImportError(
            f"You should check the version of torch、torchvision、"
            f"changan_pytorch_plugin. Make sure they meet any of the"
            f" conditions in {DEPS}"
        )
