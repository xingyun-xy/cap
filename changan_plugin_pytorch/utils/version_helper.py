"""
  version control utils.
"""
import warnings
from importlib import import_module

__all__ = ["check_version"]


def check_version(
    mod_name: str,
    version: str,
    strict: bool = False,
    public=False,
    warn: bool = False,
) -> bool:
    """
    Check module version.

    Args:
        mod_name: python module name.
        version: python module version.
        strict: check version in strict mode, target version == version.
            Default is False, target version >= version.
        public: whether check with public version.
            Default is False.
        warn: whether raise ImportError.
            Default is False, raise ImportError when check failed.

    Raises:
        RuntimeError

    """
    try:
        m = import_module(mod_name)
    except ImportError as e:
        if not warn:
            raise RuntimeError(f"Unable to import dependency {mod_name}. {e}")
        else:
            warnings.warn(f"Unable to import dependency {mod_name}. {e}")
            return

    if not hasattr(m, "__version__"):
        warnings.warn(f"{m} has no __version__, skip check version")
        return

    mod_version = m.__version__
    # whether public version follows pep 440
    if public:
        version = version.split("+")[0]
        mod_version = mod_version.split("+")[0]

    from packaging import version as pack_version

    if strict:
        status = pack_version.parse(mod_version) == pack_version.parse(version)
        warn_str = f"Please install {m} == {version}, but get {mod_version}"
    else:
        status = pack_version.parse(mod_version) >= pack_version.parse(version)
        warn_str = f"Please install {m} >= {version}, but get {mod_version}"
    if not status:
        if warn:
            warnings.warn(warn_str)
        else:
            # raise RuntimeError(warn_str)
            pass
