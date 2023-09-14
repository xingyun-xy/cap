"""import utility functions."""
import os
import errno
from importlib import import_module
from typing import Any, Callable, Tuple

from packaging.version import parse as parse_version


__all__ = [
    'try_import',
    'import_try_install',
    'check_import',
    'optional_import',
]


def min_version(pkg, version: str) -> bool:
    """Check if package meet the minimum version requirement.

    Args:
        pkg: Python package.
        version (str): Version requirement.

    Returns:
        bool: If the requirement is met.
    """
    ver1 = parse_version(pkg.__version__)
    ver2 = parse_version(version)
    return ver1 >= ver2


def try_import(package, message=None):
    """Try import specified package, with custom message support.

    Parameters
    ----------
    package : str
        The name of the targeting package.
    message : str, default is None
        If not None, this function will raise customized error message when
        import error is found.


    Returns
    -------
    module if found, raise ImportError otherwise

    """
    try:
        return __import__(package)
    except ImportError as e:
        if not message:
            raise e
        raise ImportError(message)


def import_try_install(package, extern_url=None):
    """Try import the specified package.
    If the package not installed, try use pip to install and import if success.

    Parameters
    ----------
    package : str
        The name of the package trying to import.
    extern_url : str or None, optional
        The external url if package is not hosted on PyPI.
        For example, you can install a package using:
        "pip install git+http://github.com/user/repo/tarball/master/egginfo=xxx".
        In this case, you can pass the url to the extern_url.

    Returns
    -------
    <class 'Module'>
        The imported python module.

    """  # noqa
    try:
        return __import__(package)
    except ImportError:
        try:
            from pip import main as pipmain
        except ImportError:
            from pip._internal import main as pipmain

        # trying to install package
        url = package if extern_url is None else extern_url
        # will raise SystemExit Error if fails
        pipmain(['install', '--user', url])

        # trying to load again
        try:
            return __import__(package)
        except ImportError:
            import sys
            import site
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
            return __import__(package)
    return __import__(package)


def check_import(package, package_name, err_msg=None):
    if package is None:
        if err_msg is None:
            raise ModuleNotFoundError(f'Cannot import module {package_name}')
        else:
            raise ModuleNotFoundError(f'Cannot import module {package_name}, err_msg = {err_msg}')  # noqa


def optional_import(
    package: str,
    version: str = "",
    version_checker: Callable[..., bool] = min_version,
    name: str = "",
    *version_args,
) -> Tuple[Any, bool]:
    """Optional import of a module.

    Args:
        package (str):Name of the package to be imported.
        version (str, optional): Version string used by the version_checker. \
            Defaults to "".
        version_checker (Callable[..., bool], optional): A callable to check \
            the module version. Defaults to :func:`min_version`.
        name (str, optional): Name of import attribute. Defaults to "".
        version_args (optional): Additional arguments to version_checker. \
            Defaults to None.

    Raises:
        Exception: if import failed

    Returns:
        Tuple[Any, bool]: The imported module and a boolean indicating if \
            the module is imported.

    Examples:
        >>> torch, flag = optional_import('torch', '1.1')
        >>> print(flag)
        True
        >>> the_module, flag = optional_import('unknown_module')
        >>> print(flag)
        False
    """
    if name:
        exception_msg = f"from {package} import {name}"
    else:
        exception_msg = f"import {package}"

    try:
        top_pkg = import_module(package)
        pkg = getattr(top_pkg, name) if name else top_pkg
    except Exception as import_exception:
        tb = import_exception.__traceback__
        exception_msg += f" {import_exception}"
    else:
        if version_args and version_checker(top_pkg, version, *version_args):
            return pkg, True
        elif not version_args and version_checker(top_pkg, version):
            return pkg, True
        exception_msg += f"requeirs {version} by {version_checker.__name__}"

    class _LazyRaise:
        def __init__(self, *args, **kwargs):
            msg = exception_msg
            if tb is None:
                self._exception = ImportError(msg)
            else:
                self._exception = ImportError(msg).with_traceback(tb)

        def __getattr__(self, name):
            raise self._exception

        def __call__(self, *args, **kwargs):
            raise self._exception

    return _LazyRaise(), False
