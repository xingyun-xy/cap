from . import env
from . import utils
from . import validators
from . import registry
from . import easydict
from . import patch

from . import resource_manager
from . import filestream
from . import logging

from . import distributed

from . import workflow

try:
    from . import sda
except ImportError:
    sda = None

try:
    from . import auto_fuel
except ImportError:
    auto_fuel = None

try:
    from . import auto_dp
except ImportError:
    auto_dp = None

try:
    from . import horizon_label_platform
except ImportError:
    horizon_label_platform = None

try:
    from . import atlassian
except ImportError:
    atlassian = None

try:
    from . import smb
except ImportError:
    smb = None

try:
    from . import adas_eval
except ImportError:
    adas_eval = None


__version__ = '0.8.0'

try:
    from .version import (
        __pypi_version__,
        __git_version__,
    )
except ImportError:
    __pypi_version__ = None
    __git_version__ = None
