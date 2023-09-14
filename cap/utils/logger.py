# Copyright (c) Changan Auto. All rights reserved.

import logging
import os
import sys

from .distributed import rank_zero_only

__all__ = [
    "init_logger",
    "DisableLogger",
    "MSGColor",
    "format_msg",
    "rank_zero_info",
    "rank_zero_warn",
]


def init_logger(
    log_file,
    logger_name=None,
    rank=0,
    level=logging.INFO,
    overwrite=False,
    stream=sys.stderr,
    clean_handlers=False,
):
    head = "%(asctime)-15s %(levelname)s Node[" + str(rank) + "] %(message)s"
    if rank != 0:
        log_file += "-rank%d" % rank
    if os.path.exists(log_file) and overwrite:
        os.remove(log_file)
    try:
        # may fail when multi processes do this concurrently
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    except FileExistsError:
        pass

    logger = logging.getLogger(logger_name)
    if clean_handlers:
        # duplicate handlers will cause duplicate outputs
        logger.handlers = []
    formatter = logging.Formatter(head)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.setLevel(level)


class DisableLogger(object):
    """Disable logger to logging anything under this scope.

    Args:
         enable: Whether enable `DisableLogger`.
            If True, `DisableLogger` works, will disable logging.
            If False, `DisableLogger` is no-op.
        level: used to disable less than or equal to the level.

    Examples::

        >>> import logging
        >>> logger = logging.getLogger()
        >>> with DisableLogger(enable=True):
        ...     logger.info('This info will not logging.')
        >>> logger.info('This info will logging after leaving the scope.')

        >>> with DisableLogger(enable=False):
        ...     logger.info(
        ...         'This info will logging as `DisableLogger` is not enable.')

    """

    def __init__(self, enable: bool = True, level: int = logging.WARNING):
        self.enable = enable
        self.level = level

    def __enter__(self):
        if self.enable:
            # disable level that less than or equal to self.level
            logging.disable(self.level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        if self.enable:
            logging.disable(logging.NOTSET)


class MSGColor(object):
    BLACK = 30  # default
    RED = 31  # Emergency warning
    GREEN = 32  # Harmless, just for notice


def format_msg(msg, color):
    return "\033[%dm%s\033[0m" % (color, msg)


def _info(*args, **kwargs):
    logger = logging.getLogger(__name__)
    logger.info(*args, **kwargs)


def _warn(*args, **kwargs):
    logger = logging.getLogger(__name__)
    logger.warn(*args, **kwargs)


rank_zero_info = rank_zero_only(_info)

rank_zero_warn = rank_zero_only(_warn)
