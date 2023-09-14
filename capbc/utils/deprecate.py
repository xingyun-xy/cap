import logging
from datetime import datetime
import functools
from .color_message import (
    Color, color_message
)


__all__ = ['deprecated_warning', 'deprecated_waring',
           'log_deprecated', 'deprecated']


logger = logging.getLogger(__name__)


@functools.lru_cache(128)
def _warn_once(logger, msg: str):
    logger.warning(msg)


def deprecated_warning(msg):
    _warn_once(logger, '\033[31m[Deprecated] ' + msg + '\033[0m')


deprecated_waring = deprecated_warning


def log_deprecated(name='', text='', eos=''):

    assert name or text
    if eos:
        eos = f'after {eos}'
    if name:
        if eos:
            warn_msg = '%s will be deprecated %s. %s' % (name, eos, text)
        else:
            warn_msg = '%s was deprecated. %s' % (name, text)
    else:
        warn_msg = text
        if eos:
            warn_msg += ' Legacy period ends %s' % eos
    warn_msg = color_message(f'[Deprecated] {warn_msg}', front_color=Color.red)
    _warn_once(logger, warn_msg)


def deprecated(text='', eos=''):
    """
    A decorator for deprecated.
    """

    def deprecated_inner(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            name = '{}'.format(func.__name__)
            log_deprecated(name, text, eos)
            return func(*args, **kwargs)
        return new_func
    return deprecated_inner
