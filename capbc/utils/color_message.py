from enum import Enum, unique
from functools import reduce
from .utils import _check_type


__all__ = ['ShowMethod', 'Color', 'color_message']


@unique
class ShowMethod(Enum):
    default = '0'
    highlight = '1'
    underscore = '4'

    @classmethod
    def get(cls, key):
        if isinstance(key, str):
            return cls[key]
        else:
            return cls(key)


@unique
class Color(Enum):
    black = ('30', '40')
    red = ('31', '41')
    green = ('32', '42')
    yellow = ('33', '43')
    blue = ('34', '44')
    white = ('37', '47')

    @classmethod
    def get(cls, key):
        if isinstance(key, str):
            return cls[key]
        else:
            return cls(key)


def color_message(message, show_method=None,
                  front_color=None, background_color=None):
    """
    Coloring a message.

    Parameters
    ----------
    message : str
        Message
    show_method : :py:class:`ShowMethod`, optional
        Show method, by default None
    front_color : :py:class:`Color`, optional
        Front color, by default None
    background_color : :py:class:`Color`, optional
        Background color, by default None
    """
    color_type = []
    if show_method is not None:
        show_method = ShowMethod.get(show_method)
        _check_type(show_method, ShowMethod, 'show_method')
        color_type.append(show_method.value)
    if front_color is not None:
        front_color = Color.get(front_color)
        _check_type(front_color, Color, 'front_color')
        color_type.append(front_color.value[0])
    if background_color is not None:
        background_color = Color.get(background_color)
        _check_type(background_color, Color, 'background_color')
        color_type.append(background_color.value[1])
    if not color_type:
        color_type = '0'
    else:
        color_type = reduce(lambda x, y: x + ';' + y, color_type)
    assert isinstance(message, str)

    message = f'\033[{color_type}m{message}\033[0m'
    return message
