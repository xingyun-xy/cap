# Copyright (c) 2022 by Contributors
# file: logger.py
# date: 2022-06-01
# brief: logging and string format related helper
# =============================================================================


def format_msg(msg, color):
    color_dict = {
        "black": 30,
        "b": 30,
        "red": 31,
        "r": 31,
        "green": 32,
        "g": 32,
    }
    if isinstance(color, str):
        color = color_dict.get(color, None)
        assert color, "Supported colors are: {}".format(color_dict.keys())
    return "\033[%dm%s\033[0m" % (color, msg)
