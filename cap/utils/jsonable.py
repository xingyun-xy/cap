import json


def is_jsonable(x):
    """To show if a variable is jsonable.

    Args:
        x (any): a variable.

    Returns:
        bool: True means jsonable, False the opposite.
    """
    try:
        json.dumps(x)
        return True
    except Exception:
        return False
