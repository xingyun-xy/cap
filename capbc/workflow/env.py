import dill
import cloudpickle


dill.extend(False)


__all__ = ['set_serializer', 'get_serializer']


_serializer = 'dill'


def set_serializer(serializer):
    """Set serializer."""
    global _serializer
    assert serializer in ['dill', 'cloudpickle', dill, cloudpickle], \
        'Currently, only support dill and cloudpickle'
    _serializer = serializer


def get_serializer(serializer=None, return_name=False):
    """Get serializer.

    Parameters
    ----------
    serializer: str or Module, optional
        serializer, if None, using preset serializer, by default None
    return_name : bool, optional
        If true, return the serializer module name, otherwise return the
        serializer module, by default False
    """
    if serializer is None:
        global _serializer
        serializer = _serializer

    obj2serializer = {
        'dill': dill,
        'cloudpickle': cloudpickle,
        dill: dill,
        cloudpickle: cloudpickle
    }

    serializer = obj2serializer[serializer]

    if return_name:
        return serializer.__name__
    else:
        return serializer
