from __future__ import absolute_import, print_function
import os
from .loader import ext_to_load_fn_map


__all__ = ['EasyDict', ]


class EasyDict(dict):
    """
    An wrapper of dict with easy attribute accessing and attribute
    protection.

    Examples
    --------
    >>> cfg = EasyDict(dict(a=1))
    >>> cfg.a
    1
    >>> cfg.b = 2
    >>> cfg.b
    2
    >>> cfg.set_immutable()
    >>> cfg.a = 1
    # cannot success, will raise an AttributeError

    Parameters
    ----------
    cfg_dict : dict
        The initial value.
    """

    _MUTABLE = '_MUTABLE'

    def __init__(self, cfg_dict=None):
        assert isinstance(cfg_dict, dict)
        new_dict = {}
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                new_dict[k] = EasyDict(v)
            elif isinstance(v, (list, tuple)):
                v = v.__class__([EasyDict(v_i) if isinstance(v_i, dict)
                                 else v_i for v_i in v])
                new_dict[k] = v
            else:
                new_dict[k] = v
        super(EasyDict, self).__init__(new_dict)
        self.__dict__[EasyDict._MUTABLE] = True

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        assert name not in self.__dict__, \
            "Invalid attempt to modify internal state: {}".format(name)
        if self.__dict__[EasyDict._MUTABLE]:
            if isinstance(value, dict):
                value = EasyDict(value)
            elif isinstance(value, (list, tuple)):
                value = value.__class__(
                    [EasyDict(v_i) if isinstance(v_i, dict)
                     else v_i for v_i in value])
            self[name] = value
        else:
            raise AttributeError(
                "Attempted to set {} to {}, but EasyDict is immutable".format(
                    name, value))

    @staticmethod
    def _recursive_visit(obj, fn):
        if isinstance(obj, EasyDict):
            fn(obj)
        if isinstance(obj, dict):
            for value_i in obj.values():
                EasyDict._recursive_visit(value_i, fn)
        elif isinstance(obj, (list, tuple)):
            for value_i in obj:
                EasyDict._recursive_visit(value_i, fn)

    def set_immutable(self):
        """
        Set attributes to be immutable.
        """

        def _fn(obj):
            obj.__dict__[EasyDict._MUTABLE] = False

        self._recursive_visit(self, _fn)

    def set_mutable(self):
        """
        Set attributes to be mutable.
        """

        def _fn(obj):
            obj.__dict__[EasyDict._MUTABLE] = True

        self._recursive_visit(self, _fn)

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, EasyDict) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    @staticmethod
    def _to_str(obj):
        if isinstance(obj, EasyDict):
            return obj.to_str()
        elif isinstance(obj, (list, tuple)):
            str_value = []
            for sub in obj:
                str_value.append(EasyDict._to_str(sub))
            return str_value
        elif not isinstance(obj, (int, float, bool, str)) and obj is not None:
            return obj.__str__()
        else:
            return obj

    def to_str(self):
        str_config = {}
        for k, v in self.items():
            str_config[k] = EasyDict._to_str(v)
        return str_config

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, super(EasyDict, self).__repr__())
