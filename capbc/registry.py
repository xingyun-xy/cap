import warnings


__all_ = ["Registry", "build_registry"]


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)

    To register an object with alias :

    .. code-block:: python

        @BACKBONE_REGISTRY.register
        @BACKBONE_REGISTRY.alias('custom')
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone, 'custom')
    """

    def __init__(self, name, lower_str=True):
        self._name = name
        self._name_obj_map = {}
        self._lower_str = lower_str

    def pop(self, name, default=None):
        return self._name_obj_map.pop(name, default)

    def delete(self, name):
        self.pop(name, None)

    def _get_name(self, name):
        if isinstance(name, str) and self._lower_str:
            name = name.lower()
        return name

    def __contains__(self, name):
        name = self._get_name(name)
        return name in self._name_obj_map

    def keys(self):
        return self._name_obj_map.keys()

    def _do_register(self, name, obj):
        name = self._get_name(name)
        assert (name not in self._name_obj_map), \
            "An object named '{}' was already registered in '{}' registry!".format(  # noqa
            name, self._name)
        self._name_obj_map[name] = obj

    def register(self, obj=None, *, name=None):
        """
        Register the given object under the the name `obj.__name__`
        or given name.
        """
        if obj is None and name is None:
            raise ValueError('Should provide at least one of obj and name')
        if obj is not None and name is not None:
            self._do_register(name, obj)
        elif obj is not None and name is None:  # used as decorator
            name = obj.__name__
            self._do_register(name, obj)
            return obj
        else:
            return self.alias(name)

    def alias(self, name):
        """Get registrator function that allow aliases.

        Parameters
        ----------
        name: str
            The register name

        Returns
        -------
        a registrator function
        """

        def reg(obj):

            self._do_register(name, obj)
            return obj

        return reg

    def get(self, name):
        origin_name = name
        name = self._get_name(name)
        ret = self._name_obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    origin_name, self._name
                )
            )
        return ret


def build_registry(registry, key, *args, **kwargs):
    """
    Build registry by name.

    Parameters
    ----------
    registry : :py:class:`Registry`
    key : Hashable
        Registry key
    args : tuple
        Positional arguments
    kwargs : dict
        Key value arguments

    Returns
    -------
    object: object
        The builded object.
    """
    if key in registry:
        build_fn = registry.get(key)
    elif callable(key):
        build_fn = key
    else:
        raise ValueError(f'key must be registed in registry {registry} or callable, but get key={key}, type={str(type(key))}')  # noqa
    try:
        return build_fn(*args, **kwargs)
    except BaseException as e:
        msg = f'Meet exception when building {registry._name}[{key}]'
        warnings.warn(msg)
        raise e
