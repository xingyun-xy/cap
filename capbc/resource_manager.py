import copy
from typing import Dict, Optional, Hashable, Any
from .utils import _as_list


__all__ = ['ContextResourceManager', 'get_current_resource_manager',
           "is_active", 'get_resource']


class ContextResourceManager(object):
    """
    Global context resource manager, using this class to register
    all resources.

    Parameters
    ----------
    resources : Dict[Hashable, Any]
        A dict mapping resource key to resource object.
    """
    _current = None

    def __init__(self, resources: Optional[Dict[Hashable, Any]] = None):
        self._old_scope = None
        if resources is None:
            resources = dict()
        self._resources = resources

    def add_resource(self, key, resource, overwrite=False):
        if key in self._resources:
            assert overwrite, f'{key} already exists, please set overwrite=True if you want to overwrite it'  # noqa
        self._resources[key] = resource

    @property
    def resources(self):
        return self._resources

    @resources.setter
    def resources(self, value: Optional[Dict] = None):
        if value is None:
            value = dict()
        assert isinstance(value, dict)
        self._resources = value

    def __enter__(self):
        self._old_scope = ContextResourceManager._current
        if self._old_scope is not None:
            resources = copy.copy(self._old_scope.resources)
            resources.update(self.resources)
            self.resources = resources
        ContextResourceManager._current = self
        return self

    def __exit__(self, ptype, value, trace):
        ContextResourceManager._current = self._old_scope

    def get(self, key):
        assert key in self.resources, f'Missing resource {key}'
        return self.resources[key]

    def __contains__(self, item):
        return item in self._resources


def get_current_resource_manager():
    return ContextResourceManager._current


def is_active():
    return get_current_resource_manager() is not None


def get_resource(key: Hashable):
    """
    Get resource object from :py:class:`ContextResourceManager`
    with giving resource key.

    Parameters
    ----------
    key : Hashable
        Key of resource object.
    """
    manager = get_current_resource_manager()
    assert manager is not None, \
        f'Please running the code within the `ContextResourceManager` scope'
    return manager.get(key)
