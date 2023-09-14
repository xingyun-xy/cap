from typing import Any
from dataclasses import dataclass


__all__ = ['Patcher', 'PatcherPause']


@dataclass
class _PatchMeta:
    obj: object
    attr: str
    has_source: bool
    source: Any
    target: Any


class Patcher(object):
    """
    A class that can patch object and restore at exit.

    .. note::

        Not thread safe!!
    """
    _current = None

    def __init__(self):
        self._old_scope = None
        self._patches = []

    def _patch_all(self):
        for meta_i in self._patches:
            setattr(meta_i.obj, meta_i.attr, meta_i.target)

    def _restore_all(self):
        for meta_i in self._patches[::-1]:
            if not meta_i.has_source:
                delattr(meta_i.obj, meta_i.attr)
            else:
                setattr(meta_i.obj, meta_i.attr, meta_i.source)

    def patch(self, obj: object, attr: str, target: Any):
        """
        Patch obj.attr to be target.

        Parameters
        ----------
        obj : object
            Object to be patched
        attr : str
            Patched attribute name
        target : Any
            Patched target
        """

        meta = _PatchMeta(
            obj=obj,
            attr=attr,
            has_source=hasattr(obj, attr),
            source=getattr(obj, attr, None),
            target=target)
        self._patches.append(meta)
        setattr(obj, attr, target)

    def __enter__(self):
        self._old_scope = Patcher._current
        Patcher._current = self
        return self

    def __exit__(self, ptype, value, trace):
        Patcher._current = self._old_scope
        self._old_scope = None
        self._restore_all()

    @classmethod
    def current(cls):
        return cls._current

    @classmethod
    def is_active(cls):
        return cls.current is not None


class PatcherPause(object):
    """
    Stop current patcher and restore all.
    """
    def __init__(self):
        self._old_patcher = None

    def __enter__(self):
        self._old_patcher = Patcher.current()
        if self._old_patcher is not None:
            self._old_patcher._restore_all()
        Patcher._current = None

    def __exit__(self, ptype, value, trace):
        if self._old_patcher is not None:
            self._old_patcher._patch_all()
        Patcher._current = self._old_patcher
        self._old_patcher = None
