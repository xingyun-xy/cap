# Copyright (c) Changan Auto. All rights reserved.
import logging
from collections import defaultdict
from typing import Any

__all__ = ["EventStorage"]

logger = logging.getLogger(__name__)

_CURRENT_STORAGE_STACK = []


class EventStorage:
    """
    The user-facing class that provides metrics/images storage functionalities.

    This class is adjusted from EventStorage in detectron2.
    """

    def __init__(self):
        self._history = defaultdict(list)

    def put(self, key: str, value: Any, always_dict: bool = False):
        """
        Put the specified value into storage.

        Args:
            key: Key of specified value.
            value: The specified value.
            alwarys_dict: Whether all values are dict. If True, the structure
                of history will be dict, which is easy to find.
        """

        if key not in self._history:
            self._history[key] = []
            if always_dict:
                self._history[key].append({})

        if always_dict:
            assert isinstance(value, dict)
            self._history[key][0].update(value)
        else:
            self._history[key].append(value)

    def clear(self):
        self._history = defaultdict(list)

    def clear_key(self, key: str):
        self._history[key] = []

    def get(self, key: str):
        ret = self._history.get(key, None)
        if ret is None:
            raise KeyError("{} is not available in storage!".format(key))
        return ret

    @property
    def histories(self):
        return self._history

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()
