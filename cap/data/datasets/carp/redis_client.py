# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.


import logging
from typing import Any, Callable, Iterable, List, Optional

from redis import Redis

from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class TendisClient(Redis):
    """TendisClient.

    利用redis简单封装的 tendis 接口


    Args:
        rename_handle: 用于对key进行重命名的Callable, 对key进行简单的检查和更新。
        **kwargs: 穿透给 redis.Redis 的 kwargs
    """

    def __init__(
        self, rename_handle: Optional[Callable[[str], str]] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self._rename_handle = rename_handle

    def _rename_key(self, src_key: str) -> str:
        if self._rename_handle is not None:
            ret_key = self._rename_handle(src_key)
        logging.debug(f"{src_key} -> {ret_key}")
        return ret_key

    def hmget(self, name: str, keys: Iterable[str]) -> List[Optional[Any]]:
        name = self._rename_key(name)
        return super().hmget(name, keys)

    def hget(self, name: str, key: str) -> Optional[Any]:
        name = self._rename_key(name)
        return super().hget(name, key)

    def mget(self, keys: Iterable[str]) -> List[Optional[Any]]:
        renamed_keys = [self._rename_key(key) for key in keys]
        return super().mget(renamed_keys)

    def get(self, key: str) -> Optional[Any]:
        renamed_key = self._rename_key(key)
        return super().get(renamed_key)

    def set(self, key: str, content: Any) -> Optional[bool]:
        renamed_key = self._rename_key(key)
        return super().set(renamed_key, content)


def multimodal_default_rename_handle(key: str) -> str:
    """multimodal_default_rename_handle.

    multimodal- 默认的重命名接口。检查key是否有 multimodal- 开头,
    如果没有则添加 multimodal-

    Args:
        key: 要处理的键值字符串
    """
    key = key if key.startswith("multimodal-") else f"multimodal-{key}"
    return key
