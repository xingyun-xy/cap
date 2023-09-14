# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.


from typing import Any, Iterable, Sequence, Tuple, Union

from cap.utils.apply_func import _as_list, is_list_of_type


def collect_data_by_index(
    data, indexes: Union[int, Sequence[int]]
) -> Iterable[Tuple[Any, ...]]:
    """按照指定的顺序整理数据.

    Args:
        indexes: 要保留的数据在输入数据中的顺序
    """
    indexes = _as_list(indexes)
    assert len(indexes) > 0, indexes
    assert is_list_of_type(
        indexes, element_type=int
    ), "`indexes` should be a list/tuple of int"

    for sample in data:
        sample = _as_list(sample)
        assert len(sample) > max(
            indexes
        ), f"长度为{len(sample)}, 但是要求索引列表为 {indexes}"
        sample = [sample[idx] for idx in indexes]
        yield tuple(sample)
