# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.


from bisect import bisect_left, bisect_right
from typing import Optional, Sequence, Tuple, Union


def find_closest_element(
    x: Union[int, float],
    x_list: Sequence[Union[int, float]],
    out_of_bound_threshold: Optional[Union[int, float]] = None,
    closest_left: bool = False,
    closest_right: bool = False,
) -> Tuple[Union[int, float], int]:
    """寻找最近邻元素.

    从 x_list (已经排序好的)中找出跟 x 最接近的元素;
    可以设置寻找左侧最近，也可以设置选择最右侧的元素最近。
    不设置的情况下，选择最接近的。

    Args:
        x : 需要寻找的目标元素
        x_list : 已经排序好的被查询序列
        out_of_bound_threshold : 当元素搜索到 x_list 第零个或最后一个时，
            检查第零个或最后一个元素的距离，当大于设置阈值时返回 none
        closest_left : 如果设置为 True，则检查左侧最接近的元素
        closest_right : 如果设置为 True, 则检查右侧最接近的元素

    Returns:
        (被选出的元素，元素的索引) 或者 (None, None)
    """
    # x_list = sorted(x_list)
    assert type(x) == int or type(x) == float, "x must be int or float"
    assert not closest_left or not closest_right, "can not both set true."
    # not set out_of_bound_threshold, if out of boundary, return the boundary
    if out_of_bound_threshold is None:
        if x >= x_list[-1]:
            return x_list[-1], len(x_list) - 1
        if x <= x_list[0]:
            return x_list[0], 0
    # if set out_of_bound_threshold, only in (left-threshold, right+threshold)
    # return value
    else:
        if x > x_list[0] and x < x_list[-1]:
            pass
        elif x >= x_list[-1] and x - x_list[-1] < out_of_bound_threshold:
            return x_list[-1], len(x_list) - 1
        elif x <= x_list[0] and x_list[0] - x < out_of_bound_threshold:
            return x_list[0], 0
        else:
            return None, None
    # closest_left: the left largest or equal element
    if closest_left:
        idx = bisect_right(x_list, x)
        if idx > 0:
            return x_list[idx - 1], idx - 1
        raise ValueError
    # closest_right: the right smallest or equal element.
    if closest_right:
        idx = bisect_left(x_list, x)
        if idx != len(x_list):
            return x_list[idx], idx
        raise ValueError
    # normal the cloest one
    idx = bisect_left(x_list, x)
    if idx > 0:
        left = x_list[idx - 1]
        right = x_list[idx]
        if right - x < x - left:
            return right, idx
        else:
            return left, idx - 1
    raise ValueError
