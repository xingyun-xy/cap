# Copyright (c) Changan Auto. All rights reserved.
from collections.abc import Iterable
from typing import Any, Generator, Tuple

__all__ = ["prefetch_iterator"]


def prefetch_iterator(
    iterable: Iterable,
) -> Generator[Tuple[Any, bool], None, None]:
    """
    Return an iterator that pre-fetches and caches the next item.

    The values are passed through from the given iterable with an
    added boolean indicating if this is the last item.
    See `https://stackoverflow.com/a/1630350 <https://stackoverflow.com/a/1630350>`_
    """  # noqa: E501
    it = iter(iterable)

    try:
        # the iterator may be empty from the beginning
        last = next(it)
    except StopIteration:
        return

    for val in it:
        # yield last and has next
        yield last, False
        last = val
    # yield last, no longer has next
    yield last, True
