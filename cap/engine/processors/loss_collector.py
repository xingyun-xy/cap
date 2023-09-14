# Copyright (c) Changan Auto. All rights reserved.

import re
from typing import Any, Callable, Sequence, Union

import torch

from cap.utils.apply_func import (
    _as_list,
    flatten,
    is_list_of_type,
    to_flat_ordered_dict,
)

__all__ = ["collect_loss_by_index", "collect_loss_by_regex"]


def collect_loss_by_index(
    indexes: Union[int, Sequence[int]]
) -> Callable:  # noqa: D205,D400
    """
    Collect loss by specific indexes of loss Tensors in model outputs
    like: `(losses, preds)`, `(...loss1, ...loss2, ...)` and so on.

    Args:
        indexes: Indexes of loss Tensors in model outputs.

    Returns:
        A function with model outputs as input, return loss Tensors collected
        by indexes.

    Examples::

        >>> model_outs = [
        ...     [torch.tensor(1.0), torch.tensor(2.0)],  # losses
        ...     [torch.tensor(3.0), torch.tensor(4.0)]   # preds
        ... ]
        >>> collector = collect_loss_by_index(0)
        >>> collector(model_outs)
        [tensor(1.), tensor(2.)]

    """
    indexes = _as_list(indexes)
    assert len(indexes) > 0, indexes
    assert is_list_of_type(
        indexes, element_type=int
    ), "`indexes` should be a list/tuple of int"

    def _collect(model_outs: Sequence) -> Sequence[torch.Tensor]:
        model_outs = _as_list(model_outs)
        losses = []
        for i in indexes:
            assert i < len(model_outs), "%d vs. %d" % (i, len(model_outs))
            flats = flatten(model_outs[i])[0]
            for obj in flats:
                assert isinstance(
                    obj, torch.Tensor
                ), "Expect `torch.Tensor` get %s, please check index: %d" % (
                    type(obj),
                    i,
                )

            losses.extend(flats)

        return losses

    return _collect


def collect_loss_by_regex(
    loss_name_pattern: str,
) -> Callable:  # noqa: D205,D400
    """
    Flatten model outputs into an OrderedDict, then using `re` regex to match
    the keys of loss Tensors.

    Args:
        loss_name_pattern: `re` regex, e.g. '^.*loss.*' .

    Returns:
        A function with model outputs as input, return loss Tensors matched by
        `loss_name_pattern`.

    Example::

        >>> model_outs = dict(
        ...     toy_loss_1=torch.tensor(1.0),
        ...     toy_predict=torch.tensor(2.0),
        ...     toy_loss_2=torch.tensor(3.0),
        ... )
        >>> collector = collect_loss_by_regex('^.*loss.*')
        >>> collector(model_outs)
        [tensor(1.), tensor(3.)]

    """
    loss_name_pattern = re.compile(loss_name_pattern, re.IGNORECASE)

    def _collect(model_outs: Any) -> Sequence[torch.Tensor]:  # noqa: D205,D400
        """
        Filter out loss Tensors in model outputs `model_outs` using
        'loss_name_pattern`.

        Returns:
            loss Tensors.
        """
        flat_outs = to_flat_ordered_dict(model_outs, key_prefix="")

        losses = []
        for k, v in flat_outs.items():
            if loss_name_pattern.match(k):
                if v is not None:
                    assert isinstance(v, torch.Tensor), (
                        "Expect `torch.Tensor` but get %s, please check "
                        "loss_name_pattern: %s , output key: %s ."
                        % (type(v), loss_name_pattern.pattern, k)
                    )
                    losses.append(v)

        assert len(losses) > 0, (
            "No loss matched, please check loss_name_pattern: %s, model "
            "outputs: %s." % (loss_name_pattern.pattern, flat_outs.keys())
        )

        return losses

    return _collect
