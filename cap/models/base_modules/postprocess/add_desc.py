# Copyright (c) Changan Auto. All rights reserved.

import logging
from collections import OrderedDict
from typing import Dict, Sequence, Union

import changan_plugin_pytorch as changan
import torch

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, flatten, is_list_of_type, regroup
from cap.utils.logger import MSGColor, format_msg
from .postprocess import PostProcessorBase

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class AddDesc(PostProcessorBase):
    """Add description str for each output tensor.

    Three steps to add description:
    (1) flatten `pred` into a tuple of tensor (flat_tensors).
    (2) add description to output tensors.
        In strict mode len(flat_tensors) == len(per_tensor_desc), meaning
        each desc corresponds to an output tensor.
        In non-strict mode, when there are more descs than output tensors,
        rest descs are ignored, and when there are more outputs, rest tensors
        stay unannotated.
    (3) regroup flat_tensors to recover origin format.

    You can access desc str by `Tensor.annotation`, or get all output tensor's
    desc str of a ScriptModule using
    `changan_plugin_pytorch.functional.get_output_annotation(script_model)`.

    .. note::

        AddDesc should be the last module (leaf node) of your model, otherwise
        `get_output_annotation` will fail.

    Args:
        per_tensor_desc: Description str for each prediction tensor.
            You have to known the num of output tensors and provide
            corresponding num of desc str.
        strict: True means this op is in strict mode, and otherwise non-strict.
    """

    def __init__(
        self, per_tensor_desc: Union[str, Sequence[str]], strict: bool = True
    ):
        super(AddDesc, self).__init__()
        per_tensor_desc = _as_list(per_tensor_desc)
        assert is_list_of_type(per_tensor_desc, str), (
            f"per_tensor_desc should be a list of str, "
            f"but get {per_tensor_desc}"
        )
        self.per_tensor_desc = per_tensor_desc
        self.strict = strict

    def forward(
        self,
        pred: Union[
            torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]
        ],
        *args,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]]:
        """Add desc str for each tensor in `pred`.

        Args:
            pred: Prediction tensors, should be in order of `per_tensor_desc`.
        """

        def _check_type(obj):
            if isinstance(obj, torch.Tensor):
                pass
            elif isinstance(obj, (list, tuple)):
                for i in obj:
                    _check_type(i)
            elif isinstance(obj, dict):
                if not isinstance(obj, OrderedDict):
                    logger.warning(
                        format_msg(
                            "`pred` contains dict object, dict can not insure "
                            "order when doing `flatten(pred)`, we recommend "
                            "you to convert dict to OrderedDict",
                            MSGColor.RED,
                        )
                    )
                else:
                    for _, v in obj.items():
                        _check_type(v)
            else:
                raise TypeError(f"unsupported type: {type(obj)} {obj}")

        # we are going to flatten `pred` into a tuple of tensors which
        # corresponds to `per_tensor_desc` in place.
        _check_type(pred)
        flat_tensors, fmt = flatten(pred)
        if self.strict:
            assert len(flat_tensors) == len(self.per_tensor_desc), (
                f"{len(flat_tensors)} vs. {len(self.per_tensor_desc)}, num of "
                f"prediction tensors must be equal to num of description str"
            )

        for x, desc in zip(flat_tensors, self.per_tensor_desc):
            changan.set_annotation(x, desc)

        pred, left = regroup(flat_tensors, fmt)
        assert len(left) == 0, "regroup fail"
        return pred
