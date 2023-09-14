# Copyright (c) Changan Auto. All rights reserved.
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.nn.functional import interpolate

from cap.core.data_struct.base_struct import Mask
from cap.models.base_modules.resize_parser import resize
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list

__all__ = ["SegDecoder", "VargNetSegDecoder"]


@OBJECT_REGISTRY.register
class SegDecoder(object):
    """Semantic Segmentation Decoder.

    Args:
        out_strides: List of output strides, represents the strides
            of the output from seg_head.
        output_names: Keys of returned results dict.
        decode_strides: Strides that need to be decoded,
            should be a subset of out_strides.
        upscale_times: Bilinear upscale times for each
            decode stride, default to None, which means same as decode stride.
        transforms: A list contains the transform config.
        inverse_transform_key: A list contains the inverse
            transform info key.
    """

    def __init__(
        self,
        out_strides: List[int],
        decode_strides: List[int],
        upscale_times: List[int] = None,
        transforms: List[dict] = None,
        inverse_transform_key: List[str] = None,
        output_names: Optional[str] = "pred_seg",
    ):
        self.out_strides = out_strides
        output_names = _as_list(output_names)
        decode_strides = _as_list(decode_strides)
        upscale_times = (
            _as_list(upscale_times)
            if upscale_times is not None
            else decode_strides
        )
        assert len(output_names) == len(decode_strides)
        assert set(decode_strides).issubset(out_strides)
        self.output_names = output_names
        self.decode_strides = decode_strides
        self.upscale_times = upscale_times
        self.transforms = transforms
        self.inverse_transform_key = inverse_transform_key

    def __call__(
        self, pred: Sequence[torch.Tensor], label: dict, *args
    ) -> dict:  # noqa: D205,D400
        """
        Args:
            label: Meta data dict.
            pred: Output corresponding to multiple strides, the
                shape of each element is NHWC, HW is different for each stride.
            *args: Receive extra parameters, not used.

        """
        results = {}
        new_results = []
        for stride, upscale_time, out in zip(
            self.out_strides, self.upscale_times, pred
        ):
            if stride in self.decode_strides:
                index = self.decode_strides.index(stride)
                assert len(out.shape) == 4, "bilinear need 4D input"
                _, _, h, w = out.shape
                if upscale_time > 1:
                    out = resize(
                        input=out,
                        size=(h * upscale_time, w * upscale_time),
                        mode="bilinear",
                        align_corners=False,
                        warning=False,
                    )
                if stride // upscale_time > 1:
                    out = out.argmax(dim=1, keepdim=True)
                    out = resize(
                        input=out,
                        size=(h * stride, w * stride),
                        mode="nearest",
                        align_corners=None,
                        warning=False,
                    )
                    result = out.squeeze(dim=1)
                else:
                    result = out.argmax(dim=1)
                # do inverse transform
                flag = False
                for idx, single_result in enumerate(result):
                    inverse_info = {}
                    for key, value in label.items():
                        if (
                            self.inverse_transform_key
                            and key in self.inverse_transform_key
                        ):
                            inverse_info[key] = value[idx]
                    if self.transforms:
                        for transform in self.transforms[::-1]:
                            if hasattr(transform, "inverse_transform"):
                                single_result = transform.inverse_transform(
                                    inputs=single_result,
                                    task_type="segmentation",
                                    inverse_info=inverse_info,
                                )
                                flag = True
                    if not flag:
                        results[self.output_names[index]] = result
                        break
                    new_results.append(single_result)
                if flag:
                    results[self.output_names[index]] = result.new_tensor(
                        np.array(new_results)
                    )
        return results


@OBJECT_REGISTRY.register
class VargNetSegDecoder(object):
    """Semantic Segmentation Decoder.

    Args:
        out_strides (list[int]): List of output strides, represents the strides
            of the output from seg_head.
        output_names (str or list[str]): Keys of returned results dict.
        decode_strides (int or list[int]): Strides that need to be decoded,
            should be a subset of out_strides.
        transforms (Sequence[dict]): A list contains the transform config.
        inverse_transform_key (Sequence[str]): A list contains the inverse
            transform info key.
    """

    def __init__(
        self,
        out_strides: List[int],
        input_padding: Sequence[int] = (0, 0, 0, 0),
        transforms: List[dict] = None,
    ):
        self.out_strides = out_strides
        assert len(input_padding) == 4
        self.input_padding = input_padding
        self.transforms = transforms

    def __call__(self, pred: Sequence[torch.Tensor]):

        assert len(pred) == 1

        stride = self.out_strides[0]
        pred = pred[0]

        if pred.ndim == 4 and pred.shape[1] > 1:
            pred = pred.argmax(dim=1, keepdim=True)

        pred = interpolate(
            pred.float(),
            scale_factor=(stride, stride),
            mode="nearest",
        ).long()[:, 0]

        res = []
        for p in pred:
            res_struct = Mask(p)
            res_struct.inv_pad(*self.input_padding)
            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    res_struct = transform.inverse_transform(res_struct)
            res.append(res_struct)
        return res
