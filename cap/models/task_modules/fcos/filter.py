# Copyright (c) Changan Auto. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cap.models.base_modules.postprocess import FilterModule
from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FCOSMultiStrideCatFilter(nn.Module):
    """A modified Filter used for post-processing of FCOS.

    In each stride, concatenate the scores of each task as
    the first input of FilterModule, which can reduce latency in BPU.

    Args:
        strides (Sequence[int]): A list contains the strides of feature maps.
        idx_range (Optional[Tuple[int, int]], optional): The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
        threshold (float): The lower bound of output.
        task_strides(Sequence[Sequence[int]]): A list of out_stirdes of each
            task.
    """

    def __init__(
        self,
        strides: Sequence[int],
        threshold: float,
        task_strides: Sequence[Sequence[int]],
        idx_range: Optional[Tuple[int, int]] = None,
    ):
        super(FCOSMultiStrideCatFilter, self).__init__()
        self.cat_op = nn.quantized.FloatFunctional()
        self.strides = strides
        self.num_level = len(self.strides)
        self.task_strides = task_strides
        for i in range(len(self.strides)):
            setattr(
                self,
                "filter_module_%s" % str(i),
                FilterModule(threshold=threshold, idx_range=idx_range),
            )
        self.set_qconfig()

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        **kwargs,
    ) -> Sequence[torch.Tensor]:

        mlvl_outputs = []

        for stride_ind, stride in enumerate(self.strides):
            score_list = []
            filter_input = []
            for task_ind in range(len(self.task_strides)):
                pred = preds[task_ind]
                if stride in self.task_strides[task_ind]:
                    # bbox
                    filter_input.append(
                        pred[1][self.task_strides[task_ind].index(stride)]
                    )
                    # centerness
                    filter_input.append(
                        pred[2][self.task_strides[task_ind].index(stride)]
                    )
                    # score
                    score_list.append(
                        pred[0][self.task_strides[task_ind].index(stride)]
                    )
            # concatenate the scores of each task as the first input of filter
            if len(score_list) > 1:
                per_level_cls_scores = self.cat_op.cap(score_list, dim=1)
            else:
                per_level_cls_scores = score_list[0]
            filter_input.insert(0, per_level_cls_scores)

            for per_filter_input in filter_input:
                assert (
                    len(per_filter_input.shape) == 4
                ), "should be in NCHW layout"
            filter_output = getattr(
                self, "filter_module_%s" % str(stride_ind)
            )(*filter_input)
            per_sample_outs = []
            for task_ind in range(len(filter_output)):
                per_sample_outs.append(filter_output[task_ind][2:])
            mlvl_outputs.append(per_sample_outs)

        return mlvl_outputs

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


# TODO(kongtao.hu 0.5): May need to be refactored to become more general.
@OBJECT_REGISTRY.register
class FCOSMultiStrideFilter(nn.Module):  # noqa: D205,D400
    """Filter used for post-processing of
    `FCOS <https://arxiv.org/pdf/1904.01355.pdf>`_.

    Args:
        strides (Sequence[int]): A list contains the strides of feature maps.
        idx_range (Optional[Tuple[int, int]], optional): The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
        threshold (float): The lower bound of output.
    """

    def __init__(
        self,
        strides: Sequence[int],
        threshold: float,
        idx_range: Optional[Tuple[int, int]] = None,
    ):
        super(FCOSMultiStrideFilter, self).__init__()
        self.strides = strides
        self.num_level = len(strides)
        self.filter_module = FilterModule(
            threshold=threshold,
            idx_range=idx_range,
        )

    def _filter_forward(self, preds):
        mlvl_outputs = []
        cls_scores, bbox_preds, centernesses = preds
        for level in range(self.num_level):
            (
                per_level_cls_scores,
                per_level_bbox_preds,
                per_level_centernesses,
            ) = (cls_scores[level], bbox_preds[level], centernesses[level])
            filter_input = [
                per_level_cls_scores,
                per_level_bbox_preds,
                per_level_centernesses,
            ]
            for per_filter_input in filter_input:
                assert (
                    len(per_filter_input.shape) == 4
                ), "should be in NCHW layout"
            filter_output = self.filter_module(*filter_input)
            # len(filter_output) equal to batch size
            per_sample_outs = []
            for i in range(len(filter_output)):
                (
                    _,
                    _,
                    per_img_coord,
                    per_img_score,
                    per_img_bbox_pred,
                    per_img_centerness,
                ) = filter_output[i]
                per_sample_outs.append(
                    [
                        per_img_coord,
                        per_img_score,
                        per_img_bbox_pred,
                        per_img_centerness,
                    ]
                )
            mlvl_outputs.append(per_sample_outs)

        return mlvl_outputs

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        meta_and_label: Optional[Dict] = None,
        **kwargs,
    ) -> Sequence[torch.Tensor]:
        preds = self._filter_forward(preds)
        return preds
