# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn import DetectionPostProcessV1
from changan_plugin_pytorch.nn import DetectionPostProcess
from torch.quantization import DeQuantStub

from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class AnchorPostProcess(nn.Module):
    """Post process for anchor-based object detection models.

    This operation is implemented on BPU, thus is expected to be faster
    than cpu implementation. Only supported on bernoulli2.

    This operation requires input_scale = 1 / 2 ** 4, or a rescale will be
    applied to the input data. So you can manually set the output scale
    of previous op (Conv2d for example) to 1 / 2 ** 4 to avoid the rescale
    and get best performance and accuracy.

    Major differences with DetectionPostProcess:
        1. Each anchor will generate only one pred bbox totally, but in
        DetectionPostProcess each anchor will generate one bbox for each
        class (num_classes bboxes totally).
        2. NMS has a margin param, box2 will only be supressed by box1 when
        box1.score - box2.score > margin (box1.score > box2.score in
        DetectionPostProcess).
        3. An offset can be added to the output class indices (
        using class_offsets).

    Args:
        input_key: Hashable object used to query detection output from input.
        num_classes: Class number. Should be the number of foreground
            classes.
        box_filter_threshold: Default threshold to filter box by max score.
        class_offsets: Offset to be added to output class index
            for each branch.
        strides: input_size / feature_size in (h, w).
        use_clippings: Whether clip box to image size. If input is padded, you
            can clip box to real content by providing image size.
        image_size: Fixed image size in (h, w), set to None if input have
            different sizes.
        nms_threshold: IoU threshold for nms.
        nms_margin: Only supress box2 when
            box1.score - box2.score > nms_margin
        pre_nms_top_k: Maximum number of bounding boxes in each image before
            nms.
        post_nms_top_k: Maximum number of output bounding boxes in each
            image.
        nms_padding_mode: The way to pad bbox to match the number of output
            bounding bouxes to post_nms_top_k, can be None, "pad_zero" or
            "rollover".
        bbox_min_hw: Minimum height and width of selected bounding boxes.
    """

    def __init__(
        self,
        input_key: Hashable,
        num_classes: int,
        class_offsets: List[int],
        use_clippings: bool,
        image_hw: Tuple[int, int],
        nms_iou_threshold: float,
        pre_nms_top_k: int,
        post_nms_top_k: int,
        nms_margin: float = 0.0,
        box_filter_threshold: float = 0.0,
        nms_padding_mode: Optional[str] = None,
        bbox_min_hw: Tuple[float, float] = (0, 0),
    ) -> None:

        super().__init__()
        self.input_key = input_key
        self.dequant = DeQuantStub()
        self._dpp = DetectionPostProcessV1(
            num_classes,
            box_filter_threshold,
            class_offsets,
            use_clippings,
            image_hw,
            nms_iou_threshold,
            pre_nms_top_k,
            post_nms_top_k,
            nms_margin=nms_margin,
            nms_padding_mode=nms_padding_mode,
            bbox_min_hw=bbox_min_hw,
        )

    def forward(
        self,
        anchors: List[torch.Tensor],
        head_out: Dict[str, List[torch.Tensor]],
        im_hw: Optional[Tuple[int, int]] = None,
    ) -> List[List[Tuple[torch.Tensor]]]:
        """Forward method.

        The output keyed by "pred_boxes_out" is the float version of
        "pred_boxes", which is used in qat&pt inference.
        """

        head_out = [t.detach() for t in head_out[self.input_key]]
        result_lst = self._dpp(head_out, anchors, image_sizes=im_hw)
        # import numpy as np
        # np.save(
        #     "dpp_head_out_3.npy", [it.cpu().detach().numpy() for it in head_out][3], allow_pickle=True
        # )
        # np.save(
        #     "dpp_anchors_3.npy", [it.cpu().detach().numpy() for it in anchors][3], allow_pickle=True
        # )
        # np.save(
        #     "dpp_result_lst_2.npy", [it.cpu().detach().numpy() for it in result_lst[0]][2], allow_pickle=True
        # )
        return OrderedDict(
            pred_boxes_out=[self.dequant(r[0]) for r in result_lst],
            pred_boxes=[r[0] for r in result_lst],
            pred_scores=[r[1] for r in result_lst],
            pred_cls=[r[2] for r in result_lst],
        )
