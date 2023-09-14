# Copyright (c) Changan Auto. All rights reserved.

from typing import List, Tuple

import torch


def rearrange_det_dense_head_out(
    reg_pred: List[torch.Tensor],
    cls_pred: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rearrange output of detection dense head.

    Generally, output of detection dense head should be in (N, C, H, W)
    format for each stride. This function rearranges both of them (regression
    and classification output), to concatenate predictions along each stride.
    """

    batch_size = reg_pred[0].shape[0]

    # [N, sum(h*w*num_anchors for each level), 4]
    reorg_reg_pred = [
        _reg.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for _reg in reg_pred
    ]
    hwc_lst = [out.shape[1] for out in reorg_reg_pred]

    # [N, sum(h*w*num_anchors for each level), num_classes]
    reorg_cls_pred = [
        _cls.permute(0, 2, 3, 1).reshape(batch_size, _hwc, -1)
        for _cls, _hwc in zip(cls_pred, hwc_lst)
    ]

    return torch.hstack(reorg_reg_pred), torch.hstack(reorg_cls_pred)
