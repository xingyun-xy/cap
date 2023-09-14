# Copyright (c) Changan Auto. All rights reserved.

__all__ = ["img_metas"]


img_metas = [
    # img info
    "image_name",
    "image_id",
    "img",
    "layout",
    "ori_img",
    "color_space",
    "img_shape",
    "scale_factor",
    "crop_offset",
    # cls info
    "labels",
    # bbox info
    "gt_bboxes",
    "gt_classes",
    "gt_difficult",
    # seg info
    "gt_seg",
    "gt_seg_weight",
    # flow info
    "gt_flow",
    "gt_ori_flow",
    # ldmk
    "gt_ldmk",
    "ldmk_pairs",
]
