# Copyright (c) Changan Auto. All rights reserved.

import numpy as np

__all__ = ["draw_heatmap"]


def draw_heatmap(
    heatmap,
    insert_hm,
    cxy,
    reg_map_list=None,
    insert_reg_map_list=None,
    op="max",
):
    x, y = int(cxy[0]), int(cxy[1])
    ry, rx = (np.array(insert_hm.shape[:2]) - 1) // 2
    height, width = heatmap.shape[:2]

    left, right = min(x, rx), min(width - x, rx + 1)
    top, bottom = min(y, ry), min(height - y, ry + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_insert_hm = insert_hm[
        ry - top : ry + bottom, rx - left : rx + right
    ]
    if reg_map_list and insert_reg_map_list:
        masked_reg_map_list = [
            reg_map[y - top : y + bottom, x - left : x + right]
            for reg_map in reg_map_list
        ]
        masked_insert_reg_map_list = [
            insert_reg_map[ry - top : ry + bottom, rx - left : rx + right]
            for insert_reg_map in insert_reg_map_list
        ]

    if min(masked_insert_hm.shape) > 0 and min(masked_heatmap.shape) > 0:
        if op == "max":
            mask = masked_insert_hm > masked_heatmap
            masked_heatmap[mask] = masked_insert_hm[mask]
            if reg_map_list and insert_reg_map_list:
                for (masked_reg_map, masked_insert_reg_map) in zip(
                    masked_reg_map_list, masked_insert_reg_map_list
                ):
                    masked_reg_map[mask] = masked_insert_reg_map[mask]
        elif op == "overwrite":
            masked_heatmap[:] = masked_insert_hm
            if reg_map_list and insert_reg_map_list:
                for (masked_reg_map, masked_insert_reg_map) in zip(
                    masked_reg_map_list, masked_insert_reg_map_list
                ):
                    masked_reg_map[:] = masked_insert_reg_map
        else:
            raise NotImplementedError
