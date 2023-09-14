# Copyright (c) Changan Auto. All rights reserved.
import copy
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from cap.core.data_struct.img_structures import ImgObjDet
from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info
from cap.visualize import draw_bbox2d
from .visualize import BaseVisualize

__all__ = ["Det2dVisualize", "default_match_task"]

logger = logging.getLogger(__name__)


def default_match_task(batch, model_outs):
    if "structure" in batch:
        objects = batch["structure"]
    else:
        raise KeyError("Cannot find img structure in batch data!")

    preds, _ = model_outs

    object_rets = []
    bs = len(objects)
    for i in range(bs):
        object2d = ImgObjDet(
            img_id=objects[i].img_id,
            img=objects[i].img,
            gt_bboxes=preds[i][:, :5],
            layout=objects[i].layout,
            color_space=objects[i].color_space,
            img_width=objects[i].img_width,
            img_height=objects[i].img_height,
        )
        object_rets.append(object2d)

    return object_rets, preds


@OBJECT_REGISTRY.register
class Det2dVisualize(BaseVisualize):
    """
    Det2d callback is used for visualize results on 2d-detection tasks.

    Args:
        output_dir: Output dir for saving results.
        save_viz_imgs: Whether to save viz imgs.
        viz_threshold: Score threshold of bbox viz.
        colors: Colors for plotting bbox info.
        thickness: Thickness for plotting bbox info.
        class_name: Names of class.
        match_task_output: Function to match current task outputs.
    """

    def __init__(
        self,
        output_dir: str = "./tmp_viz_imgs/",
        save_viz_imgs: bool = False,
        viz_threshold: float = 0.5,
        colors: Optional[Dict[int, Tuple[int]]] = None,
        thickness: int = 2,
        class_names: Optional[List[str]] = None,
        match_task_output: Callable = default_match_task,
    ):
        super().__init__(match_task_output=match_task_output)
        self.class_names = class_names
        self.viz_threshold = viz_threshold
        self.colors = colors
        self.thickness = thickness
        self.output_dir = output_dir
        self.save_viz_imgs = save_viz_imgs

        rank, _ = get_dist_info()
        if rank == 0 and self.save_viz_imgs:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def visualize(self, global_id, batch, result):
        det2d_results, preds = batch, result
        bs = len(det2d_results)

        new_batch = copy.deepcopy(batch)
        for i in range(bs):
            scores = preds[i][:, 5]
            ret = det2d_results[i].gt_bboxes
            valid_idx = scores > self.viz_threshold
            ret = ret[valid_idx]
            det2d_results[i].gt_bboxes = ret
            score = scores[valid_idx]

            plot_img = draw_bbox2d(
                det_object=det2d_results[i],
                scores=score,
                score_thresh=self.viz_threshold,
                class_names=self.class_names,
                colors=self.colors,
                thickness=self.thickness,
            )
            new_batch[i].img = plot_img

            if self.save_viz_imgs:
                write_path = os.path.join(
                    self.output_dir, f"{det2d_results[i].img_id}.png"
                )
                if os.path.exists(write_path):
                    break
                cv2.imwrite(write_path, plot_img)
        return global_id, new_batch, result

    def __repr__(self):
        return "Det2dVisaualize"
