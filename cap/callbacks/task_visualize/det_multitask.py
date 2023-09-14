# Copyright (c) Changan Auto. All rights reserved.
import logging
import os
from typing import Dict, Sequence, Union

import cv2

from cap.core.data_struct.img_structures import ImgBase
from cap.registry import OBJECT_REGISTRY
from cap.utils.distributed import get_dist_info
from cap.visualize.bevdepvisual.get_bboxes import BevBBoxes
from cap.visualize.bevdepvisual.visualize_nusc import bev_vis, changan_visual

# from cap.visualize.bevdepvisual.visualize_nusc import demo
from cap.visualize.vis_img_struct import vis_img_struct

from .visualize import BaseVisualize

__all__ = ["DetMultitaskVisualize"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DetMultitaskVisualize(BaseVisualize):
    """
    DetVisualize callback is used for visualize results on 2d-detection tasks.

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
        out_keys: Union[str, Sequence[str]],
        vis_configs: Dict[str, Dict],
        output_dir: str = "./tmp_viz_imgs/",
        save_viz_imgs: bool = False,
    ):
        super().__init__()
        self.out_keys = out_keys
        self.vis_configs = vis_configs
        self.output_dir = output_dir
        self.save_viz_imgs = save_viz_imgs

        rank, _ = get_dist_info()
        if rank == 0 and self.save_viz_imgs:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def visualize(self, global_id, batch, result):

        if "singletask_bev" in self.out_keys:
            result_bev = result["singletask_bev"]

            # import numpy as np
            # np.save("preds.npy", result_bev[0][0])

            bv = BevBBoxes(result_bev)
            """Remove comment when u want to visualize changan bev!!!! added by zwj"""
            # changan bev visual
            # ret_list,img_metas, path= bv.changan_gen_boxes()

            # file_name = img_metas[0][0].split("/")[-1]

            # changan_visual(ret_list, batch["sensor2ego_trans"], batch["sensor2ego_rot"], batch["intrin_mats"],
            #                path + file_name, batch["cameras_paths"],
            #                return_undistort_imgs = True, score_threshold = 0.6, limit_range = 61.2)
            # changan bev visual

            # nuscenes visual
            bev_f, all_img_metas, path, ret_lists = bv.gen_boxes()
            # process ret_lists for bev eval     add by zmj
            import torch

            for ret_list in ret_lists:
                for i in range(len(ret_list)):
                    if isinstance(ret_list[i], torch.Tensor):
                        ret_list[i] = ret_list[i].detach().cpu().numpy()
                ret_list[3].pop("box_type_3d")
            result_bev_list = list(result_bev)  # tuple -> list
            result_bev_list.append(ret_lists)
            result["singletask_bev"] = tuple(result_bev_list)

            # bev_vis(bev_f,all_img_metas,path)  #BEV评测时注释掉可以加快评测
            if len(self.out_keys) != 1:
                model_outs = [
                    result[key] for key in self.out_keys
                    if key != "singletask_bev"
                ]
            # nuscenes visual
        else:
            model_outs = [result[key] for key in self.out_keys]

        if "model_outs" in locals():
            if isinstance(batch, tuple):
                batch = batch[0]

            for i, model_out in enumerate(zip(*model_outs)):
                calib = batch.get("calib", None)
                if calib is not None:
                    calib = calib[i]
                    # real3d推理结果是在原图尺寸上的结果，
                    # 要在resize图上可视化的话，需要对内参进行变换
                    # add by zmj
                    scale_x, scale_y = (
                        batch["img_width"][i] / batch["ori_img_shape"][i][1],
                        batch["img_height"][i] / batch["ori_img_shape"][i][0],
                    )
                    calib[0], calib[1] = calib[0] * scale_x, calib[1] * scale_y

                distCoeffs = batch.get("distCoeffs", None)
                if distCoeffs is not None:
                    distCoeffs = distCoeffs[i]
                img_struct = ImgBase(
                    img_id=batch["img_id"][i],
                    img=batch["img"][i],
                    ori_img=batch["ori_img"][i],
                    layout=batch["layout"][i],
                    color_space=batch["color_space"][i],
                    img_width=batch["img_width"][i],
                    img_height=batch["img_height"][i],
                    calib=calib,
                    distCoeffs=distCoeffs,
                )

                vis_img = None
                for k, res in zip(self.out_keys, model_out):
                    vis_configs = self.vis_configs.get(k, None)
                    if self.save_viz_imgs and vis_configs is not None:
                        res = res.to("cpu")
                        vis_img = vis_img_struct(img_struct,
                                                 res,
                                                 vis_configs,
                                                 vis_image=vis_img)

                if self.save_viz_imgs:
                    # write_path = os.path.join(
                    #     self.output_dir, f"{img_struct.img_id}.png"
                    # )
                    write_path = os.path.join(
                        self.output_dir,
                        f"{os.path.splitext(batch['img_name'][i])[0]}_pred.png",
                    )
                    if os.path.exists(write_path):
                        break
                    cv2.imwrite(write_path, vis_img)

        return global_id, None, result

    def __repr__(self):
        return "DetMultitaskVisualize"
