# Copyright (c) Changan Auto. All rights reserved.

import json
import os
import shutil
import tarfile
from typing import Sequence

import cv2
import numpy as np
import yaml
from torch.utils.data._utils.collate import default_convert

from cap.metrics.metric_3dv import AbsRel
from cap.utils.filesystem import file_load


def load_depth_png(depth_path, scale):
    depth_array = cv2.imread(depth_path, -1)
    depth_array = depth_array.astype("float32")
    depth_array /= scale
    return depth_array


def eval_depth_preprocess(
    pred: np.ndarray, pred_scale: float, gt: np.ndarray, gt_crop: Sequence[int]
):
    pred *= pred_scale
    height, width, top, left = gt_crop
    gt = gt[top : top + height, left : left + width]
    return gt, pred


def evaluate(
    dataset_fpath: str,
    prediction_fpath: str,
    setting_fpath: str,
    output_dir: str,
):
    """2p5d_depth evaluation.

    Args:
        dataset_fpath: Image path to evaluation.
        prediction_fpath: Prediction results saved by sda_eval callback in
            fsd inference.It should be a tar file.
        setting_fpath: Profile the path of the configuration setting.It should
            be aYAML file.
        output_dir: The path to the result output.
    """

    cfg = yaml.load(open(setting_fpath), Loader=yaml.FullLoader)
    absrel_eval = AbsRel(range_list=cfg["range_list"], name=cfg["name"])
    tar_output = "tmp_depth"
    if not os.path.exists(tar_output):
        os.makedirs(tar_output)
    tar = tarfile.open(prediction_fpath, "r")
    tar.extractall(path=tar_output)
    pkl_file = [f for f in os.listdir(tar_output) if f.endswith(".pkl")]
    assert (
        len(pkl_file) == 1
    ), f"{prediction_fpath} should only contain one pickle file"

    pkl_file_path = os.path.join(tar_output, pkl_file[0])
    for eval_data in file_load(pkl_file_path):
        eval_data = default_convert(eval_data)
        outputs = eval_data["outputs"]
        batch = len(outputs["img_name"])
        for idx in range(batch):
            gt_image = load_depth_png(
                os.path.join(
                    dataset_fpath, outputs["img_name"][idx][:-3] + "png"
                ),
                2 ** 8,
            )
            pred_depths_image = load_depth_png(
                os.path.join(
                    tar_output,
                    "pred_depths_" + outputs["img_name"][idx][:-3] + "png",
                ),
                2 ** 8,
            )
            absrel_batch, absrel_outputs = eval_depth_preprocess(
                pred_depths_image, 10, gt_image, (2048, 3840, 0, 0)
            )
            absrel_batch = np.reshape(
                absrel_batch,
                (1, 1, np.shape(absrel_batch)[0], np.shape(absrel_batch)[1]),
            )
            absrel_outputs = np.reshape(
                absrel_outputs,
                (
                    1,
                    1,
                    np.shape(absrel_outputs)[0],
                    np.shape(absrel_outputs)[1],
                ),
            )
            absrel_outputs = np.repeat(absrel_outputs, 2, axis=1)
            absrel_batch = default_convert(absrel_batch)
            absrel_outputs = default_convert(absrel_outputs)
            absrel_eval.update(absrel_batch, absrel_outputs)
    _, result_absrel = absrel_eval.get()
    result = {}
    result["absrel"] = result_absrel
    json.dump(result, open(os.path.join(output_dir, "result.json"), "w"))
    shutil.rmtree(tar_output)
    return result


# # Comment below, since sda don't support do visualize in python3.6
# # and `Painter` is not in cap now.
# def visualize(image: np.ndarray, sample: dict):
#     """2p5d_depth visualize.
#
#     Args:
#         image: The original image.
#         sample: Data needed for visualization.It should be use result by
#             inference.
#     """
#
#     painter = Painter()
#     depth_img, residual_img = painter.draw_depth_resflow_pose(
#             sample, image, 0)
#     imgs = np.hstack([depth_img, residual_img])
#     return imgs
