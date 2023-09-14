# Copyright (c) Changan Auto. All rights reserved.

import json
import os
import shutil
import tarfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from torch.utils.data._utils.collate import default_convert

from cap.metrics.confusion_matrix import ConfusionMatrix
from cap.metrics.mean_iou import MeanIOU
from cap.metrics.metric_3dv import BevSegInstanceEval
from cap.utils.filesystem import file_load


def draw_confusion_matrix(confusion_matrix, title, cls_map, output_dir):
    """Confusion matrix visualization."""

    plt.imshow(confusion_matrix, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    axis_scale = list(range(len(cls_map["gt"])))
    num_class = len(cls_map["gt"])
    plt.xticks(axis_scale, cls_map["pred"], rotation=45)
    plt.yticks(axis_scale, cls_map["gt"])
    plt.ylabel("Groundtruth")
    plt.xlabel("Predict")
    ax = plt.gca()
    ax.xaxis.set_ticks_position("top")
    for i in range(num_class):
        for j in range(num_class):
            ax.text(
                j,
                i,
                "{:.1%}".format(confusion_matrix[i][j]),
                ha="center",
                va="center",
                color="w",
            )
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"),
        dpi=500,
        bbox_inches="tight",
    )


def evaluate(
    dataset_fpath: str,
    prediction_fpath: str,
    setting_fpath: str,
    output_dir: str,
):
    """bev_seg evaluation.

    Args:
        dataset_fpath: Image path to evaluation.
        prediction_fpath: Prediction results saved by sda_eval callback in
            fsd inference.It should be a tar file.
        setting_fpath: Profile the path of the configuration setting.It should
            be aYAML file.
        output_dir: The path to the result output.
    """

    cfg = yaml.load(open(setting_fpath), Loader=yaml.FullLoader)
    seg_class = cfg["seg_class"]
    metric_miou = MeanIOU(
        seg_class=cfg["seg_class"],
        global_ignore_index=cfg["global_ignore_index"],
        verbose=True,
    )
    metric_bevseg_instance_eval = BevSegInstanceEval(
        seg_class=cfg["bev_seg_names"],
        metirc_key=cfg["metirc_key"],
        vcs_origin_coord=cfg["vcs_origin_coord"],
        target_categorys=cfg["target_categorys"],
        target_x_intervals=cfg["target_x_intervals"],
        target_y_intervals=cfg["target_y_intervals"],
        gt_min_x=cfg["gt_min_x"],
        gt_max_x=cfg["gt_max_x"],
        max_distance=cfg["max_distance"],
        verbose=True,
    )
    metric_bevseg_instance_eval.reset()
    metric_CM = ConfusionMatrix(
        seg_class=cfg["seg_class"], ignore_index=cfg["ignore_index"]
    )
    tar_output = "tmp_bev"
    if not os.path.exists(tar_output):
        os.makedirs(tar_output)
    tar = tarfile.open(prediction_fpath, "r")
    tar.extractall(path=tar_output)
    pkl_file = [f for f in os.listdir(tar_output) if f.endswith(".pkl")]
    assert (
        len(pkl_file) == 1
    ), f"{prediction_fpath} should only contain one pickle file"
    gt_json_path = os.path.join(dataset_fpath, "bev_seg.json")
    gt_json = json.load(open(gt_json_path))
    pkl_file_path = os.path.join(tar_output, pkl_file[0])
    for eval_data in file_load(pkl_file_path):
        eval_data = default_convert(eval_data)
        outputs = eval_data["outputs"]
        batch = len(outputs["img_name"])
        for idx in range(batch):
            image = cv2.imread(
                os.path.join(dataset_fpath, outputs["img_name"][idx]), 0
            )
            image_reshape = np.reshape(
                image, (1, np.shape(image)[0], np.shape(image)[1])
            )

            bev_outputs = cv2.imread(
                os.path.join(
                    tar_output, "bev_seg_" + outputs["img_name"][idx]
                ),
                0,
            )
            bev_reshape = np.reshape(
                bev_outputs,
                (1, np.shape(bev_outputs)[0], np.shape(bev_outputs)[1]),
            )
            bev_anno = gt_json[outputs["img_name"][idx].split(".")[0]]
            for _key in bev_anno.keys():
                bev_anno[_key] = default_convert(
                    np.expand_dims(np.array(bev_anno[_key]), axis=0)
                )
            image_reshape = default_convert(image_reshape)
            bev_reshape = default_convert(bev_reshape)
            metric_miou.update(image_reshape, bev_reshape)
            metric_CM.update(image_reshape, bev_reshape)
            metric_bevseg_instance_eval.update(bev_anno, image_reshape)

    _, result_miou = metric_miou.get()
    _, result_bevseg_eval = metric_bevseg_instance_eval.get()
    result_miou_json = {}
    result_miou_json["bev_seg_distance"] = result_bevseg_eval
    result_miou_json["mean_iou"] = "{:.4f}".format(result_miou[0].cpu().item())
    result_miou_json["mean_acc"] = "{:.4f}".format(result_miou[1].cpu().item())
    result_miou_json["all_acc"] = "{:.4f}".format(result_miou[2].cpu().item())
    result_miou_json["classes"] = [
        {
            "name": "%s" % x,
            "iou": "%.4f" % y.cpu().item(),
            "acc": "%.4f" % z.cpu().item(),
        }
        for x, y, z in zip(result_miou[5], result_miou[3], result_miou[4])
    ]
    json.dump(
        result_miou_json,
        open(os.path.join(output_dir, "result.json"), "w"),
    )

    _, result_CM = metric_CM.get()
    cls_map = {
        "gt": ["%s(%d)" % (x, y) for x, y in zip(seg_class, result_CM[1])],
        "pred": ["%s(%d)" % (x, y) for x, y in zip(seg_class, result_CM[2])],
    }
    draw_confusion_matrix(
        result_CM[0], "confusion matrix", cls_map, output_dir
    )

    return_dict = {
        "name": "bev_seg eval",
        "state": "success",
    }
    shutil.rmtree(tar_output)
    return return_dict


# # Comment below, since sda don't support do visualize in python3.6
# # and `Painter` is not in cap now.
# def visualize(image: np.ndarray, sample: dict):
#     """bev_seg visualize.
#
#     Args:
#         image: The gt image.
#         sample: Data needed for visualization.It should be use result by
#             inference.
#     """
#
#     painter = Painter()
#     bev_img, _ = painter.draw_bev(sample, image, 0)
#     return bev_img
