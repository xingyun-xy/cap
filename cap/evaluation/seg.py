# Copyright (c) Changan Auto. All rights reserved.

import json
import os
import shutil
import tarfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rich.progress import track
from torch.utils.data._utils.collate import default_convert

from cap.metrics.confusion_matrix import ConfusionMatrix
from cap.metrics.mean_iou import MeanIOU
from cap.metrics.metric_3dv import BevSegInstanceEval
from cap.utils.filesystem import file_load


def draw_confusion_matrix(confusion_matrix, title, cls_map, output_dir):
    """Confusion matrix visualization."""

    if len(cls_map["gt"]) > 10:
        plt.figure(figsize=(20, 20))
    plt.imshow(confusion_matrix, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    axis_scale = list(range(len(cls_map["gt"])))
    num_class = len(cls_map["gt"])
    plt.xticks(
        axis_scale,
        cls_map["pred"],
        rotation=45,
        ha="left",
        rotation_mode="anchor",
    )
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
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


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
        #   dist_sync_on_step=True)
        dist_sync_on_step=False,
    )
    metric_CM = ConfusionMatrix(
        seg_class=cfg["seg_class"],
        ignore_index=cfg["ignore_index"],
        # dist_sync_on_step=True)
        dist_sync_on_step=False,
    )
    tar_output = "tmp_seg"
    output_dir = prediction_fpath[: prediction_fpath.rfind("/")]
    if not os.path.exists(tar_output):
        os.makedirs(tar_output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tar = tarfile.open(prediction_fpath, "r")
    tar.extractall(path=tar_output)
    tar_output_res = os.path.join(
        tar_output,
        prediction_fpath[2 : prediction_fpath.rfind("/")],
        "tmp_result",
    )
    pkl_file = [f for f in os.listdir(tar_output_res) if f.endswith(".pkl")]
    assert (
        len(pkl_file) == 1
    ), f"{prediction_fpath} should only contain one pickle file"
    pkl_file_path = os.path.join(tar_output_res, pkl_file[0])
    total = os.path.getsize(pkl_file_path)

    for eval_data in track(
        file_load(pkl_file_path), description="Evaluation..."
    ):
        eval_data = default_convert(eval_data)
        outputs = eval_data[0]
        batch = len(outputs["image_name"])
        # for idx in range(batch):
        image_pth = os.path.join(
            dataset_fpath,
            "gt/" + outputs["image_name"].replace(".png", "_label.png"),
        )
        # print("image_pth=", image_pth)
        image = cv2.imread(image_pth, 0)
        image_reshape = np.reshape(
            image, (1, np.shape(image)[0], np.shape(image)[1])
        )
        # image = cv2.resize(image,
        #                          (320, 576))
        # image_reshape = np.reshape(image,
        #                            (1,320, 576))
        seg_outputs = cv2.imread(
            os.path.join(tar_output_res, outputs["image_name"]),
            0,
        )

        seg_reshape = cv2.resize(
            seg_outputs, (np.shape(image)[1], np.shape(image)[0])
        )
        # seg_reshape = cv2.resize(seg_outputs,
        #                          (320, 576))

        seg_reshape = np.reshape(
            seg_reshape,
            (1, np.shape(image)[0], np.shape(image)[1]),
        )
        # seg_reshape = np.reshape(
        #     seg_outputs,
        #     (1, 320, 576),
        # )
        image_reshape = default_convert(image_reshape)
        seg_reshape = default_convert(seg_reshape)
        metric_miou.update(image_reshape, seg_reshape)
        metric_CM.update(image_reshape, seg_reshape)

    _, result_miou = metric_miou.get()
    result_miou_json = {}
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
        result_CM[0].cpu(), "confusion matrix", cls_map, output_dir
    )

    return_dict = {
        "name": "seg eval",
        "state": "success",
    }
    shutil.rmtree(tar_output)
    return return_dict
