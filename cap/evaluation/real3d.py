# Copyright (c) Changan Auto. All rights reserved.

import json
import os
import shutil
import tarfile
from rich.progress import track

import yaml
from torch.utils.data._utils.collate import default_convert

from cap.registry import build_from_registry
from cap.utils.apply_func import convert_numpy
from cap.utils.filesystem import file_load
from cap.metrics.real3d import Real3dEval


def evaluate(
    dataset_fpath: str,
    prediction_fpath: str,
    setting_fpath: str,
    output_dir: str,
):
    """real3d evaluation.

    Args:
        dataset_fpath: Image path to evaluation.
        prediction_fpath: Prediction results saved by sda_eval callback in
            fsd inference.It should be a tar file.
        setting_fpath: Profile the path of the configuration setting.It should
            be aYAML file.
        output_dir: The path to the result output.
    """
    # tmp_real3d/eval_res/kitti_eval/kitti_eval/cap_eval/tmp_result/result.pkl
    cfg = yaml.load(open(setting_fpath), Loader=yaml.FullLoader)
    # disable save result in real3d metric
    if "save_path" in cfg:
        cfg["save_path"] = None
    # real3d_eval = build_from_registry(cfg)
    real3d_eval = Real3dEval(
        num_dist=cfg["num_dist"],
        need_eval_categories=cfg["need_eval_categories"],
        eval_camera_names=cfg["eval_camera_names"],
        metrics=cfg["metrics"],
        iou_threshold=cfg["iou_threshold"],
        score_threshold=cfg["score_threshold"],
        depth_intervals=cfg["depth_intervals"],
        gt_max_depth=cfg["gt_max_depth"],
        # match_basis=cfg["match_basis"],
    )
    tar_output = "tmp_real3d"
    output_dir = prediction_fpath[:prediction_fpath.rfind("/")]
    if not os.path.exists(tar_output):
        os.makedirs(tar_output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tar = tarfile.open(prediction_fpath, "r")
    tar.extractall(path=tar_output)
    tar_output_res = os.path.join(
        tar_output,
        prediction_fpath[2:prediction_fpath.rfind("/")] + "/tmp_result")
    pkl_file = [f for f in os.listdir(tar_output_res) if f.endswith(".pkl")]
    assert (len(pkl_file) == 1
            ), f"{prediction_fpath} should only contain one pickle file"

    pkl_file_path = os.path.join(tar_output_res, pkl_file[0])
    total = os.path.getsize(pkl_file_path)
    for eval_data in track(file_load(pkl_file_path),
                           description="Evaluation..."):
        eval_data = default_convert(eval_data)
        batch = eval_data[0]
        outputs = batch["vehicle_heatmap_3d_detection"]
        if outputs == []:
            continue
        new_outputs = []
        for output in outputs:
            new_output = {}
            for key in output.keys():
                # if "predict" in key:
                #     index = key.index("predict") + len("predict")
                #     new_outputs[key[index + 1:]] = outputs[key]
                if key == "dim":
                    new_output[key] = output[key]
                elif key == "score":
                    new_output[key] = output[key]
                elif key == "dep":
                    new_output['dep'] = output[key]
                elif key == "location":
                    new_output[key] = output[key]
                elif key == "rotation_y":
                    new_output[key] = output[key]
                elif key == "category_id":
                    new_output[key] = output[key]
                elif key == "center":
                    new_output[key] = output[key]
                elif key == "bbox":
                    new_output[key] = output[key]
                elif key == "alpha":
                    new_output[key] = output[key]

            new_output["image_id"] = batch["id"]
            new_outputs.append(new_output)
        real3d_eval.update(batch, new_outputs)

    names, values = real3d_eval.get()
    result = {}
    result["name"] = "real3d eval"
    result["header"] = names
    result["data"] = []
    for i in range(len(values[0])):
        data_dict = {}
        for j, name in enumerate(names):
            data_dict[name] = values[j][i]
        result["data"].append(data_dict)
    result = convert_numpy(result, True)
    json.dump(result, open(os.path.join(output_dir, "result.json"), "w"))
    return_dict = {
        "name": "real3d eval",
        "state": "success",
    }
    # shutil.rmtree(tar_output)
    return return_dict


# # Comment below, since sda don't support do visualize in python3.6
# # and `Painter` is not in cap now.
# def visualize(image: np.ndarray, sample: dict):
#     """real3d visualize.
#
#     Args:
#         image: The original image.
#         sample: Data needed for visualization.It should be use result by
#             inference.
#     """
#
#     painter = Painter()
#     img = painter.draw_real3d(sample, image, 0)
#     return img
