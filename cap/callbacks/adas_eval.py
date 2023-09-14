# Copyright (c) Changan Auto. All rights reserved.
import json
import logging
import os
import time

import cv2
import torch

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from cap.utils.distributed import rank_zero_only
from .callbacks import CallbackMixin

logger = logging.getLogger(__name__)


def _generate_name_dir_list(
    task_name_list, task_type_list, save_path
):  # noqa: D205,D400
    """Generate a list, which is consist of filename.json and segmentation's
    path.

    Args:
        task_name_list (list[str]):
        task_type_list (list[str]):
        save_path str: Results' path

    Returns (list[str]): The result's list
        if the value is None, the task we do not save

    """
    name_dir_list = []
    for task_name, task_type in zip(task_name_list, task_type_list):
        if task_name is None or task_type is None:
            name_dir_list.append(None)
            continue
        if task_type == "detection":
            json_file = os.path.join(save_path, task_name + ".json")
            name_dir_list.append(json_file)
        elif task_type == "segmentation":
            seg_save_path = os.path.join(save_path, task_name)
            if not os.path.exists(seg_save_path):
                os.makedirs(seg_save_path)
            name_dir_list.append(seg_save_path)
        else:
            raise Exception(
                "error task_type, task_type_list's value "
                "must in ['detection', 'segmentation']"
            )  # noqa
    return name_dir_list


def _tensor_to_ndarray(output):
    """Convert output type from tensor to ndarray.

    Args:
        output (dict): Model's output

    Returns (dict): Type is ndarray

    """
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            output[k] = v.cpu().numpy()
        else:
            output[k] = v
    return output


def _to_json_format(batch, output, task_name):
    """Change the output to json format.

    Args:
        batch: Model input, which is consisted of image name..
        output: Model output, which is consisted of detection results
        task_name (str): The Json's key

    """
    # check the output's format tensor to np.ndarray
    output = _tensor_to_ndarray(output)
    img_data = []
    assert isinstance(batch["image_name"], list)
    for i in range(len(batch["image_name"])):
        one_img_data = {}
        image_key = batch["image_name"][i]
        one_img_data["image_key"] = image_key
        one_img_data[task_name] = []

        bboxes = output["pred_bboxes"][i][:, :4]
        bboxes_scores = output["pred_bboxes"][i][:, 4]
        for j in range(bboxes.shape[0]):
            one_bbox = {}
            one_bbox["bbox"] = bboxes[j].tolist()
            one_bbox["bbox_score"] = float(bboxes_scores[j])
            one_img_data[task_name].append(one_bbox)
        img_data.append(one_img_data)
    return img_data


def _save_segmentation_png(save_path, batch, output):
    """Save segmentation results to png.

    Args:
        save_path (str): Path is fullded with image
        batch: Model input, which is consisted of image name..
        output: Model output, which is consisted of segmentation results

    """
    isinstance(batch["image_name"], list)
    output = _tensor_to_ndarray(output)
    for i in range(len(batch["image_name"])):
        save_file = os.path.join(save_path, batch["image_name"][i])
        cv2.imwrite(save_file, output["pred_seg"][i])


@OBJECT_REGISTRY.register
class AdasEval(CallbackMixin):  # noqa: D205,D400
    """AdasEval is used to generate detection json or segmentation png
    to adas evaluation system.

    Args:
        dataloaders (list[dataloader]): the eval dataloaders
        save_dir (str): The adas_eval results save directory, we will create
        task_type_list (list[str]):
            for example list['detection',None, 'segmentation']
            if the value = None, the dataloader will not be evaled
        task_name_list (list(str)):
            for example [vehicle, None, lane_line]
            if the value = None, the dataloader will not be evaled
        Warning: the length of task_type_list、
            task_name_list、dataloaders must be same

    """

    def __init__(
        self,
        dataloaders,
        save_dir=None,
        task_type_list=None,
        task_name_list=None,
    ):
        dataloaders = _as_list(dataloaders)
        self.dataloaders = dataloaders
        self.datanames = ["dataset_%i" % (i) for i in range(len(dataloaders))]
        self.save_dir = save_dir
        self.task_type_list = task_type_list
        self.task_name_list = task_name_list

        # check the save path exit
        now_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.save_dir = os.path.join(self.save_dir, now_time + "_adas_eval")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # check the task_type_list 、dataloaders 、task_name_list
        assert len(self.task_type_list) > 0
        assert (
            len(self.task_type_list)
            == len(self.dataloaders)
            == len(self.task_name_list)
        )  # noqa

    def save_part_results(self, index, batch, output, result):
        """Save part detection result into list.

        Save part segmentation result into dir.

        Args:
            index (int): The dataloader's index
            batch (dict): The model's input
            output (dict): The model's output
            result (list): The list collect the result of detection

        """
        task_type = self.task_type_list[index]
        task_name = self.task_name_list[index]
        if task_type == "detection":
            result.extend(_to_json_format(batch, output, task_name))
        elif task_type == "segmentation":
            _save_segmentation_png(self.name_dir_list[index], batch, output)
        else:
            raise Exception(
                "error task_type, task_type_list's value "
                "must in ['detection', 'segmentation']"
            )

    def save_detecion_to_json(self, result, json_name):
        if len(result) > 0:
            # total_detection_res.append(single_detection_res)
            # save single detection result into json
            save_file = open(json_name, "w")
            for one_img_data in json_name:
                jsObj = json.dumps(one_img_data)
                save_file.write(jsObj + "\n")
            save_file.close()

    @rank_zero_only
    def on_loop_end(self, model, **kwargs):
        # if task_type_list = [], we do not generate adas_eval file
        if len(self.task_type_list) == 0:
            return
        # generate detection name, generate segmentation dir
        self.name_dir_list = _generate_name_dir_list(
            self.task_name_list, self.task_type_list, self.save_dir
        )

        # model forward and save results
        model.eval()
        with torch.no_grad():
            for index, (task_type, dataloader) in enumerate(
                zip(self.task_type_list, self.dataloaders)
            ):
                detName_or_segDir = self.name_dir_list[index]
                if task_type is None or detName_or_segDir is None:
                    continue
                # single detection result
                single_detection_res = []
                for _, batch in enumerate(dataloader):
                    output = model(batch)
                    self.save_part_results(
                        index, batch, output, single_detection_res
                    )  # noqa
                # save detection result to json
                self.save_detecion_to_json(
                    single_detection_res, detName_or_segDir
                )  # noqa
