# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import copy
import json
from glob import glob
from typing import Dict, Tuple

import mmcv
import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from ..common.data_classes import EvalBoxes
from ..detection.data_classes import DetectionBox
from ..detection.utils import category_to_detection_name
from ..tracking.data_classes import TrackingBox


def load_prediction(
    result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False
) -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert "results" in data, (
        "Error: No field `results` in result file. Please note that the result format changed."
        "See https://www.nuscenes.org/object-detection for more information."
    )

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data["results"], box_cls)
    meta = data["meta"]
    if verbose:
        print(
            "Loaded results from {}. Found detections for {} samples.".format(
                result_path, len(all_results.sample_tokens)
            )
        )

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, (
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample
        )

    return all_results, meta


def get_gt(info_paths):
    json_path_list = sorted(glob(info_paths + "/*.json", recursive=True))
    gt_infos = []
    for json_path in json_path_list:
        sample_list = mmcv.load(json_path)["sample_list"]
        for sample in sample_list:
            gt_infos.append(sample)
    return gt_infos


def load_gt(gt_infos, box_cls, verbose: bool = False) -> EvalBoxes:
    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    for sample_anno in tqdm.tqdm(gt_infos, leave=verbose):
        sample_boxes = []
        timestamp = str(sample_anno["timestamp"])
        sample_annotations = sample_anno["sample_annotation"]
        for sample_annotation in sample_annotations:
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation["type"])
                if detection_name is None:
                    continue
                sample_annotation_3d = sample_annotation["3D"]
                # if sample_annotation_3d['visible_in_2d'] == 0: # 过滤掉2d不可见的
                #     continue
                yaw_radians = np.array([sample_annotation_3d["yaw"]])
                import pyquaternion

                quaternion = pyquaternion.Quaternion(
                    axis=[0, 0, 1], radians=yaw_radians
                )
                whl = sample_annotation_3d["whl"]
                sample_boxes.append(
                    box_cls(
                        sample_token=detection_name,
                        translation=sample_annotation_3d["xyz"],
                        size=[whl[0], whl[2], whl[1]],  # size 定义 wlh
                        rotation=quaternion.elements.tolist(),
                        velocity=sample_annotation_3d["velocity"][:2],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name="",
                    )
                )
            else:
                raise NotImplementedError("Error: Invalid box_cls %s!" % box_cls)

        all_annotations.add_boxes(timestamp, sample_boxes)

    return all_annotations


def add_center_dist(gt_infos, eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        result = [d for d in gt_infos if str(d.get("timestamp")) == sample_token]
        if result == []:
            continue
        CAM_FRONT_translation = np.array(result[0]["CAM_FRONT"]["cam2car"])[:3, -1]
        CAM_BACK_translation = np.array(result[0]["CAM_BACK"]["cam2car"])[:3, -1]

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (
                box.translation[0],
                box.translation[1],
                box.translation[2],
            )

            cam_translation = (
                box.translation[0]
                - (CAM_FRONT_translation[0] + CAM_BACK_translation[0]) / 2,
                box.translation[1]
                - (CAM_FRONT_translation[1] + CAM_BACK_translation[1]) / 2,
                box.translation[2]
                - (CAM_FRONT_translation[2] + CAM_BACK_translation[2]) / 2,
            )

            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = ego_translation
                box.cam_translation = cam_translation
            else:
                raise NotImplementedError

    return eval_boxes


def filter_eval_boxes(
    eval_boxes: EvalBoxes,
    max_dist: Dict[str, float],
    ranges: list,
    verbose: bool = False,
) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])

        # eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
        # box.ego_dist < max_dist[box.__getattribute__(class_field)]]

        eval_boxes.boxes[sample_token] = [
            box
            for box in eval_boxes[sample_token]
            if abs(box.translation[1]) <= ranges[2]
            and (
                box.translation[0] <= ranges[0]
                if box.translation[0] > 0
                else abs(box.translation[0]) <= ranges[1]
            )
        ]

        # for box in eval_boxes[sample_token]:
        # print(max_dist[box.__getattribute__(class_field)])

        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [
            box for box in eval_boxes[sample_token] if not box.num_pts == 0
        ]
        point_filter += len(eval_boxes[sample_token])

        # # Perform bike-rack filtering.
        # sample_anns = nusc.get("sample", sample_token)["anns"]
        # bikerack_recs = [
        #     nusc.get("sample_annotation", ann) for ann in sample_anns
        #     if nusc.get("sample_annotation", ann)["category_name"] ==
        #     "static_object.bicycle_rack"
        # ]
        # bikerack_boxes = [
        #     Box(rec["translation"], rec["size"], Quaternion(rec["rotation"]))
        #     for rec in bikerack_recs
        # ]
        # filtered_boxes = []
        # for box in eval_boxes[sample_token]:
        #     if box.__getattribute__(class_field) in ["bicycle", "motorcycle"]:
        #         in_a_bikerack = False
        #         for bikerack_box in bikerack_boxes:
        #             if (np.sum(
        #                     points_in_box(
        #                         bikerack_box,
        #                         np.expand_dims(np.array(box.translation),
        #                                        axis=1),
        #                     )) > 0):
        #                 in_a_bikerack = True
        #         if not in_a_bikerack:
        #             filtered_boxes.append(box)
        #     else:
        #         filtered_boxes.append(box)

        # eval_boxes.boxes[sample_token] = filtered_boxes
        # bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        # print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = "detection_name"
    elif isinstance(box, TrackingBox):
        class_field = "tracking_name"
    else:
        raise Exception("Error: Invalid box type: %s" % box)

    return class_field


def range_eval_boxes(
    eval_boxes: EvalBoxes,
    # max_dist: Dict[str, float],
    ranges: list,
) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """

    eval_boxes = copy.deepcopy(eval_boxes)

    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        eval_boxes.boxes[sample_token] = [
            box
            for box in eval_boxes[sample_token]
            if np.linalg.norm(box.ego_translation) > ranges[0]
            and np.linalg.norm(box.ego_translation) <= ranges[1]
        ]

    return eval_boxes
