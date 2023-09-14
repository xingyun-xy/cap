# Copyright (c) Changan Auto. All rights reserved.

import copy
import glob
import json
import logging
import os
import pickle
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

try:
    import imutils

    IMUTILS_AVAILABLE = True
except ImportError:
    imutils = object
    IMUTILS_AVAILABLE = False
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from cap.metrics.metric_3dv_utils import (
    NpEncoder,
    bev3d_bbox_eval,
    calap,
    collect_data,
    draw_curves,
)
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from cap.utils.apply_func import convert_numpy as to_numpy
from cap.utils.distributed import get_dist_info, rank_zero_only
from cap.utils.logger import rank_zero_info
from cap.visualize.bev_3d import Bev3DVisualize
from .metric import EvalMetric

__all__ = ["AbsRel", "PoseRTE", "BEVDetEval", "BevSegInstanceEval"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class AbsRel(EvalMetric):
    """calculation multi range absrel for depth task.

    math.abs(pred-gt)/gt.

    Args:
        range_list (list) : A list contains multi range to calculation absrel.
        name (str): metric name.
    """

    def __init__(self, range_list=((0, 150)), name="absrel"):
        names = []
        for (low, high) in range_list:
            assert low < high, "range setting not valid"
            names.append("absrel(%s,%s)" % (str(low), str(high)))
        self.range_list = range_list
        super(AbsRel, self).__init__(names)

    def _init_states(self):
        self.add_state(
            "sum_metric",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )

    def get_absrel(self, pred, gt, low, high):
        valid_mask = (gt > low) * (gt < high)
        mask_zeros = (gt == 0) * (1e-6)
        diff = (pred - gt).abs()
        abs_rel = diff / (gt + mask_zeros)  # Not divide zero.
        abs_rel_valid = abs_rel * valid_mask
        abs_rel_valid_mean = abs_rel_valid.sum() / valid_mask.sum()
        return abs_rel_valid_mean

    def update(
        self,
        gt_depth: torch.Tensor,
        pred_depth: Union[Sequence[torch.Tensor], torch.Tensor],
    ):

        pred = _as_list(pred_depth)[0][:, 0:1]

        pred = F.interpolate(
            pred, [gt_depth.size(2), gt_depth.size(3)], mode="nearest"
        )

        for i, (low, high) in enumerate(self.range_list):
            abs_rel = self.get_absrel(pred, gt_depth, low, high)
            if not torch.any(torch.isnan(abs_rel)):
                self.sum_metric[i] += abs_rel
                self.num_inst[i] += 1


@OBJECT_REGISTRY.register
class PoseRTE(EvalMetric):
    """RTE metric for pose.

    Args:
        gt_pose_path (str): gt pose path.
        name (str): Name of this metric instance for display.
        save_traj_path (str): path to save trajectory result,
            None means not save.
        scale_coe (float): scale coefficient for pose. default is 0.01.
            donot recommend changing it.
    """

    def __init__(
        self,
        gt_pose_path,
        name="pose_rte",
        save_traj_path=None,
        scale_coe=0.01,
    ):

        self.gt_pose_path = gt_pose_path
        self.gt_pose = np.loadtxt(gt_pose_path)

        try:
            from changan_driving_dataset import PoseEvaluator, PoseTransformer

            self.pt = PoseTransformer()
            self.pe = PoseEvaluator(alignment="7dof")
        except ImportError:
            raise ModuleNotFoundError("changan_driving_dataset is required.")

        self.save_dir = None
        self.epoch = 1
        self.scale_coe = scale_coe

        if save_traj_path is not None:
            self.save_dir = os.path.join(
                save_traj_path, self.gt_pose_path.split("/")[-2]
            )
            if not os.path.exists(self.save_dir):
                try:
                    os.makedirs(self.save_dir)
                except Exception:
                    pass

        super(PoseRTE, self).__init__(name)

    def _init_states(self):
        self.add_state("timestamp", [], dist_reduce_fx="cap")
        self.add_state("pred_trans", [], dist_reduce_fx="cap")
        self.add_state("pred_axis", [], dist_reduce_fx="cap")

    def reset(self) -> None:
        self.pt.reset()
        super().reset()

    def update(
        self,
        timestamp: Union[Sequence[torch.Tensor], torch.Tensor],
        axisangle: Union[Sequence[torch.Tensor], torch.Tensor],
        translation: Union[Sequence[torch.Tensor], torch.Tensor],
    ) -> None:

        self.timestamp.append(timestamp)
        translation = _as_list(translation)[-1]
        self.pred_trans.append(translation.view(-1, 3))
        axisangle = _as_list(axisangle)[-1]
        self.pred_axis.append(axisangle.view(-1, 3))

    def compute(self):

        trans = (self.pred_trans * self.scale_coe).cpu().numpy()
        axis = (self.pred_axis * self.scale_coe).cpu().numpy()
        timestamp = self.timestamp.cpu().numpy()

        self.pt.from_relative_axis_angle(axis)
        self.pt.from_relative_translation(trans)
        self.pt.load_timestamp(timestamp)
        self.pt.sort_by_timestamps()
        tum_array = self.pt.dumparray(style="tum")

        if self.save_dir:
            np.savetxt(
                os.path.join(self.save_dir, "pred_pose_%d.txt" % self.epoch),
                tum_array,
            )
            self.epoch += 1

        if os.getenv("CAP_PIPELINE_TEST", "0") == "0":
            result_dict = self.pe.eval(self.gt_pose, tum_array)
            key = [
                "scale",
                "RTE",
                "RRE",
                "EulerRoll",
                "EulerPitch",
                "EulerYaw",
                "ATE",
                "RRE_m",
                "RRE_deg",
                "ITE",
                "IRE",
                "instant_roll",
                "instant_pitch",
                "instant_yaw",
            ]
            for k in key:
                rank_zero_info(k + "  :%.2f" % result_dict[k])
        else:
            rank_zero_info("skip PoseRTE metric in pipeline_test")
            result_dict = {"RTE": 0.0}
        return (self.name, result_dict["RTE"])


@OBJECT_REGISTRY.register_module
class BevSegInstanceEval(EvalMetric):
    """The bev seg instance eval metrics.

    Args:
        seg_class(list(str)): A list of classes the segmentation dataset
            includes, the order should be the same as the label.
        metrics (list(str)): A list of eval metrics
        vcs_origin_coord (list(str)) : The position of Car in vcs coordinate
        target_categorys(list(str)) : The output category to validation
        target_x_intervals (list(float)): X-coordinate range to validation
        target_y_intervals(list(float)): y-coordinate range to validation
        vcs_range(list(float)):vcs range.(order is (bottom,right,top,left))
        bev_seg_target_size (sequence):bev_seg_target_size,
                (order is (height,width))
        gt_min_x(float): Max depth for gts.
        gt_max_x(float): Max depth for gts.
        max_distance(float): maximum distance threshold.
        verbose(bool):  Whether to return verbose value for sda eval, default
            is False.
        name(str): Name of this metric instance for display
        enable_occlusion(bool):Whether to use occlusion. default False
    """

    def __init__(
        self,
        seg_class: List[str],
        metirc_key: List[str],
        vcs_origin_coord: List[int],
        target_categorys: Sequence[str],
        target_x_intervals: Sequence[float],
        target_y_intervals: Sequence[float],
        vcs_range: Sequence[float],
        bev_seg_target_size: Sequence[float],
        gt_min_x: float = -30.0,
        gt_max_x: float = 72.4,
        max_distance: float = 1.5,
        verbose: bool = False,
        enable_occlusion: bool = False,
        name: str = "BevSegInstanceEval",
    ):
        self.seg_class = seg_class
        self.verbose = verbose
        self.name = name
        self.vcs_origin_coord = vcs_origin_coord
        self.max_distance = max_distance
        self.metirc_key = metirc_key
        self.enable_occlusion = enable_occlusion
        self.target_categorys = target_categorys
        target_x_intervals = [gt_min_x] + list(target_x_intervals) + [gt_max_x]
        self.full_target_x_intervals = [
            "({},{})".format(start, end)
            for start, end in zip(
                target_x_intervals[:-1], target_x_intervals[1:]
            )
        ] + ["({},{})".format(gt_min_x, gt_max_x)]

        self.full_target_y_intervals = [
            "({},{})".format(start, end)
            for start, end in zip(
                [0 - item for item in target_y_intervals], target_y_intervals
            )
        ]
        scope_H = vcs_range[2] - vcs_range[0]
        scope_W = vcs_range[3] - vcs_range[1]
        bev_height, bev_width = bev_seg_target_size
        self.vcs2bev = np.array(
            [
                0.0,
                -float(bev_width) / scope_W,
                vcs_origin_coord[1],
                -float(bev_height) / scope_H,
                0.0,
                vcs_origin_coord[0],
                0,
                0,
                1,
            ],
            dtype=np.float,
        ).reshape((3, 3))
        if not IMUTILS_AVAILABLE:
            raise ImportError("Please install imutils")
        super(BevSegInstanceEval, self).__init__("BevSegInstanceEval")

    def collect_bev_seg_data(
        self,
        label: torch.Tensor,
        preds: Union[Sequence[torch.Tensor], torch.Tensor],
    ):
        if self.enable_occlusion:
            assert "occlusion" in label
        # If there is no occlusion in the label,
        # the occlusion mask will be all zeros
        occlusion = (
            label.pop("occlusion")
            if "occlusion" in label
            else torch.zeros_like(_as_list(preds)[0])
        )
        pred_label = _as_list(preds)[0].detach().cpu().numpy()
        occlusion_label = _as_list(occlusion)[0].detach().cpu().numpy()
        batch_size = pred_label.shape[0]
        output_label_list = []
        output_pred_list = []
        occlusion_label_list = []
        for bs in range(batch_size):
            output_pred_list.append(self.skeletonize_pred(pred_label[bs]))
            occlusion_label_list.append(occlusion_label[bs])
            label_dict = {}
            for category in self.target_categorys:
                _current_category_list = (
                    label[category][bs].detach().cpu().numpy().tolist()
                )
                label_dict[category] = _current_category_list
            output_label_list.append(label_dict)
        return output_pred_list, output_label_list, occlusion_label_list

    def split_intervals(self, intervals_str):
        str_list = intervals_str[1:-1].split(",")
        float_list = [float(_item) for _item in str_list]
        return float_list

    def skeletonize_pred(self, pred):
        output = np.zeros((512, 512), dtype=np.uint8)
        process_index = [
            self.seg_class.index(category)
            for category in self.target_categorys
        ]
        for skeletonize_index in process_index:
            pred_copy = copy.deepcopy(pred)
            pred_copy = pred_copy.astype("uint8")
            for i in range(len(self.seg_class)):
                if i == skeletonize_index:
                    continue
                pred_copy[pred_copy == i] = 0
            pred_copy[pred_copy == skeletonize_index] = 1
            pred_copy_skele = imutils.skeletonize(pred_copy, size=(3, 3))
            # pred[pred == skeletonize_index] = 0
            output[pred_copy_skele != 0] = skeletonize_index
        return output

    def convert_to_dict(self, x_origin, y_origin):
        def merge(values, thresh=0.2):
            values.sort()
            values = [round(_item, 2) for _item in values]
            values = np.array(values)
            res = []
            cur_res = [values[0]]
            for i in range(1, len(values)):
                if abs(values[i] - values[i - 1]) <= 0.21:
                    cur_res.append(values[i])
                else:
                    res.append(np.mean(cur_res))
                    cur_res = [values[i]]
            res.append(np.mean(cur_res))
            return res

        x_y_pred = [
            [
                (self.vcs_origin_coord[0] - y) * 0.2 - 0.1,
                (self.vcs_origin_coord[1] - x) * 0.2 - 0.1,
            ]
            for (x, y) in zip(list(x_origin), list(y_origin))
        ]
        x_y_pred_dict = {}
        for x_pred, y_pred in x_y_pred:
            x_pred = round(x_pred, 1)
            if x_pred in x_y_pred_dict.keys():
                x_y_pred_dict[x_pred].append(y_pred)
            else:
                x_y_pred_dict[x_pred] = []
                x_y_pred_dict[x_pred].append(y_pred)
        output_dict = {}
        for key in x_y_pred_dict.keys():
            y_pred_list = x_y_pred_dict[key]
            _new_key = str(round(key, 1))
            output_dict[_new_key] = merge(y_pred_list)
            # x_y_pred_dict[key]=merge(y_pred_list)
        return output_dict

    def update_pred_num(self, x_y_pred_dict, category):
        for target_x_interval in self.full_target_x_intervals:
            (
                target_x_begin,
                target_x_end,
            ) = self.split_intervals(target_x_interval)
            for x in x_y_pred_dict.keys():
                _str_x = float(x)
                if _str_x > target_x_begin and _str_x < target_x_end:
                    y_list = x_y_pred_dict[x]
                    for y in y_list:
                        for target_y_interval in self.full_target_y_intervals:
                            (
                                target_y_begin,
                                target_y_end,
                            ) = self.split_intervals(target_y_interval)
                            if y > target_y_begin and y < target_y_end:
                                prefix = "{}_{}_{}".format(
                                    target_y_interval,
                                    category,
                                    target_x_interval,
                                )
                                setattr(
                                    self,
                                    prefix + "_pred_num",
                                    getattr(self, prefix + "_pred_num", None)
                                    + 1,
                                )

    def in_occlusion(self, x, y, occlusion):
        """Whether x y in occlusion.

        Args:
            x,y: the vcs coordinate, (x,y).
            occlusion: occlusion mask
        """
        height, width = occlusion.shape
        pt = self.vcs2bev @ np.array([x, y, 1.0], dtype=np.float).reshape(
            (3, 1)
        )
        v = int(pt[0] + 0.5)
        u = int(pt[1] + 0.5)
        if u >= height or u < 0 or v >= width or v < 0:
            return False
        if occlusion[u, v] == 1:
            return True
        else:
            return False

    def update(
        self,
        label: torch.Tensor,
        preds: Union[Sequence[torch.Tensor], torch.Tensor],
    ):

        matched_num_list = {}
        # save matched distance
        for target_y_interval in self.full_target_y_intervals:
            current_target_y_interval = {}
            for category in self.target_categorys:
                current_category = {}
                for target_x_interval in self.full_target_x_intervals:
                    current_category[target_x_interval] = []
                current_target_y_interval[category] = current_category
            matched_num_list[target_y_interval] = current_target_y_interval

        (
            pred_list,
            label_list,
            occlusion_label_list,
        ) = self.collect_bev_seg_data(label, preds)

        for pred, label, occlusion in zip(
            pred_list, label_list, occlusion_label_list
        ):
            for category in self.target_categorys:
                pred = (
                    pred * (1 - occlusion) if self.enable_occlusion else pred
                )
                y_origin, x_origin = np.where(
                    pred == self.seg_class.index(category)
                )
                x_y_pred_dict = self.convert_to_dict(x_origin, y_origin)

                self.update_pred_num(x_y_pred_dict, category)

                if label[category] is not None:
                    for target_y_interval in self.full_target_y_intervals:
                        (
                            target_y_begin,
                            target_y_end,
                        ) = self.split_intervals(target_y_interval)
                        x_y_pred_dict_copy = copy.deepcopy(x_y_pred_dict)
                        for [x, y] in list(label[category]):
                            if x == 1000.0 and y == 1000.0:
                                break
                            if self.enable_occlusion and self.in_occlusion(
                                x, y, occlusion
                            ):
                                continue

                            if y <= target_y_begin or y >= target_y_end:
                                continue

                            # get gt num
                            for (
                                target_x_interval
                            ) in self.full_target_x_intervals:
                                (
                                    target_x_begin,
                                    target_x_end,
                                ) = self.split_intervals(target_x_interval)

                                if x > target_x_begin and x < target_x_end:
                                    prefix = "{}_{}_{}".format(
                                        target_y_interval,
                                        category,
                                        target_x_interval,
                                    )
                                    setattr(
                                        self,
                                        prefix + "_gt_num",
                                        getattr(self, prefix + "_gt_num", None)
                                        + 1,
                                    )

                            # matching stat
                            _distance = []
                            _str_key = str(round(x, 1))
                            if _str_key not in x_y_pred_dict_copy.keys():
                                continue
                            y_pred_list = x_y_pred_dict_copy[_str_key]
                            _distance = [
                                [abs(y_pred - y), y_pred]
                                for y_pred in y_pred_list
                                if abs(y_pred - y) < self.max_distance
                            ]
                            if len(_distance) == 0:
                                continue
                            _distance.sort()
                            x_y_pred_dict_copy[_str_key].remove(
                                _distance[0][1]
                            )
                            for (
                                target_x_interval
                            ) in self.full_target_x_intervals:
                                (
                                    target_x_begin,
                                    target_x_end,
                                ) = self.split_intervals(target_x_interval)
                                if x > target_x_begin and x < target_x_end:
                                    prefix = "{}_{}_{}".format(
                                        target_y_interval,
                                        category,
                                        target_x_interval,
                                    )
                                    matched_num_list[target_y_interval][
                                        category
                                    ][target_x_interval].append(
                                        _distance[0][0]
                                    )
        for target_y_interval in self.full_target_y_intervals:
            for category in self.target_categorys:
                for target_x_interval in self.full_target_x_intervals:
                    prefix = "{}_{}_{}".format(
                        target_y_interval,
                        category,
                        target_x_interval,
                    )
                    match_list = getattr(self, prefix + "_matched_num")
                    curren_list = matched_num_list[target_y_interval][
                        category
                    ][target_x_interval]
                    current_tensor_list = torch.tensor(
                        np.array(curren_list),
                        device=self._device_key.device,
                        dtype=torch.float16,
                    )

                    match_list.extend([current_tensor_list])
                    setattr(
                        self,
                        prefix + "_matched_num",
                        match_list,
                    )

    def get_per_metric(
        self, current_calculate_list, current_pred_num, current_gt_num
    ):
        current_metirc = {}
        _current_len = len(current_calculate_list)
        if _current_len == 0:
            _current_mean = 0.0
            _current_max = 0.0
            _current_Percentile_0_9967 = 0.0
            _current_Percentile_0_996 = 0.0
            _current_std = 0.0
            _current_expression_3std = 0.0
            _current_expression_0_075 = 0.0
            _recall = 0.0
            _precision = 0.0
        else:
            current_calculate_list.sort()
            _recall = _current_len / current_gt_num
            _precision = _current_len / current_pred_num
            _current_mean = np.mean(current_calculate_list)
            _current_max = max(current_calculate_list)
            _current_Percentile_0_9967 = current_calculate_list[
                int(0.9967 * _current_len)
            ]
            _current_Percentile_0_996 = current_calculate_list[
                int(0.96 * _current_len)
            ]
            _current_std = np.std(np.array(current_calculate_list), ddof=1)
            _current_expression_3std = (
                np.count_nonzero(
                    np.array(current_calculate_list) < 3 * _current_std
                )
                / _current_len
            )
            _current_expression_0_075 = (
                np.count_nonzero(np.array(current_calculate_list) < 0.075)
                / _current_len
            )

        current_metirc["Recall"] = round(_recall, 4)
        current_metirc["Precision"] = round(_precision, 4)
        current_metirc["Num"] = round(_current_len, 4)
        current_metirc["Mean"] = round(_current_mean, 4)
        current_metirc["Max"] = round(_current_max, 4)
        current_metirc["Percentile:0.9976"] = round(
            _current_Percentile_0_9967, 4
        )
        current_metirc["Percentile:0.96"] = round(_current_Percentile_0_996, 4)
        current_metirc["Expression:[VALUE<3*Std]"] = round(
            _current_expression_3std, 4
        )
        current_metirc["Expression:[VALUE<0.075]"] = round(
            _current_expression_0_075, 4
        )
        return current_metirc

    def get_metric_names(self):
        metric_names = []
        for target_y_interval in self.full_target_y_intervals:
            for category in self.target_categorys:
                for target_x_interval in self.full_target_x_intervals:
                    prefix = "{}_{}_{}".format(
                        target_y_interval, category, target_x_interval
                    )
                    metric_names += [prefix + "_gt_num"]
                    metric_names += [prefix + "_pred_num"]
                    metric_names += [prefix + "_matched_num"]
        metric_names += ["_device_key"]
        return metric_names

    def _init_states(self):
        for name in self.get_metric_names():
            if "matched" in name:
                self.add_state(
                    name,
                    default=[],
                    dist_reduce_fx="cap",
                )
            else:
                self.add_state(
                    name,
                    default=torch.tensor([0.0]),
                    dist_reduce_fx="sum",
                )

    def compute(self):
        """Get evaluation metrics."""

        result_metric = {}
        for target_y_interval in self.full_target_y_intervals:
            _category_metric = {}
            for category in self.target_categorys:
                _intervals_metric = {}
                for target_x_interval in self.full_target_x_intervals:
                    prefix = "{}_{}_{}".format(
                        target_y_interval, category, target_x_interval
                    )
                    current_calculate_list = getattr(
                        self, prefix + "_matched_num"
                    )
                    current_calculate_list = [
                        item.cpu().numpy().tolist()
                        for item in current_calculate_list
                    ]
                    # remove self.max_distance in current_calculate_list
                    current_calculate_array = np.array(current_calculate_list)
                    delete_index = np.where(
                        np.equal(current_calculate_array, self.max_distance)
                    )
                    current_calculate_list = np.delete(
                        current_calculate_array, delete_index
                    ).tolist()
                    current_pred_num = getattr(self, prefix + "_pred_num")
                    current_pred_num = (
                        current_pred_num.cpu().numpy().tolist()[0]
                    )

                    current_gt_num = getattr(self, prefix + "_gt_num")
                    current_gt_num = current_gt_num.cpu().numpy().tolist()[0]

                    current_metirc = self.get_per_metric(
                        current_calculate_list,
                        current_pred_num,
                        current_gt_num,
                    )
                    _intervals_metric.update(
                        {target_x_interval: current_metirc}
                    )
                _category_metric.update({category: _intervals_metric})
            result_metric.update({target_y_interval: _category_metric})
        if self.verbose:
            return result_metric

        summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
        dividing_line = "-" * 180 + "\n"
        line_format = "{:^8} {:^15} {:^20} {:^8} {:^8} {:^8} {:^8} {:^8} {:^20} {:^20} {:^20} {:^20} \n"  # noqa
        summary_str += line_format.format(
            "Range Y",
            "class",
            "Range X",
            "Recall",
            "Precision",
            "Num",
            "Mean",
            "Max",
            "Percentile:0.9976",
            "Percentile:0.96",
            "Expression:[VALUE<3*Std]",
            "Expression:[VALUE<0.075]",
        )
        for target_y_interval in self.full_target_y_intervals:
            for category in self.target_categorys:
                for target_x_interval in self.full_target_x_intervals:
                    _show_metric = []
                    _show_metric.append(target_y_interval)
                    _show_metric.append(category)
                    _show_metric.append(target_x_interval)
                    _current_metirc = result_metric[target_y_interval][
                        category
                    ][target_x_interval]
                    for metric_name in self.metirc_key:
                        assert metric_name in _current_metirc.keys()
                        _show_metric.append(_current_metirc[metric_name])

                    summary_str += line_format.format(*_show_metric)
                summary_str += dividing_line
        logger.info(summary_str)


@OBJECT_REGISTRY.register_module
class BEVDetEval(EvalMetric):
    """The BEV 3D detection eval metrics.

    The BEV 3D detection metric calculation is based on the real3d eval metric,
    for more detail please refer to Read3dEval (cap/metrics/real3d.py)

    Args:
        eval_category_ids (tuple of int): The categories need to be evaluation.
        score_threshold (float, defualt: 0.1): Threshold for score.
        iou_threshold (float, defualt: 0.2): Threshold for IoU.
        gt_max_depth (float, default: 100): Max depth for gts.
        save_path (str, default: None): Path to save bev3d results,
            format: pred.pkl.
        save_score_thr (float, defualt: 0.0): The score thr for saved pred box.
        save_real3d_res (bool, default: False): Whether save the eval results
            of real3d data for sda adas_eval. (TODO: Remove it when the real3d
            dataset not in training dataset, xiangyu.li)
        overrides_save_res (bool, default: False): Whether to override the
            pred.pkl in each metric eval interval, if True, only the latest
            pred.pkl will be save.
        save_vis_dir (str, default: None): Path to save bev3d visulize results.
        vis_setting (Dict, default: None): Settings of visualize.
        metrics (tuple of str, default: None): Tuple of eval metrics, using
            ("dxyp", "drot") if not special.
        depth_intervals (tuple of int, default: None): Depth range to
            validation, using (20, 50, 70) if not special.
        visibility_intervals (tuple of float, default: None): Piecewise
            interval, which used to evaluate in each visibility segment.
            When seting visibility_interval to (0,0.5,1), we will get
            evaluation results of different instance visibility intervals
            [0,0.5) and [0.5,1).
        enable_ignore (bool, default: True): Whether to use ignore_mask.
        prcurv_save_path (str, default: None): Path to save pr results
            and curves, NOTE: since the ap calculate should contains all
            pred's results, so we should create a dict to save the ap res.
        verbose (bool, default: True): Whether print the recall/precision
            etc of sub-distance.
    """

    def __init__(
        self,
        eval_category_ids: Sequence[int],
        score_threshold: float,
        iou_threshold: float,
        gt_max_depth: float,
        save_path: Optional[str] = None,
        save_score_thr: float = 0.0,
        save_real3d_res: bool = False,
        overrides_save_res: bool = False,
        save_vis_dir: Optional[str] = None,
        vis_setting: Optional[dict] = None,
        prcurv_save_path: Optional[dict] = None,
        metrics: Optional[Sequence[str]] = (
            "dx",
            "dxp",
            "dy",
            "dyp",
            "dxyp",
            "drot",
        ),
        depth_intervals: Optional[Sequence[str]] = (20, 50, 70),
        visibility_intervals: Optional[Sequence[float]] = None,
        enable_ignore: Optional[bool] = True,
        verbose: bool = True,
    ):
        self.eval_category_ids = eval_category_ids
        self.metrics = metrics
        self.gt_max_depth = gt_max_depth
        self.score_threshold = score_threshold
        self.depth_intervals = depth_intervals
        self.visib_intervals = visibility_intervals
        self.iou_threshold = iou_threshold
        self.save_path = save_path
        self.save_score_thr = save_score_thr
        self.overrides_save_res = overrides_save_res
        self.save_vis_dir = save_vis_dir
        self.verbose = verbose
        self.prcurv_save_path = prcurv_save_path

        if save_vis_dir:
            self.bev3d_vis = Bev3DVisualize(
                save_path=save_vis_dir, **vis_setting
            )

        if save_real3d_res:
            assert (
                save_path
            ), "save_real3d_res cannot be True if save_path is None"
        self.save_real3d_res = save_real3d_res

        assert (
            self.gt_max_depth > depth_intervals[-1]
        ), "gt_max_depth must be greater than depth_intervals[-1]"
        self.eps = 1e-9

        self.metric_index = 0
        self.pkl_save_path = None
        self.rank_pkl_save_path = None
        # json_save_path used for save AP results
        self.json_save_path = None
        self.rank_json_save_path = None
        self.enable_ignore = enable_ignore
        depth_intervals = [0] + list(depth_intervals) + [self.gt_max_depth]
        self.all_depth_intervals = [
            "({},{})".format(start, end)
            for start, end in zip(depth_intervals[:-1], depth_intervals[1:])
        ]
        # Use overall evaluation by default, all intervals to be evaluated
        # are contained in all_vis_intervals
        self.all_vis_intervals = {"(0,1)"}
        if self.visib_intervals is not None:
            for start, end in zip(
                self.visib_intervals[:-1], self.visib_intervals[1:]
            ):
                self.all_vis_intervals.add("({},{})".format(start, end))

        gt_cids = []
        det_cids = []
        for cid in eval_category_ids:
            gt_cids.append(cid)
            det_cids.append(cid)
        self.gt_cids = list(set(gt_cids))
        self.det_cids = list(set(det_cids))

        if self.verbose:
            self.pr_metric = ("TP", "FP", "FN", "Recall", "Precision")
            self.vis_metric = ("TP", "FN", "Recall")
        else:
            self.pr_metric = None
            self.vis_metric = None
        if self.prcurv_save_path:
            self.ap_metric = ("AP_Dist",)
        else:
            self.ap_metric = None
        super(BEVDetEval, self).__init__("BEVDetEval")

    def get_metric_names(self):
        metric_names = []
        for cate_id in self.eval_category_ids:
            for vis_interval in self.all_vis_intervals:
                prefix = "category_{}_{}_".format(cate_id, vis_interval)
                metric_names += [prefix + "seg_tp"]
                metric_names += [prefix + "seg_fn"]
                metric_names += [prefix + "seg_fp"]
                metric_names += [prefix + "num_valid_image"]
                metric_names += [prefix + "num_total_image"]
                for metric_name in self.metrics:
                    for dep_interval in self.all_depth_intervals:
                        metric_names += [
                            prefix + dep_interval + "_" + metric_name
                        ]
        return metric_names

    def _init_states(self):
        for name in self.get_metric_names():
            if "seg_" in name:
                self.add_state(
                    name,
                    default=torch.zeros((len(self.all_depth_intervals),)),
                    dist_reduce_fx="sum",
                )
            else:
                self.add_state(
                    name,
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.add_state(
                    name + "_num",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )

    def reset(self) -> None:
        super().reset()
        if self.prcurv_save_path:
            rank, word_size = get_dist_info()
            self.rank_json_save_path = os.path.join(
                self.prcurv_save_path,
                f"Eval_{str(self.metric_index)}",
                f"det_aps/rank_{rank}.json",
            )
            self.json_save_path = os.path.join(
                self.prcurv_save_path,
                f"Eval_{str(self.metric_index)}",
                f"pred_ap_{rank}.json",
            )
            json_save_root = os.path.dirname(self.rank_json_save_path)
            if not os.path.exists(json_save_root):
                os.makedirs(json_save_root, exist_ok=True)
        if self.save_path:
            # each rank's det results.
            self.rank_pkl_save_path = os.path.join(
                self.save_path,
                f"Eval_{str(self.metric_index)}",
                f"dets/rank_{rank}.pkl",
            )
            # all det's results.
            self.pkl_save_path = os.path.join(
                self.save_path,
                f"Eval_{str(self.metric_index)}",
                f"pred_rank_{rank}.pkl",
            )
            if self.overrides_save_res:
                self.rank_pkl_save_path = self.rank_pkl_save_path.replace(
                    f"Eval_{str(self.metric_index)}", "Eval_0"
                )
                self.pkl_save_path = self.pkl_save_path.replace(
                    f"Eval_{str(self.metric_index)}", "Eval_0"
                )
            pkl_save_root = os.path.dirname(self.rank_pkl_save_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)
        if self.save_vis_dir:
            os.makedirs(self.save_vis_dir, exist_ok=True)

    @rank_zero_only
    def reduce_rank_pkl(
        self,
    ):
        """Reduce all rank's pkl to total pred pkl."""

        pred_res = {}
        ordered_res = OrderedDict()
        rank_pkl_save_root = os.path.dirname(self.rank_pkl_save_path)
        pkl_file_list = glob.glob(rank_pkl_save_root + "/*.pkl")

        for pkl_file in pkl_file_list:
            rank_pred_res = self.load_pkl_file(pkl_file)
            pred_res.update(rank_pred_res)

        for key, value in sorted(pred_res.items(), key=lambda t: t[0]):
            ordered_res[key] = value

        logger.info("remove all rank's pkl...")
        os.system(f"rm {rank_pkl_save_root}/*.pkl")

        with open(self.pkl_save_path, "wb") as f:
            logger.info(f"dump pred.pkl to {self.pkl_save_path}")
            pickle.dump(ordered_res, f)

    @rank_zero_only
    def reduce_ap_resluts(
        self,
    ):
        """Reduce all rank's ap results to total ap results.

        and cal ap metrics and draw p-r curves.

        """

        all_pred_aps = []
        rank_json_save_root = os.path.dirname(self.rank_json_save_path)
        json_file_list = glob.glob(rank_json_save_root + "/*.json")

        for json_file in json_file_list:
            with open(json_file, "r") as fp:
                rank_pred_aps = fp.read().splitlines()
            for _pred in rank_pred_aps:
                _aps = json.loads(_pred)
                all_pred_aps.append(_aps)

        logger.info("remove all rank's json...")
        os.system(f"rm {rank_json_save_root}/*.json")

        num_gt = defaultdict(int)
        det_tp_mask, all_scores, det_tp_pred_loc, det_gt_mask = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        for res in all_pred_aps:
            for _cid, _cid_ap_res in res.items():
                for _ap_res in _cid_ap_res:
                    det_tp_mask[int(_cid)] += _ap_res["det_tp_mask"]
                    all_scores[int(_cid)] += _ap_res["all_scores"]
                    det_tp_pred_loc[int(_cid)] += _ap_res["det_tp_pred_loc"]
                    det_gt_mask[int(_cid)] += _ap_res["det_gt_mask"]
                    num_gt[int(_cid)] += _ap_res["num_gt"]

        ap_results = {}
        ap_depth_results = {
            cid: [0.0 for _ in range(len(self.depth_intervals) + 1)]
            for cid in self.eval_category_ids
        }
        ap_curves = defaultdict(dict)
        for cid in self.eval_category_ids:
            _det_tp_pred_loc = np.array(det_tp_pred_loc[cid])
            _det_tp_mask = np.array(det_tp_mask[cid])
            _all_scores = np.array(all_scores[cid])
            _det_gt_mask = np.array(det_gt_mask[cid])

            if len(_det_tp_pred_loc) == 0 or len(_det_gt_mask) == 0:
                for dep_ind in range(len(self.depth_intervals) + 1):
                    ap_depth_results[cid][dep_ind] = 0.0
            else:
                det_dep_thresh_inds = np.sum(
                    abs(_det_tp_pred_loc[:, 0:1]) > self.depth_intervals,
                    axis=-1,
                )
                dep_inds, cnts = np.unique(
                    det_dep_thresh_inds, return_counts=True
                )

                gt_count = np.zeros(len(self.depth_intervals) + 1)
                gt_dep_thresh_inds = np.sum(
                    abs(_det_gt_mask[:, 0:1]) > self.depth_intervals, axis=-1
                )
                gt_dep_id, gt_cnts = np.unique(
                    gt_dep_thresh_inds, return_counts=True
                )
                gt_count[gt_dep_id] += gt_cnts

                # cal ap in different detth threshold.
                for dep_ind in dep_inds:
                    mask = det_dep_thresh_inds == dep_ind
                    dep_tp_depth = _det_tp_mask[mask]
                    all_score_depth = _all_scores[mask]
                    arginds_dep = np.argsort(-all_score_depth)
                    all_score_depth = all_score_depth[arginds_dep]
                    dep_tp_depth = dep_tp_depth[arginds_dep]
                    det_fp_mask_depth = 1 - dep_tp_depth
                    det_tp = np.cumsum(dep_tp_depth)
                    det_fp = np.cumsum(det_fp_mask_depth)
                    cus_recall = det_tp / (gt_count[dep_ind] + 1e-6)
                    cus_precision = det_tp / (det_tp + det_fp + 1e-6)
                    ap_depth = calap(cus_recall, cus_precision)
                    # NOTE: Since the TP match in all depth range first and
                    # then split according to depth range, the TP matching
                    # will cross different depth, e.g. TP: vcs_x=19.9 match
                    # GT: vcs_x=20.1, which will cause num_tp > num_gt in
                    # range(0-20), so we clip ap > 1 in such case to 1.
                    if ap_depth > 1.0:
                        ap_depth = 1.0
                    ap_depth_results[cid][dep_ind] = ap_depth

            # cal all result's ap and save results.
            arginds = np.argsort(-_all_scores)
            _all_scores = _all_scores[arginds]
            _det_tp_mask = _det_tp_mask[arginds]
            det_fp_mask = 1 - _det_tp_mask
            det_tp = np.cumsum(_det_tp_mask)
            det_fp = np.cumsum(det_fp_mask)
            cus_recall = det_tp / (num_gt[cid] + 1e-6)
            cus_precision = det_tp / (det_tp + det_fp + 1e-6)
            ap = calap(cus_recall, cus_precision)
            ap_results[cid] = ap

            # save pr results for pr-curves
            ap_curves[cid]["recall"] = cus_recall
            ap_curves[cid]["precision"] = cus_precision
            ap_curves[cid]["conf"] = _all_scores
            ap_curves[cid]["num_gt"] = num_gt[cid] + 1e-6

            with open(self.json_save_path, "w") as f:
                json.dump(ap_curves, f, cls=NpEncoder)

            # draw_pr_curves
            draw_curves(
                [self.json_save_path],
                ["all"],
                os.path.join(
                    os.path.dirname(self.json_save_path),
                    f"pr_curv_eval_{self.metric_index-1}.png",
                ),
            )
        return ap_results, ap_depth_results

    def get(self):
        name, values = self.compute()
        if self.save_path:
            self.reduce_rank_pkl()
        return name, values

    def update(self, batch, output):

        _gt_group_by_cid, _det_group_by_cid, _timestamps = collect_data(
            batch, output, self.gt_cids, self.det_cids, self.save_real3d_res
        )
        res_aps = defaultdict(list)
        for cid in self.eval_category_ids:
            gt = {
                "timestamps": _timestamps,
                "annotations": _gt_group_by_cid[cid],
            }
            det = _det_group_by_cid[cid]
            res = bev3d_bbox_eval(
                det,
                gt,
                self.depth_intervals,
                self.score_threshold,
                self.iou_threshold,
                self.gt_max_depth,
                self.enable_ignore,
                self.all_vis_intervals,
            )
            for vidx, interval in enumerate(self.all_vis_intervals):
                prefix = "category_{}_{}_".format(cid, interval)
                device = getattr(self, prefix + "seg_tp", None).device
                seg_tp = torch.tensor(
                    res["counts"]["gt_matched"][vidx], device=device
                )
                seg_fn = torch.tensor(
                    res["counts"]["gt_missed"][vidx], device=device
                )
                seg_fp = torch.tensor(
                    res["counts"]["redundant_det"], device=device
                )
                num_valid_image = res["counts"]["timestamp_count"]
                num_total_image = res["counts"]["total_timestamp_count"]
                setattr(
                    self,
                    prefix + "seg_tp",
                    getattr(self, prefix + "seg_tp", None) + seg_tp,
                )
                setattr(
                    self,
                    prefix + "seg_fn",
                    getattr(self, prefix + "seg_fn", None) + seg_fn,
                )
                setattr(
                    self,
                    prefix + "seg_fp",
                    getattr(self, prefix + "seg_fp", None) + seg_fp,
                )
                setattr(
                    self,
                    prefix + "num_valid_image",
                    getattr(self, prefix + "num_valid_image", None)
                    + num_valid_image,
                )
                setattr(
                    self,
                    prefix + "num_total_image",
                    getattr(self, prefix + "num_total_image", None)
                    + num_total_image,
                )

                for metric_name in self.metrics:
                    for dep_interval, metric in zip(
                        self.all_depth_intervals, res[metric_name][interval]
                    ):
                        name = prefix + dep_interval + "_" + metric_name
                        setattr(
                            self,
                            name,
                            getattr(self, name, None) + np.sum(metric),
                        )
                        setattr(
                            self,
                            name + "_num",
                            getattr(self, name + "_num", None) + len(metric),
                        )
                # ap results
                cid_aps = res["counts"]["result_aps"]
                res_aps[cid].append(cid_aps)

        if self.save_path:
            self.save_results(batch, output)

        if self.prcurv_save_path:
            with open(self.rank_json_save_path, "a") as f:
                write_item = json.dumps(res_aps)
                f.write(write_item + "\n")

        # save BEV3D visualize results
        if self.save_vis_dir:
            self.bev3d_vis.save_imgs(output, batch)

    def save_results(self, batch, output):
        pkl_res = self.convert_to_save_format(batch, output)
        with open(self.rank_pkl_save_path, "ab") as f:
            pickle.dump(pkl_res, f)

    def convert_to_save_format(self, batch, output):
        save_res = defaultdict(list)
        assert "timestamp" in batch
        batch_timestamps = np.array(batch["timestamp"].cpu())
        if self.save_real3d_res:
            batch_timestamps = [
                str(int(_bs_time)) for _bs_time in batch_timestamps
            ]
            rec_date = batch["pack_dir"]
            batch_timestamps = [
                date + "__" + time
                for date, time in zip(rec_date, batch_timestamps)
            ]
        else:
            batch_timestamps = [
                str(int(_bs_time * 1000)) for _bs_time in batch_timestamps
            ]
        batch_size, num_objs = output["bev3d_ct"].shape[:2]
        # cap bev3d_ct (x,y) with bev3d_loc_z (z) -> vcs_location (x, y, z)
        location = torch.cat(
            (output["bev3d_ct"], output["bev3d_loc_z"].unsqueeze(-1)),
            dim=-1,
        )
        for i in range(batch_size):
            front_img_timestamp = batch_timestamps[i]
            if self.save_real3d_res:
                key = batch_timestamps[i]
            else:
                pack_dir = batch["pack_dir"][i]
                # the default timestamp in auto3dv is camera_front
                key = os.path.join(pack_dir, front_img_timestamp)

            for j in range(num_objs):
                # filter the padded objs in data transform
                if output["bev3d_score"][i][j] > self.save_score_thr:
                    save_res[key].append(
                        # all the key_names are used to adap to adas_eval
                        {
                            "dimensions": to_numpy(
                                output["bev3d_dim"][i][j]
                            ).tolist(),
                            "class_id": to_numpy(
                                output["bev3d_cls_id"][i][j], dtype="int16"
                            ),
                            "score": to_numpy(output["bev3d_score"][i][j]),
                            "yaw": to_numpy(output["bev3d_rot"][i][j]),
                            "location": to_numpy(location[i][j]).tolist(),
                            "timestamp": str(front_img_timestamp),
                        }
                    )
            if len(save_res[key]) < 1:
                save_res[key] = []
        return save_res

    def load_pkl_file(self, pkl_path):
        result = {}
        f = open(pkl_path, "rb")
        while True:
            try:
                data = pickle.load(f)
                result.update(data)
            except EOFError:
                break
        f.close()
        return result

    def compute(self):
        self.metric_index += 1
        # reduce all rank's ap results.
        if self.prcurv_save_path and dist.get_rank() == 0:
            all_ap_results, ap_depth_results = self.reduce_ap_resluts()

        for cid in self.eval_category_ids:
            for interval in self.all_vis_intervals:
                names, values = [], []
                prefix = "category_{}_{}_".format(cid, interval)
                num_valid_image = int(
                    getattr(self, prefix + "num_valid_image", None)
                    .cpu()
                    .numpy()
                )
                num_total_image = int(
                    getattr(self, prefix + "num_total_image", None)
                    .cpu()
                    .numpy()
                )
                # porcess seg precision-recall
                pr_results = {}
                seg_recall = []
                seg_precision = []
                seg_tp = getattr(self, prefix + "seg_tp", None).cpu().numpy()
                seg_fn = getattr(self, prefix + "seg_fn", None).cpu().numpy()
                seg_fp = getattr(self, prefix + "seg_fp", None).cpu().numpy()

                for i in range(len(seg_tp)):
                    tmp_precision = seg_tp[i] / (
                        seg_tp[i] + seg_fp[i] + self.eps
                    )
                    tmp_recall = seg_tp[i] / (seg_tp[i] + seg_fn[i] + self.eps)
                    seg_recall.append(tmp_recall)
                    seg_precision.append(tmp_precision)

                pr_results["TP"] = seg_tp
                pr_results["FP"] = seg_fp
                pr_results["FN"] = seg_fn
                pr_results["Recall"] = seg_recall
                pr_results["Precision"] = seg_precision
                tp = seg_tp.sum()
                fn = seg_fn.sum()
                fp = seg_fp.sum()
                precision = tp / (tp + fp + self.eps)
                recall = tp / (tp + fn + self.eps)
                logger_show_metric = self.metrics

                if self.verbose:
                    for metric_name in self.pr_metric:
                        for dep_interval, metric in zip(
                            self.all_depth_intervals, pr_results[metric_name]
                        ):
                            names += [
                                prefix + dep_interval + "_" + metric_name
                            ]
                            values += [metric]
                    if self.prcurv_save_path and dist.get_rank() == 0:
                        for ap_metric_name in self.ap_metric:
                            for dep_interval, metric in zip(
                                self.all_depth_intervals,
                                ap_depth_results[cid],
                            ):
                                names += [
                                    prefix
                                    + dep_interval
                                    + "_"
                                    + ap_metric_name
                                ]
                                values += [metric]

                for metric in self.metrics:
                    for dep_interval in self.all_depth_intervals:
                        metric_name = prefix + dep_interval + "_" + metric
                        val = getattr(self, metric_name, None).cpu().numpy()
                        num = int(
                            getattr(self, metric_name + "_num", None)
                            .cpu()
                            .numpy()
                        )
                        names += [metric_name]
                        values += [val / (num + self.eps)]

                summary_str = "~~~~ BEV3D class %s Summary metrics ~~~~\n" % (
                    str(str(cid))
                )
                summary_str += "Summary:\n"
                summary_str += "BEV_3D Overview: \n"
                nl = "\n"
                # Only evaluation in whole interval supports PR metric.
                # Otherwise we can only settle for using TP metric.
                if interval == "(0,1)":
                    summary_str += f" total_images : {num_total_image} \
                                {nl} valid_images : {num_valid_image} \
                                {nl} TP: {tp} {nl} FP: {fp} {nl} FN: {fn} \
                                {nl} Recall: {recall: .5f} \
                                {nl} Precision: {precision: .5f} \
                                {nl}"
                    if self.prcurv_save_path and dist.get_rank() == 0:
                        summary_str += f" AP: {all_ap_results[cid]: .4f} \
                                        {nl}"
                    logger_show_metric += self.pr_metric
                else:
                    summary_str += f" visibility : {interval} \
                                {nl} total_images : {num_total_image} \
                                {nl} valid_images : {num_valid_image} \
                                {nl} TP: {tp} {nl} FN: {fn} \
                                {nl} Recall: {recall: .5f} \
                                {nl}"
                    logger_show_metric += self.vis_metric

                for _metric in logger_show_metric:
                    summary_str += _metric.ljust(10)
                    value_str = "".ljust(10)
                    for _dep_interval in self.all_depth_intervals:
                        _metric_name = prefix + _dep_interval + "_" + _metric
                        if _metric_name in names:
                            name_idx = names.index(_metric_name)
                            summary_str += _dep_interval.ljust(15)
                            format = (
                                ".3f"
                                if _metric not in ["TP", "FP", "FN"]
                                else ".1f"
                            )
                            value_str += f"{values[name_idx]: {format}}".ljust(
                                15
                            )
                    summary_str += "\n"
                    summary_str += value_str
                    summary_str += "\n"

                if (
                    interval == "(0,1)"
                    and self.prcurv_save_path
                    and dist.get_rank() == 0
                ):
                    for _metric in self.ap_metric:
                        summary_str += _metric.ljust(10)
                        value_str = "".ljust(10)
                        for _dep_interval in self.all_depth_intervals:
                            _metric_name = (
                                prefix + _dep_interval + "_" + _metric
                            )
                            if _metric_name in names:
                                name_idx = names.index(_metric_name)
                                summary_str += _dep_interval.ljust(15)
                                format = (
                                    ".3f"
                                    if _metric not in ["TP", "FP", "FN"]
                                    else ".1f"
                                )
                                value_str += (
                                    f"{values[name_idx]: {format}}".ljust(15)
                                )
                        summary_str += "\n"
                        summary_str += value_str
                        summary_str += "\n"
                logger.info(summary_str)
        return ["recall", "precision"], [recall, precision]
