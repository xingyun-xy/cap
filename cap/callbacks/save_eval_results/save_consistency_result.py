# Copyright (c) Changan Auto. All rights reserved.
import json
import logging
import os
from dataclasses import fields
from typing import Dict, List, Optional

import numpy as np

from cap.core.data_struct.base import BaseDataList
from cap.core.data_struct.base_struct import (
    ClsLabel,
    DetBox2D,
    DetBox3D,
    Line2D,
    Mask,
    MultipleBox2D,
    Point2D_2,
)
from cap.registry import OBJECT_REGISTRY
from .save_eval_result import SaveEvalResult

__all__ = ["SaveDetConsistencyResult"]

logger = logging.getLogger(__name__)


def dump_bbox(det_box: DetBox2D):
    box = det_box.box.numpy().astype(float)
    return {
        "conf": det_box.score.item() * 10,
        "x1": box[0],
        "x2": box[2],
        "y1": box[1],
        "y2": box[3],
    }


def dump_kps_2(kps: Point2D_2):
    p0 = kps.point0.point.numpy().astype(float)
    p1 = kps.point1.point.numpy().astype(float)
    return [
        {
            "conf": kps.point0.score.item() * 10,
            "type": 0,
            "x1": p0[0],
            "y1": p0[1],
        },
        {
            "conf": kps.point1.score.item() * 10,
            "type": 1,
            "x1": p1[0],
            "y1": p1[1],
        },
    ]


def dump_multi_box(multi_box: MultipleBox2D):
    boxes = multi_box.boxes.numpy().astype(float)

    return [
        {
            "conf": score.item() * 10,
            "x1": box[0],
            "x2": box[2],
            "y1": box[1],
            "y2": box[3],
        }
        for box, score in zip(boxes, multi_box.scores)
    ]


def dump_line(line: Line2D):
    res = dump_kps_2(line)
    res[0]["type"] = 6
    res[1]["type"] = 7

    return res


def dump_cls_label(cls_label: ClsLabel):
    return [
        {
            "conf": cls_label.score.item() * 10,
            "property_id": cls_label.cls_idx.item(),
            "property_name": cls_label.cls_name,
        }
    ]


def dump_bbox_3d(box_3d: DetBox3D):
    return {
        "width": box_3d.w.item(),
        "height": box_3d.h.item(),
        "length": box_3d.l.item(),
        "x": box_3d.x.item(),
        "y": box_3d.y.item(),
        "z": box_3d.z.item(),
        "yaw": box_3d.yaw.item(),
    }


DUMP_FUNC_MAPS = {
    ClsLabel: dump_cls_label,
    DetBox2D: dump_bbox,
    MultipleBox2D: dump_multi_box,
    Point2D_2: dump_kps_2,
    Line2D: dump_line,
    DetBox3D: dump_bbox_3d,
}

DUMP_KEY_MAPS = {
    DetBox2D: "box2d",
    ClsLabel: "category",
    DetBox3D: "box3d",
}


@OBJECT_REGISTRY.register
class SaveDetConsistencyResult(SaveEvalResult):
    def __init__(
        self,
        output_dir: str,
        task_type_mapping: Dict,
        task_name_mapping: Dict,
        dump_extra_key: Optional[str] = None,
    ):
        super().__init__(output_dir)
        self.task_type_mapping = task_type_mapping
        self.task_name_mapping = task_name_mapping
        self.dump_extra_key = dump_extra_key

    def save_result(self, batch, result):
        res_dict = {}
        for i, img_id in enumerate(batch["img_id"]):
            for k, name in self.task_type_mapping.items():
                data = result[k][i].to("cpu")
                data = data.to("cpu")
                if isinstance(data, BaseDataList):
                    res_dict[name] = []
                    for item in iter(data):
                        dd = {}
                        for f in fields(item):
                            attr = getattr(item, f.name)
                            dump_func = DUMP_FUNC_MAPS.get(type(attr), None)
                            if dump_func is not None:
                                assert (
                                    type(attr) in DUMP_KEY_MAPS
                                    or f.name in self.task_name_mapping
                                )
                                dump_key = self.task_name_mapping.get(
                                    f.name, None
                                )
                                if dump_key is None:
                                    dump_key = DUMP_KEY_MAPS[type(attr)]

                                if dump_key in dd:
                                    dd[dump_key] += dump_func(attr)
                                else:
                                    dd[dump_key] = dump_func(attr)

                        res_dict[name].append(dd)

            if self.dump_extra_key is not None:
                res_dict = {self.dump_extra_key: res_dict}

            res_dict = {img_id: res_dict}

            with open(
                os.path.join(self.output_dir, f"{img_id}.json"), "w"
            ) as wf:
                json.dump(res_dict, wf, indent=2)

    def on_batch_end(self, batch, model_outs, **kwargs):
        if self.match_task_output:
            batch_output, match_output = self.match_task_output(
                batch, model_outs
            )
        else:
            batch_output, match_output = batch, model_outs

        self.save_result(batch_output, match_output)

    def __repr__(self):
        return "SaveDetConsistencyResult"


@OBJECT_REGISTRY.register
class SaveSegConsistencyResult(SaveEvalResult):
    def __init__(
        self,
        output_dir: str,
        task_list: Optional[List[str]] = None,
        dump_extra_key: Optional[str] = None,
    ):
        super().__init__(output_dir)
        self.task_list = task_list
        self.dump_extra_key = dump_extra_key

    def save_result(self, batch, result):

        for i, img_id in enumerate(batch["img_id"]):
            data = (
                {k: result[k][i] for k in self.task_list if k in result}
                if self.task_list is not None
                else {None: result[i]}
            )

            for k, task_data in data.items():
                assert isinstance(task_data, Mask)
                task_data = task_data.to("cpu")

                output_names = []
                if self.dump_extra_key is not None:
                    output_names.append(self.dump_extra_key)
                if k is not None:
                    output_names.append(k)
                output_names.append(img_id)

                out_name = "_".join(output_names) + ".bin"

                with open(os.path.join(self.output_dir, out_name), "wb") as wf:
                    task_data.mask.numpy().astype(np.uint8).tofile(wf)

    def on_batch_end(self, batch, model_outs, **kwargs):
        if self.match_task_output:
            batch_output, match_output = self.match_task_output(
                batch, model_outs
            )
        else:
            batch_output, match_output = batch, model_outs

        self.save_result(batch_output, match_output)

    def __repr__(self):
        return "SaveSegConsistencyResult"
