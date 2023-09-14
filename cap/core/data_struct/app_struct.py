import os
from dataclasses import dataclass, make_dataclass
from typing import ClassVar

from cap.utils.apply_func import _as_list
from .base import BaseData, BaseDataList


def build_task_struct(
        singleton_name,
        list_name,
        fields_desc,
        bases=(BaseData, BaseDataList),
):
    assert len(bases) == 2
    singleton_bases = _as_list(bases[0])
    list_bases = _as_list(bases[1])

    singleton_cls = make_dataclass(
        singleton_name,
        [(fd[0], fd[-1].singleton_cls) for fd in fields_desc],
        bases=tuple(singleton_bases),
    )

    list_cls = make_dataclass(
        list_name,
        [(fd[-2], fd[-1]) for fd in fields_desc] +
        [("singleton_cls", ClassVar[singleton_cls], singleton_cls)],
        bases=tuple(list_bases),
    )

    return singleton_cls, list_cls


@dataclass
class DetObject(BaseData):
    pass


@dataclass
class DetObjects(BaseDataList):
    singleton_cls: ClassVar[BaseData] = DetObject


def reformat_det_to_cap_eval(
    batch_data,
    batch_outputs,
    obj_key,
    det_task_key,
    extra_task_key=None,
    dump_obj_key=None,
    dump_extra_obj_key=None,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1],
            str), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"

        batch_data, batch_obj_key = batch_data
    batch_objects = batch_outputs[obj_key]
    out_key = dump_obj_key if dump_obj_key is not None else obj_key
    rets = []
    image_width = batch_data["img_width"]
    image_height = batch_data["img_height"]
    calib = batch_data["calib"]
    image_id = batch_data["img_id"]
    distCoeffs = batch_data["distCoeffs"]
    annotations = batch_data["annotations"]
    for img_name, objects in zip(batch_data["img_name"], batch_objects):
        objects = objects.to("cpu")
        ret = {
            "image_key":
            img_name,
            "image_name":
            img_name,
            "image_width":
            image_width,
            "image_height":
            image_height,
            "calib":
            calib,
            "id":
            image_id,
            "distCoeffs":
            distCoeffs,
            "annotations":
            annotations,
            out_key: [
                obj.to_sda_eval()
                for obj in iter(getattr(objects, det_task_key))
            ],
        }

        if extra_task_key is not None:
            extra_update_res = []
            for i, extra_res in enumerate(
                    iter(getattr(objects, extra_task_key))):
                extra_dump_res = extra_res.to_sda_eval()

                if isinstance(extra_dump_res, list):
                    # hotfix for weird results of bin det task
                    for res in extra_update_res:
                        res["parent_box"] = ret[out_key][i]["bbox"]
                    extra_update_res += extra_dump_res
                else:
                    ret[out_key][i].update(extra_dump_res)

                    if "attrs" in extra_dump_res:
                        if isinstance(dump_extra_obj_key, str):
                            dump_extra_obj_key = [dump_extra_obj_key]
                        extra_out_keys = (dump_extra_obj_key
                                          if dump_extra_obj_key is not None
                                          else [])
                        if len(extra_out_keys) == 0:
                            continue
                        # replace 2pe out key
                        attrs_key = list(ret[out_key][i]["attrs"].keys())
                        assert len(attrs_key) == len(extra_out_keys)
                        new_attrs = {}
                        for key, extra_out_key in zip(attrs_key,
                                                      extra_out_keys):
                            new_attrs[extra_out_key] = ret[out_key][i][
                                "attrs"][key]
                        ret[out_key][i]["attrs"] = new_attrs
                    else:
                        # deal with 3d
                        ret[out_key][i]["score"] *= ret[out_key][i].pop(
                            "bbox_score")
                        ret[out_key][i]["bbox_2d"] = ret[out_key][i].pop(
                            "bbox")

            if extra_update_res:
                ret[out_key] = extra_update_res

        rets.append(ret)

    return rets


def reformat_seg_to_cap_eval(
    batch_data,
    batch_outputs,
    obj_key,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1],
            str), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"

        batch_data, batch_obj_key = batch_data
    batch_objects = batch_outputs[obj_key]
    rets = []
    for img_name, objects in zip(batch_data["img_name"], batch_objects):
        objects = objects.to("cpu")
        if os.path.splitext(img_name)[-1] != ".png":
            img_postfix = os.path.splitext(img_name)[-1]
            assert img_postfix in [".jpg", ".jpeg", "bmp"]
            img_name = img_name.replace(img_postfix, ".png")
        assert img_name.endswith(
            ".png"), f"Image type error! expect .png file, but get {img_name}"
        ret = {
            "image_name": img_name,
            "out_img": objects.mask.numpy(),
        }

        rets.append(ret)

    return rets
