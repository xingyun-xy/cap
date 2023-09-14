# flake8: noqa
import json

from cap.utils.apply_func import _as_list

from .parsing import get_roi_desc

_det_task_name_list = {
    # 4pe
    "detection",
    "frcnn_detection",
    # 2pe
    "vehicle_license_plate_box_detection",
    "person_body_detection",
    "person_head_detection",
    "rear_detection",
    "vehicle_detection",
    "vehicle_full_fast_detection",
    "vehicle_rear_fast_detection",
    "fast_rear_detection",
    "traffic_sign_assist_detection",
    "fast_traffic_sign_detection",
    "fast_traffic_light_detection",
    "vehicle_wheel_detection",
    "cyclist_2pe_detection",
    "person_2pe_detection",
    "traffic_lens_detection",
    "tl_lens_detection,"
    "traffic_cone_fast_detection",
}

_class_name_map = {
    "vehicle_rear": "vehicle_rear",
    "rear": "vehicle_rear",
    "vehicle": "vehicle",
    "person": "person",
    "cyclist": "cyclist",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
}

classname2id = dict(  # noqa
    person=1,
    traffic_light=2,
    traffic_sign=3,
    vehicle=4,
    vehicle_rear=5,
    rear=5,
    cyclist=7,
    traffic_cone=9,
    parking_lock=10,
)


def get_class_names_used_in_desc(class_names):
    mapped_class_names = []
    for name in _as_list(class_names):
        assert name in _class_name_map, ("can not fine %s in _class_name_map" %
                                         name)
        mapped_class_names.append(_class_name_map[name])
    return mapped_class_names


def get_det_default_merge_fn_type_and_params():
    merge_fn_type = "default"
    merge_fn_params = {
        "inside_roi_threshold": 0.9,
        "inside_roi_match_threshold": 0.5,
        "outside_roi_match_threshold": 0.5,
    }
    return merge_fn_type, merge_fn_params


_reg_type_map = {"densebox": "densebox", "rcnn": "rcnn", "frcnn": "rcnn"}

_det_merge_fn_list = {
    "default": {
        "key": [
            "inside_roi_threshold",
            "inside_roi_match_threshold",
            "outside_roi_match_threshold",
        ]
    }
}


def check_det_merge_fn_type_and_params(merge_fn_type, merge_fn_params):
    assert (merge_fn_type in _det_merge_fn_list
            ), "unexpected merge_fn_type %s, all valid types are %s" % (
                merge_fn_type,
                _det_merge_fn_list.keys(),
            )
    assert merge_fn_params is not None and isinstance(merge_fn_params, dict)
    for key in merge_fn_params:
        assert (key in _det_merge_fn_list[merge_fn_type]["key"]
                ), "unexpected param %s, all valid params %s" % (
                    key,
                    _det_merge_fn_list[merge_fn_type]["key"],
                )


def get_det_frcnn_merge_fn_type_and_params(crop_distance, w_or_h,
                                           obj_real_size):  # noqa
    if crop_distance is None:
        crop_distance = 20
    if w_or_h is None:
        w_or_h = "h"
    if obj_real_size is None:
        obj_real_size = 1.7
    merge_fn_type = "frcnn"
    merge_fn_params = {
        "inside_roi_threshold": 0.9,
        "inside_roi_match_threshold": 0.5,
        "outside_roi_match_threshold": 0.5,
        "crop_distance": crop_distance,
        "w_or_h": w_or_h,
        "obj_real_size": obj_real_size,
    }
    return merge_fn_type, merge_fn_params


def get_frcnn_det_desc(
        class_names,
        class_agnostic,
        score_act_type,
        with_background,
        cls_output_name,
        reg_output_name,
        mean=(0, 0, 0, 0),
        std=(1, 1, 1, 1),
        reg_type="rcnn",
        roi_regions=None,
        vanishing_point=None,
        merge_fn_type="default",
        merge_fn_params=None,
        conf_map_list=None,
        task_name="frcnn_detection",
        score_threshold_per_class=None,
        legacy_bbox=True,
        nms_threshold=0.5,
        crop_distance=None,
        w_or_h=None,
        obj_real_size=None,
        cascade=0,
        input_padding=(0, 0, 0, 0),
):
    """
    Get __desc__ descriptor for detection models.

    Parameters
    ----------
    class_names: list of str
        Class names.
    class_agnostic: bool or int
        If class_agnostic is true then rcnn box reg will be agnostic.
    score_act_type: str
        Type of score_act.
    with_background: bool or int
        Calculate cls score with background or not.
    reg_type: str
        Regression type.
    roi_regions: tuple
        Region of interest.
    vanishing_point: list
        Vanishing point.
    merge_fn_type: str
        Type of merge.
    merge_fn_params: dict
        The params be used when merging resize and crop.
    conf_map_list: list or tuple of dict
        Conf_value and conf_percentage pairs for each class:
        [
            {
                "conf_value": [
                    ...
                ],
                "conf_percentage": [
                    ...
                ]
            },
            {
                "conf_value": [
                    ...
                ],
                "conf_percentage": [
                    ...
                ]
            },
            ...
        ]
    task_name: str
        Task name.
    score_threshold_per_class: list/tuple of float
        Score threshold for each class.
    legacy_bbox: bool
        Useful when reg_type is `rcnn`, +1 pixel when computing bbox offset
        if True.
    crop_distance: float
        more than distance will use crop model results.
    w_or_h: str
        use w or h cal bbox pixel
    obj_real_size: float
        object real size
    Returns
    -------
    desc: str
        The __desc__ descriptor in json string format.
    """
    if score_threshold_per_class is not None:
        score_threshold_per_class = _as_list(score_threshold_per_class)
    if class_names is not None:
        class_names = _as_list(class_names)

    def _check():
        assert (reg_type in _reg_type_map
                ), "unexpected reg_type %s, all valid reg_types are %s" % (
                    reg_type,
                    _reg_type_map.keys(),
                )
        if conf_map_list is not None:
            assert isinstance(
                conf_map_list,
                (list,
                 tuple)), "expect list/tuple, but get %s" % type(conf_map_list)
            assert len(conf_map_list) == len(class_names), "%d vs. %d" % (
                len(conf_map_list),
                len(class_names),
            )
            for conf_dict in conf_map_list:
                assert isinstance(
                    conf_dict,
                    dict), "expect dict, but get %s" % type(conf_map_list)
                assert ("conf_value" in conf_dict
                        and "conf_percentage" in conf_dict)
                assert len(conf_dict["conf_value"]) == len(
                    conf_dict["conf_percentage"]), "%d vs. %d" % (
                        len(conf_dict["conf_value"]),
                        len(conf_dict["conf_percentage"]),
                    )
        if score_threshold_per_class is not None:
            assert len(score_threshold_per_class) == len(class_names)
        assert task_name in _det_task_name_list, (
            "Unknown task: %s, register in `_det_task_names`" % task_name)
        assert isinstance(legacy_bbox, bool)

    _check()

    desc = {
        "task": task_name,
        "class_name": get_class_names_used_in_desc(class_names),
        "class_agnostic": class_agnostic,
        "score_act_type": score_act_type,
        "with_background": with_background,
        "mean": mean,
        "std": std,
        "reg_type": reg_type,
        "legacy_bbox": int(legacy_bbox),  # use 0/1 instead of False/True
        "nms_threshold": nms_threshold,
        "cascade": cascade,
        "input_padding": list(input_padding),
    }
    # add conf_map
    if conf_map_list is not None:
        desc.update({"conf_map": conf_map_list})
    if score_threshold_per_class is not None:
        desc.update({"score_threshold": score_threshold_per_class})

    if roi_regions is not None:

        (
            frcnn_merge_fn_type,
            frcnn_merge_fn_params,
        ) = get_det_frcnn_merge_fn_type_and_params(crop_distance, w_or_h,
                                                   obj_real_size)
        if merge_fn_type is None:
            merge_fn_type = frcnn_merge_fn_type
        if merge_fn_params is None:
            merge_fn_params = frcnn_merge_fn_params
        # check_det_merge_fn_type_and_params(
        #     merge_fn_type, merge_fn_params)

        roi_desc = get_roi_desc(
            roi_regions=roi_regions,
            vanishing_point=vanishing_point,
            merge_fn_type=merge_fn_type,
            merge_fn_params=merge_fn_params,
        )

        desc.update(roi_desc)
    cls_desc = desc.copy()
    reg_desc = desc.copy()
    cls_desc["output_name"] = cls_output_name
    reg_desc["output_name"] = reg_output_name
    cls_desc = json.dumps(cls_desc)
    reg_desc = json.dumps(reg_desc)

    return cls_desc, reg_desc
