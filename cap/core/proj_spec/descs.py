# flake8: noqa

import json

from .classification import get_rcnn_classification_desc
from .detection import get_frcnn_det_desc
from .lane_parsing import _get_lane_parsing_labels
from .parsing import (
    _get_image_fail_parsing_labels,
    get_default_parsing_labels,
    get_parsing_desc_template,
)
from .tracking_feature import get_rcnn_tracking_feature_desc


def frcnn_det_desc(
    task_name,
    class_names,
    class_agnostic_reg,
    with_background,
    roi_regions,
    vanishing_point,
    legacy_bbox,
    score_threshold_per_class=None,
    score_act_type="identity",
    reg_type="rcnn",
    merge_fn_type="default",
    merge_fn_params=None,
    conf_map_list=None,
    mean=(0, 0, 0, 0),
    std=(1, 1, 1, 1),
    nms_threshold=0.5,
    cls_output_name="box_score",
    reg_output_name="box_reg",
    input_padding=(0, 0, 0, 0),
):
    input_padding = list(input_padding)
    det_cls_desc, det_reg_desc = get_frcnn_det_desc(
        task_name=task_name,
        class_names=class_names,
        class_agnostic=1 if class_agnostic_reg else 0,
        with_background=1 if with_background else 0,
        cls_output_name=cls_output_name,
        reg_output_name=reg_output_name,
        roi_regions=roi_regions,
        vanishing_point=vanishing_point,
        legacy_bbox=legacy_bbox,
        score_threshold_per_class=score_threshold_per_class,
        score_act_type=score_act_type,
        reg_type=reg_type,
        merge_fn_type=merge_fn_type,
        merge_fn_params=merge_fn_params,
        conf_map_list=conf_map_list,
        mean=mean,
        std=std,
        nms_threshold=nms_threshold,
        input_padding=input_padding,
    )
    return det_cls_desc, det_reg_desc


def frcnn_classification_desc(
    task_name,
    output_name,
    class_names,
    desc_id,
    with_tracking_feat=False,
    feature_size=None,
):
    descs = [
        get_rcnn_classification_desc(
            task_name,
            output_name,
            class_names,
            desc_id,
        )
    ]
    if with_tracking_feat:
        assert feature_size is not None
        descs.append(
            get_rcnn_tracking_feature_desc(  # noqa
                feature_size=feature_size,
                class_name=class_names,
            )
        )

    return descs


def frcnn_roi_det_desc(
    task_name,
    class_names,
    label_output_name,
    offset_output_name,
):
    assert isinstance(class_names, list)
    desc_label = dict(
        task=task_name,
        class_name=class_names,
        output_name=label_output_name,
        properties=[
            dict(
                channel_labels=["conf_head_center"],
            )
        ],
    )
    desc_label = json.dumps(desc_label)

    # TODO: Fix hard-coding
    desc_offset = dict(
        task=task_name,
        class_name=class_names,
        output_name=offset_output_name,
        properties=[
            dict(
                channel_labels=[
                    "person_head_lefttop_offset_x",
                    "person_head_lefttop_offset_y",
                    "person_head_rightbottom_offset_x",
                    "person_head_rightbottom_offset_y",
                ],
            )
        ],
    )
    desc_offset = json.dumps(desc_offset)

    return desc_label, desc_offset


def frcnn_roi_det_with_subscore_desc(
    task_name,
    class_names,
    sub_class_name,
    label_output_name,
    offset_output_name,
    subbox_score_thresh,
):
    assert isinstance(class_names, list)
    desc_label = dict(
        task=task_name,
        class_name=class_names,
        output_name=label_output_name,
        properties=[
            dict(
                channel_labels=["conf_{}_center".format(sub_class_name)],
            )
        ],
        subbox_score_thresh=subbox_score_thresh,
    )
    desc_label = json.dumps(desc_label)

    desc_offset = dict(
        task=task_name,
        class_name=class_names,
        output_name=offset_output_name,
        properties=[
            dict(
                channel_labels=[
                    "{}_lefttop_offset_x".format(sub_class_name),
                    "{}_lefttop_offset_y".format(sub_class_name),
                    "{}_rightbottom_offset_x".format(sub_class_name),
                    "{}_rightbottom_offset_y".format(sub_class_name),
                ],
            )
        ],
    )
    desc_offset = json.dumps(desc_offset)

    return desc_label, desc_offset


def frcnn_veh_wheel_detection_desc(
    task_name,
    class_names,
    label_output_name,
    offset_output_name,
    subbox_score_thresh,
    max_subbox_num,
    roi_w_expand_scale=1.0,
    roi_h_expand_scale=1.0,
    nms_threshold=0.01,
):
    assert isinstance(class_names, list)
    desc_label = dict(
        task=task_name,
        class_name=class_names,
        output_name=label_output_name,
        properties=[
            dict(
                channel_labels=["conf_wheel_center"],
            )
        ],
        subbox_score_thresh=subbox_score_thresh,
        max_subbox_num=max_subbox_num,
        roi_w_expand_scale=float(roi_w_expand_scale),
        roi_h_expand_scale=float(roi_h_expand_scale),
        nms_threshold=nms_threshold,
    )
    desc_label = json.dumps(desc_label)
    desc_offset = dict(
        task=task_name,
        class_name=class_names,
        output_name=offset_output_name,
        properties=[
            dict(
                channel_labels=[
                    "vhicle_wheel_lefttop_offset_x",
                    "vhicle_wheel_lefttop_offset_y",
                    "vhicle_wheel_rightbottom_offset_x",
                    "vhicle_wheel_rightbottom_offset_y",
                ],
            )
        ],
    )
    desc_offset = json.dumps(desc_offset)

    return [desc_label, desc_offset]


def frcnn_kps_desc(
    task_name,
    class_names,
    label_output_name,
    roi_expand_param,
    offset_output_name,
    score_channel_labels=None,
    offset_channel_labels=None,
):
    roi_expand_param = float(roi_expand_param)
    if score_channel_labels is None:
        score_channel_labels = ["conf_kp_back", "conf_kp_front"]
    if offset_channel_labels is None:
        offset_channel_labels = [
            "wheel_kp_back_offset_x",
            "wheel_kp_back_offset_y",
            "wheel_kp_front_offset_x",
            "wheel_kp_front_offset_y",
        ]
    desc_label = dict(
        task=task_name,
        class_name=class_names,
        output_name=label_output_name,
        roi_expand_param=roi_expand_param,
        properties=[
            dict(
                channel_labels=score_channel_labels,
            )
        ],
    )

    desc_label = json.dumps(desc_label)
    desc_offset = dict(
        task=task_name,
        class_name=class_names,
        output_name=offset_output_name,
        roi_expand_param=roi_expand_param,
        properties=[
            dict(
                channel_labels=offset_channel_labels,
            )
        ],
    )
    desc_offset = json.dumps(desc_offset)

    return [desc_label, desc_offset]


def frcnn_roi_3d_detection_desc(
    task_name,
    class_names,
    roi_expand_param,
    roi_regions,
    vanishing_point,
    score_threshold,
    focal_length_default,
    scale_wh,
    undistort_2dcenter,
    undistort_depth_uv=False,
    input_padding=(0, 0, 0, 0),
):
    input_padding = list(input_padding)
    roi_expand_param = float(roi_expand_param)
    descs = []
    pred_2d_center_output = dict(
        task=task_name,
        output_name="2d_center_output",
        class_name=class_names,
        undistort_2dcenter=int(undistort_2dcenter),
        undistort_depth_uv=int(undistort_depth_uv),
        roi_expand_param=roi_expand_param,
        focal_length_default=focal_length_default,
        roi_input={
            "fp_x": vanishing_point[0] - roi_regions[0],
            "fp_y": vanishing_point[1] - roi_regions[1],
            "width": roi_regions[2] - roi_regions[0],
            "height": roi_regions[3] - roi_regions[1],
        },
        vanishing_point=[vanishing_point[0], vanishing_point[1]],
        scale_wh=[scale_wh[0], scale_wh[1]],
        score_threshold=score_threshold,
        input_padding=input_padding,
        properties=[
            dict(
                channel_labels=["_x", "_y"],
            )
        ],
    )
    descs.append(json.dumps(pred_2d_center_output))

    pred_2d_offset_output = dict(
        task=task_name,
        output_name="2d_offset_output",
        class_name=class_names,
        roi_expand_param=roi_expand_param,
        properties=[
            dict(
                channel_labels=["offset_x", "offset_y"],
            )
        ],
    )
    descs.append(json.dumps(pred_2d_offset_output))

    pred_3d_offset_output = dict(
        task=task_name,
        output_name="3d_offset_output",
        class_name=class_names,
        roi_expand_param=roi_expand_param,
        properties=[
            dict(
                channel_labels=["offset_x", "offset_y"],
            )
        ],
    )
    descs.append(json.dumps(pred_3d_offset_output))

    if undistort_depth_uv:
        depth_u_output = dict(
            task=task_name,
            output_name="depth_u_output",
            class_name=class_names,
            roi_expand_param=roi_expand_param,
            undistort_depth_uv=int(undistort_depth_uv),
            properties=[
                dict(
                    channel_labels=["depth_u"],
                )
            ],
        )
        descs.append(json.dumps(depth_u_output))
        depth_v_output = dict(
            task=task_name,
            output_name="depth_v_output",
            class_name=class_names,
            roi_expand_param=roi_expand_param,
            undistort_depth_uv=int(undistort_depth_uv),
            properties=[
                dict(
                    channel_labels=["depth_v"],
                )
            ],
        )
        descs.append(json.dumps(depth_v_output))
    else:
        depth_output = dict(
            task=task_name,
            output_name="depth_output",
            class_name=class_names,
            roi_expand_param=roi_expand_param,
            properties=[
                dict(
                    channel_labels=["depth"],
                )
            ],
        )
        descs.append(json.dumps(depth_output))

    pred_dim_output = dict(
        task=task_name,
        output_name="dim_output",
        class_name=class_names,
        properties=[
            dict(
                channel_labels=["height", "width", "length"],
            )
        ],
    )
    descs.append(json.dumps(pred_dim_output))
    # rot
    rot_output = dict(
        task=task_name,
        output_name="rot_output",
        class_name=class_names,
        properties=[
            dict(
                channel_labels=["sin", "cos"],  # noqa
            )
        ],
    )
    descs.append(json.dumps(rot_output))

    iou_output = dict(
        task=task_name,
        output_name="iou_output",
        class_name=class_names,
        properties=[
            dict(
                channel_labels=["score"],  # noqa
            )
        ],
    )
    descs.append(json.dumps(iou_output))

    return descs


def frcnn_gdl_desc(
    task_name,
    class_names,
    label_output_name,
    reg_output_name,
    roi_expand_param=1.0,
):
    desc_label = dict(
        task=task_name,
        class_name=class_names,
        output_name=label_output_name,
        roi_expand_param=roi_expand_param,
        properties=[
            dict(
                channel_labels=["conf_ground_line"],
            )
        ],
    )
    desc_label = json.dumps(desc_label)

    desc_reg = dict(
        task=task_name,
        class_name=class_names,
        output_name=reg_output_name,
        roi_expand_param=roi_expand_param,
        properties=[
            dict(
                channel_labels=[
                    "reg_ground_line_left",
                    "reg_ground_line_right",
                ],
            )
        ],
    )
    desc_reg = json.dumps(desc_reg)

    return [desc_label, desc_reg]


def frcnn_headmap_3d_detection_desc(
    task_name,
    classnames,
    focal_length_default,
    use_multibin,
    score_threshold,
    vanishing_point,
    roi_regions,
    undistort_2dcenter=False,
    undistort_depth_uv=False,
    output_head_with_2d_wh=True,
    input_padding=None,
):
    if input_padding is None:
        input_padding = [0, 0, 0, 0]
    heat_map_output = dict(
        task=task_name,
        output_name="heatmap_output",
        score_threshold=score_threshold,
        undistort_2dcenter=undistort_2dcenter,
        undistort_depth_uv=undistort_depth_uv,
        roi_input={
            "fp_x": vanishing_point[0] - roi_regions[0],
            "fp_y": vanishing_point[1] - roi_regions[1],
            "width": roi_regions[2] - roi_regions[0],
            "height": roi_regions[3] - roi_regions[1],
        },
        vanishing_point=[vanishing_point[0], vanishing_point[1]],
        focal_length_default=focal_length_default,
        input_padding=input_padding,
        properties=[
            dict(
                channel_labels=classnames,
            )
        ],
    )
    descs = []
    descs.append(json.dumps(heat_map_output))
    # depth
    if undistort_depth_uv:
        depth_u_output = dict(
            task=task_name,
            output_name="depth_u_output",
            properties=[
                dict(
                    channel_labels=["depth_u"],
                )
            ],
        )
        descs.append(json.dumps(depth_u_output))
        depth_v_output = dict(
            task=task_name,
            output_name="depth_v_output",
            properties=[
                dict(
                    channel_labels=["depth_v"],
                )
            ],
        )
        descs.append(json.dumps(depth_v_output))
    else:
        depth_output = dict(
            task=task_name,
            output_name="depth_output",
            properties=[
                dict(
                    channel_labels=["depth"],
                )
            ],
        )
        descs.append(json.dumps(depth_output))
    # rot
    # [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    # bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    rot_output = dict(
        task=task_name,
        output_name="rot_output",
        use_multibin=1 if use_multibin else 0,
        properties=[
            dict(
                channel_labels=[
                    "bin1_cls_0",
                    "bin1_cls_1",
                    "bin1_sin",
                    "bin1_cos",  # noqa
                    "bin2_cls_0",
                    "bin2_cls_1",
                    "bin2_sin",
                    "bin2_cos",
                ],  # noqa
            )
        ]
        if use_multibin
        else [
            dict(
                channel_labels=["sin", "cos"],  # noqa
            )
        ],
    )
    descs.append(json.dumps(rot_output))
    # dimensions
    dim_output = dict(
        task=task_name,
        output_name="dim_output",
        properties=[
            dict(
                channel_labels=["height", "width", "length"],
            )
        ],
    )
    descs.append(json.dumps(dim_output))
    # location_offset
    loc_offset_output = dict(
        task=task_name,
        output_name="loc_offset_output",
        properties=[
            dict(
                channel_labels=["offset_x", "offset_y"],
            )
        ],
    )
    descs.append(json.dumps(loc_offset_output))
    # 2d width hight
    if output_head_with_2d_wh:
        wh_output = dict(
            task=task_name,
            output_name="wh_output",
            properties=[
                dict(
                    channel_labels=["bbox_w", "bbox_h"],
                )
            ],
        )
        descs.append(json.dumps(wh_output))

    return descs


_supported_norm_method = [
    "resize",
    "width",
    "height",
    "diagonal",
    "mean_width_height",
    "sqrt_width_height",
    "max_width_height",
    "max_2width_height",
]


def get_crop_desc(
    image_size, norm_len, norm_method, padding_context=None, extra_info=None
):
    """Get classification model description.

    Parameters
    ----------
    image_size: list or tuple
        Image size.
    norm_len: int
        Normalization length.
    norm_method: str
        Normalization method, supported norm_method:
            ['resize', 'width', 'height', 'diagonal',
            'mean_width_height', 'sqrt_width_height',
            'max_width_height', 'max_2width_height']
    padding_context: list or tuple, optional
        Padding context, by default None.
    """

    assert isinstance(image_size, (list, tuple)) and len(image_size) == 2
    assert isinstance(norm_len, int)
    assert norm_method in _supported_norm_method, (
        "unsupported norm_method: %s" % norm_method
    )
    if norm_method == "resize":
        assert (
            isinstance(padding_context, (list, tuple))
            and len(padding_context) == 4
        )
        for p in padding_context:
            assert type(p) == float

    # 1. do not use true/false in json files
    # 2. use 1.0 instead of 1 for float number

    if norm_method == "resize":
        desc = {
            "image_size": image_size,  # is image_hw
            "norm_method": norm_method,
            "padding": padding_context,
        }
    else:
        desc = {
            "image_size": image_size,
            "norm_method": norm_method,
            "norm_len": norm_len,
        }

    if extra_info is not None:
        desc.update(extra_info)

    return json.dumps(desc)


def merge_str_desc(desc1, desc2):
    assert isinstance(desc1, str)
    assert isinstance(desc2, str)
    desc1 = json.loads(desc1)
    desc2 = json.loads(desc2)
    assert isinstance(desc1, dict)
    assert isinstance(desc2, dict)
    desc1.update(desc2)
    return json.dumps(desc1)


def get_default_parsing_desc(
    desc_id,
    roi_regions=None,
    vanishing_point=None,
    merge_fn_type="default",
    merge_fn_params=None,
    input_padding=None,
):
    """Get __desc__ descriptor for parsing, not lane parsing.

    Parameters
    ----------
    desc_id: str
        id of a desc map, e.g cn_24, including a 2-char for data identifier
        and class_num. The 2-char identifier is used in case that different
        datasets have the same class number.

    Returns
    -------
    desc: str
        The __desc__ descriptor in json string format.
    """
    desc = get_parsing_desc_template(
        task_name="parsing",
        get_label_fn=get_default_parsing_labels,
        desc_id=desc_id,
        roi_regions=roi_regions,
        vanishing_point=vanishing_point,
        merge_fn_type=merge_fn_type,
        merge_fn_params=merge_fn_params,
        input_padding=input_padding,
    )
    return desc


def get_lane_parsing_desc(
    desc_id,
    roi_regions=None,
    vanishing_point=None,
    merge_fn_type="default",
    merge_fn_params=None,
    input_padding=None,
):
    """Get __desc__ descriptor for lane parsing, not for default parsing.

    Parameters
    ----------
    desc_id: str
        id of a desc map, e.g gl_4, including a 2-char for data identifier
        and class_num. The 2-char identifier is used in case that different
        datasets have the same class number.

    Returns
    -------
    desc: str
        The __desc__ descriptor in json string format.
    """
    desc = get_parsing_desc_template(
        task_name="lane_parsing",
        get_label_fn=_get_lane_parsing_labels,
        desc_id=desc_id,
        roi_regions=roi_regions,
        vanishing_point=vanishing_point,
        merge_fn_type=merge_fn_type,
        merge_fn_params=merge_fn_params,
        input_padding=input_padding,
    )
    return desc


def get_image_fail_parsing_desc(
    desc_id,
    roi_regions=None,
    vanishing_point=None,
    merge_fn_type="default",
    merge_fn_params=None,
    input_padding=None,
):
    """Get __desc__ descriptor for image fail parsing.

    Parameters
    ----------
    desc_id: str
        id of a desc map, e.g gl_4, including a 2-char for data identifier
        and class_num. The 2-char identifier is used in case that different
        datasets have the same class number.

    Returns
    -------
    desc: str
        The __desc__ descriptor in json string format.
    """
    desc = get_parsing_desc_template(
        task_name="image_fail_parsing",
        get_label_fn=_get_image_fail_parsing_labels,
        desc_id=desc_id,
        roi_regions=roi_regions,
        vanishing_point=vanishing_point,
        merge_fn_type=merge_fn_type,
        merge_fn_params=merge_fn_params,
        input_padding=input_padding,
    )
    return desc


def frcnn_flank_desc(
    task_name,
    class_names,
    label_output_name=None,
    offset_output_name=None,
    score_channel_labels=None,
    offset_channel_labels=None,
    roi_expand_param=1.0,
    feature="undistortion",
):
    if label_output_name is None:
        label_output_name = "vehicle_ground_line_label"
    if offset_output_name is None:
        offset_output_name = "vehicle_ground_line_reg"
    if score_channel_labels is None:
        score_channel_labels = [
            "conf_flank",
        ]
    if offset_channel_labels is None:
        offset_channel_labels = [
            "flank_corner_back_offset_x",
            "flank_corner_back_offset_y",
            "flank_corner_front_offset_x",
            "flank_corner_front_offset_y",
        ]

    desc_label = dict(
        task=task_name,
        class_name=class_names,
        output_name=label_output_name,
        roi_expand_param=roi_expand_param,
        properties=[dict(channel_labels=score_channel_labels)],
        feature=feature,
    )
    desc_label = json.dumps(desc_label)

    desc_reg = dict(
        task=task_name,
        class_name=class_names,
        output_name=offset_output_name,
        roi_expand_param=roi_expand_param,
        properties=[dict(channel_labels=offset_channel_labels)],
        feature=feature,
    )
    desc_reg = json.dumps(desc_reg)

    return [desc_label, desc_reg]
