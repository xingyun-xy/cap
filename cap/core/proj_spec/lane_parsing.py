import numpy as np

from .parsing import (
    ParsingDescClassName,
    ParsingDescColorMap,
    get_parsing_desc_template,
)


def _get_lane_parsing_labels(desc_id):
    """Get lane parsing labels by the number of classes.

    Parameters
    ----------
    desc_id: str
        id of a desc map, e.g gl_3, including a 2-char for data identifier
        and class_num. The 2-char identifier is used in case that different
        datasets have the same class number.

    Returns
    -------
    labels: list of dict
        The desc labels.
    """
    assert desc_id
    num_classes = int(desc_id.split("_")[-1])
    if num_classes == 3:
        labels = [
            {
                ParsingDescClassName: [
                    "road",
                    "pothole",
                    "sidewalk",
                    "building",
                    "fence",
                    "pole",
                    "vegetation",
                    "terrain",
                    "sky",
                    "traffic_cone",
                    "bollard",
                    "traffic_light",
                    "traffic_sign",
                    "traffic_guide_sign",
                    "person",
                    "rider",
                    "car",
                    "truck",
                    "bus",
                    "train",
                    "motorcycle",
                    "bicycle",
                    "tricycle",
                    "stop_line",
                    "crosswalk_line",
                    "traffic_arrow",
                    "sign_line",
                    "guide_line",
                    "slow_down_triangle",
                    "speed_sign",
                    "diamond",
                    "bicycle_sign",
                    "speed_bump",
                ],
                ParsingDescColorMap: [60, 60, 60],
            },
            {
                ParsingDescClassName: ["traffic_line"],
                ParsingDescColorMap: [0, 0, 255],
            },
            {ParsingDescClassName: ["curb"], ParsingDescColorMap: [0, 255, 0]},
        ]
    elif num_classes == 4:
        labels = [
            {
                ParsingDescClassName: [
                    "road",
                    "pothole",
                    "sidewalk",
                    "building",
                    "fence",
                    "pole",
                    "vegetation",
                    "terrain",
                    "sky",
                    "traffic_cone",
                    "bollard",
                    "traffic_light",
                    "traffic_sign",
                    "traffic_guide_sign",
                    "person",
                    "rider",
                    "car",
                    "truck",
                    "bus",
                    "train",
                    "motorcycle",
                    "bicycle",
                    "tricycle",
                    "stop_line",
                    "crosswalk_line",
                    "traffic_arrow",
                    "sign_line",
                    "guide_line",
                    "slow_down_triangle",
                    "speed_sign",
                    "diamond",
                    "bicycle_sign",
                    "speed_bump",
                ],
                ParsingDescColorMap: [60, 60, 60],
            },
            {
                ParsingDescClassName: ["traffic_line"],
                ParsingDescColorMap: [0, 0, 255],
            },
            {ParsingDescClassName: ["curb"], ParsingDescColorMap: [0, 255, 0]},
            {
                ParsingDescClassName: ["other"],
                ParsingDescColorMap: [250, 0, 0],
            },
        ]
    elif num_classes == 5:
        if desc_id.split("_")[0] == "wd":
            labels = [
                {
                    ParsingDescClassName: [
                        "road",
                        "pothole",
                        "sidewalk",
                        "building",
                        "fence",
                        "pole",  # noqa
                        "vegetation",
                        "terrain",
                        "sky",
                        "traffic_cone",
                        "bollard",  # noqa
                        "traffic_light",
                        "traffic_sign",
                        "traffic_guide_sign",
                        "person",
                        "rider",
                        "car",
                        "truck",
                        "bus",
                        "train",
                        "motorcycle",
                        "bicycle",
                        "tricycle",
                        "stop_line",
                        "crosswalk_line",
                        "traffic_arrow",
                        "sign_line",
                        "guide_line",
                        "slow_down_triangle",
                        "speed_sign",
                        "diamond",
                        "bicycle_sign",
                        "speed_bump",
                    ],
                    ParsingDescColorMap: [60, 60, 60],
                },
                {
                    ParsingDescClassName: ["traffic_line"],
                    ParsingDescColorMap: [0, 0, 255],
                },
                {
                    ParsingDescClassName: ["curb"],
                    ParsingDescColorMap: [0, 255, 0],
                },
                {
                    ParsingDescClassName: ["other"],
                    ParsingDescColorMap: [250, 0, 0],
                },
                {
                    ParsingDescClassName: ["wide_dashed"],
                    ParsingDescColorMap: [250, 255, 0],
                },
            ]
        else:
            labels = [
                {
                    ParsingDescClassName: [
                        "road",
                        "pothole",
                        "sidewalk",
                        "building",
                        "fence",
                        "pole",  # noqa
                        "vegetation",
                        "terrain",
                        "sky",
                        "traffic_cone",
                        "bollard",  # noqa
                        "traffic_light",
                        "traffic_sign",
                        "traffic_guide_sign",
                        "person",
                        "rider",
                        "car",
                        "truck",
                        "bus",
                        "train",
                        "motorcycle",
                        "bicycle",
                        "tricycle",
                        "stop_line",
                        "crosswalk_line",
                        "traffic_arrow",
                        "sign_line",
                        "guide_line",
                        "slow_down_triangle",
                        "speed_sign",
                        "diamond",
                        "bicycle_sign",
                        "speed_bump",
                    ],
                    ParsingDescColorMap: [60, 60, 60],
                },
                {
                    ParsingDescClassName: ["traffic_line"],
                    ParsingDescColorMap: [0, 0, 255],
                },
                {
                    ParsingDescClassName: ["curb"],
                    ParsingDescColorMap: [0, 255, 0],
                },
                {
                    ParsingDescClassName: ["other"],
                    ParsingDescColorMap: [250, 0, 0],
                },
                {
                    ParsingDescClassName: ["wide"],
                    ParsingDescColorMap: [250, 255, 0],
                },
            ]
    elif num_classes == 6:
        # labels = [
        #     {
        #         ParsingDescClassName: ["background"],
        #         ParsingDescColorMap: [60, 60, 60],
        #     },
        #     {
        #         ParsingDescClassName: ["single_dashed"],
        #         ParsingDescColorMap: [0, 0, 255],
        #     },
        #     {
        #         ParsingDescClassName: ["single_solid"],
        #         ParsingDescColorMap: [0, 255, 0],
        #     },
        #     {
        #         ParsingDescClassName: ["double_dashed"],
        #         ParsingDescColorMap: [255, 255, 0],
        #     },
        #     {
        #         ParsingDescClassName: ["double_solid"],
        #         ParsingDescColorMap: [0, 255, 255],
        #     },
        #     {
        #         ParsingDescClassName: ["curb"],
        #         ParsingDescColorMap: [255, 0, 255],
        #     },
        # ]
        labels = [
            {
                ParsingDescClassName: ["road"], #background
                ParsingDescColorMap: [0, 0, 0],
            },
            {
                ParsingDescClassName: ["dashed"], #dashed,solid,mixed,wide_solid,tidal_lane
                ParsingDescColorMap: [0, 0, 255],
            },
            {
                ParsingDescClassName: ["Road_teeth"],
                ParsingDescColorMap: [0, 255, 0],
            },
            {
                ParsingDescClassName: ["double_line"],
                ParsingDescColorMap: [255, 255, 0],
            },
            {
                ParsingDescClassName: ["wide_dashed"], #wide_dashed,wide_lane
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["deceleration_lane"],
                ParsingDescColorMap: [0, 255, 255],
            },
        ]
    elif num_classes == 10:
        labels = [
            {
                ParsingDescClassName: ["background"],
                ParsingDescColorMap: [60, 60, 60],
            },
            {
                ParsingDescClassName: ["white_single_solid"],
                ParsingDescColorMap: [0, 0, 255],
            },
            {
                ParsingDescClassName: ["white_single_dashed"],
                ParsingDescColorMap: [0, 0, 128],
            },
            {
                ParsingDescClassName: ["yellow_single_solid"],
                ParsingDescColorMap: [0, 255, 0],
            },
            {
                ParsingDescClassName: ["yellow_single_dashed"],
                ParsingDescColorMap: [0, 128, 0],
            },
            {
                ParsingDescClassName: ["white_double_solid"],
                ParsingDescColorMap: [255, 255, 0],
            },
            {
                ParsingDescClassName: ["white_double_dashed"],
                ParsingDescColorMap: [128, 128, 0],
            },
            {
                ParsingDescClassName: ["yellow_double_solid"],
                ParsingDescColorMap: [255, 0, 255],
            },
            {
                ParsingDescClassName: ["yellow_double_dashed"],
                ParsingDescColorMap: [128, 0, 128],
            },
            {
                ParsingDescClassName: ["curb"],
                ParsingDescColorMap: [255, 128, 255],
            },
        ]
    else:
        raise ValueError("unsupported number of classes %s" % num_classes)
    # check again
    assert len(labels) == num_classes
    return labels


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


_lane_parsing_colors = np.zeros((256, 1, 3), dtype="uint8")
_lane_parsing_colors[0, :, :] = [0, 0, 0]
_lane_parsing_colors[1, :, :] = [0, 0, 255]
_lane_parsing_colors[2, :, :] = [0, 255, 0]
_lane_parsing_colors[3, :, :] = [255, 255, 0]
_lane_parsing_colors[255, :, :] = [255, 0, 0]
