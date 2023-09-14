import json

import numpy as np

ParsingDescClassName = "class_name"
ParsingDescColorMap = "color_map"
ParsingDescTask = "task"
ParsingDescNumClasses = "num_classes"
ParsingDescLabels = "labels"


_parsing_task_list = ["parsing", "lane_parsing", "image_fail_parsing"]


_seg_merge_fn_list = {"default": {"key": {}}}

colormap = [
    [128, 64, 128],  # 0
    [70, 70, 70],  # 1
    [190, 153, 15],  # 2
    [153, 153, 15],  # 3
    [250, 170, 30],  # 4
    [220, 20, 60],  # 5
    [0, 0, 142],  # 6
    [0, 0, 70],  # 7
    [200, 200, 20],
    [64, 192, 0],
    [128, 0, 192],
    [200, 200, 12],
    [0, 192, 192],
    [0, 64, 64],
    [255, 0, 0],
    [220, 220, 0],  # 15
    [128, 192, 192],
    [64, 64, 64],
    [192, 64, 64],
    [64, 192, 64],
    [192, 192, 64],
]


def get_seg_default_merge_fn_type_and_params():
    merge_fn_type = "default"
    merge_fn_params = {}
    return merge_fn_type, merge_fn_params


def check_seg_merge_fn_type_and_params(merge_fn_type, merge_fn_params):
    assert (
        merge_fn_type in _seg_merge_fn_list
    ), "unexpected merge_fn_type %s, all valid types are %s" % (
        merge_fn_type,
        _seg_merge_fn_list.keys(),
    )
    assert merge_fn_params is not None and isinstance(merge_fn_params, dict)
    for key in merge_fn_params:
        assert (
            key in _seg_merge_fn_list[merge_fn_type]["key"]
        ), "unexpected param %s, all valid params %s" % (
            key,
            _seg_merge_fn_list[merge_fn_type]["key"],
        )


def get_default_parsing_labels(desc_id):
    """Get parsing labels by the number of classes.

    Parameters
    ----------
    desc_id: str
        description id for a parsing label config e.g. us_16.
        must ends with numbers.

    Returns
    -------
    labels: list of dict
        The desc labels.
    """
    if desc_id == "us_8":
        labels = [
            {
                ParsingDescClassName: ["road", "pothole"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "building",
                    "fence",
                    "pole",
                    "vegetation",
                    "terrain",
                    "sky",
                    "traffic_cone",
                    "bollard",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: [
                    "traffic_light",
                    "traffic_sign",
                    "traffic_guide_sign",
                ],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["motorcycle", "bicycle", "tricycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["traffic_lane", "stop_line"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: [
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
                ParsingDescColorMap: [200, 200, 128],
            },
        ]
    elif desc_id == "gl_12":
        labels = [
            {
                ParsingDescClassName: ["road", "pothole", "parking_space"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "building",
                    "pole",
                    "traffic_light",
                    "traffic_sign",
                    "vegetation",
                    "terrain",
                    "sky",
                    "traffic_guide_sign",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["motorcycle", "bicycle", "tricycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: [
                    "traffic_lane",
                    "traffic_arrow",
                    "sign_line",
                    "slow_down_triangle",
                    "speed_sign",
                    "diamond",
                    "bicycle_sign",
                    "parking_line",
                ],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [0, 192, 192],
            },
            {
                ParsingDescClassName: ["traffic_cone", "bollard"],
                ParsingDescColorMap: [0, 64, 64],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["speed_bump"],
                ParsingDescColorMap: [220, 220, 0],
            },
        ]
    elif desc_id == "gl2_12":
        labels = [
            {
                ParsingDescClassName: [
                    "road",
                    "pothole",
                    "parking_space",
                    "traversable_obstruction",
                ],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "building",
                    "pole",
                    "traffic_light",
                    "traffic_sign",
                    "vegetation",
                    "terrain",
                    "sky",
                    "traffic_guide_sign",
                    "parking_rod",
                    "parking_lock",
                    "column",
                    "no_forward_marker",
                    "untraversable_obstruction",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: [
                    "car",
                    "truck",
                    "bus",
                    "train",
                    "tricycle",
                ],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["motorcycle", "bicycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["traffic_lane"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [0, 192, 192],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: [
                    "traffic_arrow",
                    "sign_line",
                    "slow_down_triangle",
                    "speed_sign",
                    "diamond",
                    "bicycle_sign",
                    "speed_bump",
                    "parking_line",
                ],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["traffic_cone", "bollard"],
                ParsingDescColorMap: [0, 64, 64],
            },
        ]
    elif desc_id == "gl_13":
        labels = [
            {
                ParsingDescClassName: ["road", "pothole", "parking_space"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "building",
                    "traffic_light",
                    "traffic_sign",
                    "vegetation",
                    "terrain",
                    "sky",
                    "traffic_guide_sign",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["motorcycle", "bicycle", "tricycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: [
                    "traffic_lane",
                    "traffic_arrow",
                    "sign_line",
                    "slow_down_triangle",
                    "speed_sign",
                    "diamond",
                    "bicycle_sign",
                    "parking_line",
                ],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [0, 192, 192],
            },
            {
                ParsingDescClassName: ["traffic_cone", "bollard"],
                ParsingDescColorMap: [0, 64, 64],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["speed_bump"],
                ParsingDescColorMap: [220, 220, 0],
            },
            {
                ParsingDescClassName: ["pole"],
                ParsingDescColorMap: [153, 153, 153],
            },
        ]

    elif desc_id == "gl_33":
        labels = [
            {
                ParsingDescClassName: [
                    "road",
                    "pothole",
                    "parking_space",
                    "traversable_obstruction",
                ],
                ParsingDescColorMap: [0, 0, 0],
            },
            {
                ParsingDescClassName: ["sidewalk"],
                ParsingDescColorMap: [244, 35, 232],
            },
            {
                ParsingDescClassName: ["vegetation"],
                ParsingDescColorMap: [107, 142, 35],
            },
            {
                ParsingDescClassName: ["terrain"],
                ParsingDescColorMap: [152, 251, 152],
            },
            {
                ParsingDescClassName: ["pole"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["traffic_sign"],
                ParsingDescColorMap: [220, 220, 0],
            },
            {
                ParsingDescClassName: ["traffic_light"],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["sign_line"],
                ParsingDescColorMap: [219, 112, 147],
            },
            {
                ParsingDescClassName: ["traffic_lane", "parking_line"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: ["person"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["rider"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["bicycle"],
                ParsingDescColorMap: [119, 11, 32],
            },
            {
                ParsingDescClassName: ["motorcycle"],
                ParsingDescColorMap: [0, 0, 230],
            },
            {
                ParsingDescClassName: ["tricycle"],
                ParsingDescColorMap: [128, 192, 0],
            },
            {ParsingDescClassName: ["car"], ParsingDescColorMap: [0, 0, 142]},
            {ParsingDescClassName: ["truck"], ParsingDescColorMap: [0, 0, 70]},
            {ParsingDescClassName: ["bus"], ParsingDescColorMap: [0, 60, 100]},
            {
                ParsingDescClassName: ["train"],
                ParsingDescColorMap: [0, 80, 100],
            },
            {
                ParsingDescClassName: [
                    "building",
                    "parking_rod",
                    "parking_lock",
                    "column",
                    "no_forward_marker",
                    "untraversable_obstruction",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [139, 0, 139],
            },
            {
                ParsingDescClassName: ["sky"],
                ParsingDescColorMap: [70, 130, 180],
            },
            {
                ParsingDescClassName: ["traffic_cone"],
                ParsingDescColorMap: [238, 18, 137],
            },
            {
                ParsingDescClassName: ["bollard"],
                ParsingDescColorMap: [255, 246, 143],
            },
            {
                ParsingDescClassName: ["traffic_guide_sign"],
                ParsingDescColorMap: [139, 69, 19],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [255, 127, 80],
            },
            {
                ParsingDescClassName: ["traffic_arrow"],
                ParsingDescColorMap: [47, 79, 79],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [0, 128, 0],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [192, 0, 64],
            },
            {
                ParsingDescClassName: ["slow_down_triangle"],
                ParsingDescColorMap: [0, 250, 154],
            },
            {
                ParsingDescClassName: ["speed_sign"],
                ParsingDescColorMap: [173, 255, 47],
            },
            {
                ParsingDescClassName: ["diamond"],
                ParsingDescColorMap: [0, 64, 192],
            },
            {
                ParsingDescClassName: ["bicycle_sign"],
                ParsingDescColorMap: [128, 0, 192],
            },
            {
                ParsingDescClassName: ["speed_bump"],
                ParsingDescColorMap: [192, 128, 64],
            },
        ]

    elif desc_id == "us_16":
        labels = [
            {
                ParsingDescClassName: ["road", "pothole", "parking_space"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "building",
                    "vegetation",
                    "terrain",
                    "sky",
                    "parking_lock",
                    "column",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["pole"],
                ParsingDescColorMap: [153, 153, 153],
            },
            {
                ParsingDescClassName: [
                    "traffic_light",
                    "traffic_sign",
                    "traffic_guide_sign",
                ],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["motorcycle", "bicycle", "tricycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["traffic_lane"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["traffic_arrow"],
                ParsingDescColorMap: [128, 0, 192],
            },
            {
                ParsingDescClassName: [
                    "sign_line",
                    "slow_down_triangle",
                    "speed_sign",
                    "diamond",
                    "bicycle_sign",
                    "parking_line",
                ],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [0, 192, 192],
            },
            {
                ParsingDescClassName: ["traffic_cone"],
                ParsingDescColorMap: [0, 64, 64],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["speed_bump"],
                ParsingDescColorMap: [220, 220, 0],
            },
        ]
    elif desc_id == "us_21":
        labels = [
            {
                ParsingDescClassName: ["road", "pothole", "parking_space","traversable_obstruction"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "building",
                    "vegetation",
                    "terrain",
                    "sky",
                    "parking_rod",
                    "parking_lock",
                    "column",
                    "no_forward_marker",
                    "untraversable_obstruction",
                    "toll_pole",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["pole"],
                ParsingDescColorMap: [153, 153, 153],
            },
            {
                ParsingDescClassName: [
                    "traffic_light",
                    "traffic_sign",
                    "Traffic_Sign1",
                    "traffic_guide_sign",
                    "Guide_Post",
                ],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train","tricycle"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["motorcycle", "bicycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["lane_marking"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: ["Crosswalk_Line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["Traffic_Arrow"],
                ParsingDescColorMap: [128, 0, 192],
            },
            {
                ParsingDescClassName: [
                    "Sign_Line",
                    "Slow_Down_Triangle",
                    "Speed_Sign",
                    "Diamond",
                    "BicycleSign",
                    "parking_line",
                ],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["Guide_Line"],
                ParsingDescColorMap: [0, 192, 192],
            },
            {
                ParsingDescClassName: ["Traffic_Cone","Bollard"],
                ParsingDescColorMap: [0, 64, 64],
            },
            {
                ParsingDescClassName: ["Stop_Line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["SpeedBumps"],
                ParsingDescColorMap: [220, 220, 0],
            },
            {
                ParsingDescClassName: ["parking_column"],
                ParsingDescColorMap: [128, 192, 192],
            },
            {
                ParsingDescClassName: ["no_parking_line"],
                ParsingDescColorMap: [64, 64, 64],
            },
            {
                ParsingDescClassName: ["slow_down_line"],
                ParsingDescColorMap: [192, 64, 64],
            },
            {
                ParsingDescClassName: ["road_text"],
                ParsingDescColorMap: [64, 192, 64],
            },
            {
                ParsingDescClassName: ["stop_attention_line"],
                ParsingDescColorMap: [192, 192, 64],
            },
        ]
    elif desc_id == "cn_24":
        labels = [
            {
                ParsingDescClassName: ["pothole", "road"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: ["sidewalk"],
                ParsingDescColorMap: [244, 35, 232],
            },
            {
                ParsingDescClassName: ["vegetation"],
                ParsingDescColorMap: [107, 142, 35],
            },
            {
                ParsingDescClassName: ["terrain"],
                ParsingDescColorMap: [152, 251, 152],
            },
            {
                ParsingDescClassName: ["pole"],
                ParsingDescColorMap: [153, 153, 153],
            },
            {
                ParsingDescClassName: ["traffic_sign"],
                ParsingDescColorMap: [220, 220, 0],
            },
            {
                ParsingDescClassName: ["traffic_light"],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: [
                    "sign_line",
                    "diamond",
                    "bicycle_sign",
                    "speed_bump",
                ],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["traffic_lane"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["bicycle", "motorcycle", "tricycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["building"],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["sky"],
                ParsingDescColorMap: [70, 130, 180],
            },
            {
                ParsingDescClassName: ["traffic_cone"],
                ParsingDescColorMap: [0, 64, 64],
            },
            {
                ParsingDescClassName: ["bollard"],
                ParsingDescColorMap: [128, 128, 192],
            },
            {
                ParsingDescClassName: ["traffic_guide_sign"],
                ParsingDescColorMap: [192, 192, 0],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["traffic_arrow"],
                ParsingDescColorMap: [128, 0, 192],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [192, 192, 128],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["slow_down_triangle"],
                ParsingDescColorMap: [102, 102, 156],
            },
            {
                ParsingDescClassName: ["speed_sign"],
                ParsingDescColorMap: [0, 0, 230],
            },
        ]
    elif desc_id == "kr_23":
        labels = [
            {
                ParsingDescClassName: ["road"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "vegetation",
                    "terrain",
                    "building",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["pole"],
                ParsingDescColorMap: [153, 153, 153],
            },
            {
                ParsingDescClassName: ["traffic_sign", "traffic_guide_sign"],
                ParsingDescColorMap: [220, 220, 0],
            },
            {
                ParsingDescClassName: ["traffic_light"],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["sign_line"],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["traffic_lane"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
            {
                ParsingDescClassName: ["tricycle", "motorcycle", "bicycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: ["sky"],
                ParsingDescColorMap: [70, 130, 180],
            },
            {
                ParsingDescClassName: ["traffic_cone"],
                ParsingDescColorMap: [0, 64, 64],
            },
            {
                ParsingDescClassName: ["bollard"],
                ParsingDescColorMap: [128, 128, 192],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["traffic_arrow"],
                ParsingDescColorMap: [128, 0, 192],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [192, 192, 128],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["slow_down_triangle"],
                ParsingDescColorMap: [102, 102, 156],
            },
            {
                ParsingDescClassName: ["speed_sign"],
                ParsingDescColorMap: [0, 0, 230],
            },
            {
                ParsingDescClassName: ["diamond"],
                ParsingDescColorMap: [192, 64, 128],
            },
            {
                ParsingDescClassName: ["bicycle_sign"],
                ParsingDescColorMap: [192, 0, 64],
            },
            {
                ParsingDescClassName: ["speed_bump"],
                ParsingDescColorMap: [244, 35, 232],
            },
        ]
    elif desc_id == "nv_16":
        labels = [
            {
                ParsingDescClassName: ["traffic_lane"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: ["crosswalk_line"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["traffic_arrow"],
                ParsingDescColorMap: [128, 0, 192],
            },
            {
                ParsingDescClassName: ["fence"],
                ParsingDescColorMap: [190, 153, 153],
            },
            {
                ParsingDescClassName: [
                    "sign_line",
                    "slow_down_triangle",
                    "speed_sign",
                    "diamond",
                    "bicycle_sign",
                    "parking_line",
                ],
                ParsingDescColorMap: [200, 200, 128],
            },
            {
                ParsingDescClassName: ["guide_line"],
                ParsingDescColorMap: [0, 192, 192],
            },
            {
                ParsingDescClassName: ["stop_line"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["traffic_cone"],
                ParsingDescColorMap: [0, 64, 64],
            },
            {
                ParsingDescClassName: ["speed_bump"],
                ParsingDescColorMap: [220, 220, 0],
            },
            {
                ParsingDescClassName: [
                    "traffic_light",
                    "traffic_sign",
                    "traffic_guide_sign",
                ],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["pole"],
                ParsingDescColorMap: [153, 153, 153],
            },
            {
                ParsingDescClassName: ["motorcycle", "bicycle", "tricycle"],
                ParsingDescColorMap: [0, 0, 70],
            },
            {
                ParsingDescClassName: ["pothole", "road", "parking_space"],
                ParsingDescColorMap: [128, 64, 128],
            },
            {
                ParsingDescClassName: [
                    "sidewalk",
                    "building",
                    "vegetation",
                    "terrain",
                    "sky",
                    "parking_lock",
                    "column",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
            {
                ParsingDescClassName: ["car", "truck", "bus", "train"],
                ParsingDescColorMap: [0, 0, 142],
            },
            {
                ParsingDescClassName: ["person", "rider"],
                ParsingDescColorMap: [220, 20, 60],
            },
        ]
    elif desc_id == "ces_2":
        labels = [
            {
                ParsingDescClassName: ["building"],
                ParsingDescColorMap: [0, 0, 0],
            },
            {
                ParsingDescClassName: ["person"],
                ParsingDescColorMap: [0, 128, 0],
            },
        ]
    elif desc_id == "apa_8":
        labels = [
            {
                ParsingDescClassName: ["road", "pothole", "sidewalk"],
                ParsingDescColorMap: [0, 0, 0],
            },
            {
                ParsingDescClassName: ["traffic_arrow"],
                ParsingDescColorMap: [79, 79, 47],
            },
            {
                ParsingDescClassName: ["lane_marking", "stop_line"],
                ParsingDescColorMap: [200, 200, 200],
            },
            {
                ParsingDescClassName: [
                    "sign_line",
                    "speed_sign",
                    "slow_down_triangle",
                    "diamond",
                    "bicycle_sign",
                    "speed_bumps",
                    "guide_line",
                    "crosswalk_line",
                ],
                ParsingDescColorMap: [147, 112, 219],
            },
            {
                ParsingDescClassName: ["parking_line"],
                ParsingDescColorMap: [64, 0, 64],
            },
            {
                ParsingDescClassName: ["Parking_space"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["parking_rod", "parking_lock"],
                ParsingDescColorMap: [192, 0, 192],
            },
            {
                ParsingDescClassName: [
                    "vegetation",
                    "terrain",
                    "building",
                    "sky",
                    "person",
                    "rider",
                    "bicycle",
                    "motorcycle",
                    "tricycle",
                    "car",
                    "truck",
                    "bus",
                    "train",
                    "traffic_cone",
                    "bollard",
                    "fence",
                    "traffic_sign",
                    "guide_post",
                    "traffic_light",
                    "pole",
                    "column",
                ],
                ParsingDescColorMap: [70, 70, 70],
            },
        ]
    else:
        raise ValueError("unsupported number of classes %s" % desc_id)
    # check again
    assert len(labels) == int(desc_id.split("_")[-1])
    return labels


def get_parsing_desc_template(
    get_label_fn,
    task_name,
    desc_id,
    roi_regions=None,
    vanishing_point=None,
    merge_fn_type="default",
    merge_fn_params=None,
    input_padding=None,
):

    assert task_name in _parsing_task_list, (
        "Unknown task: %s, register in `_parsing_task_list`" % task_name
    )
    assert desc_id
    num_classes = int(desc_id.split("_")[-1])
    desc = {
        ParsingDescTask: task_name,
        ParsingDescNumClasses: num_classes,
        ParsingDescLabels: get_label_fn(desc_id),
    }
    if input_padding is not None:
        assert (
            isinstance(input_padding, list) and len(input_padding) == 4
        ), "input_padding must be a 4-int list"
        desc.update({"input_padding": input_padding})
    if roi_regions is not None:

        (
            default_merge_fn_type,
            default_merge_fn_params,
        ) = get_seg_default_merge_fn_type_and_params()
        if merge_fn_type is None:
            merge_fn_type = default_merge_fn_type
        if merge_fn_params is None:
            merge_fn_params = default_merge_fn_params
        check_seg_merge_fn_type_and_params(merge_fn_type, merge_fn_params)

        roi_desc = get_roi_desc(
            roi_regions=roi_regions,
            vanishing_point=vanishing_point,
            merge_fn_type=merge_fn_type,
            merge_fn_params=merge_fn_params,
        )

        desc.update(roi_desc)

    return json.dumps(desc)


def get_roi_desc(roi_regions, vanishing_point, merge_fn_type, merge_fn_params):
    def _check():
        assert len(roi_regions) == 4, "x1, y1, x2, y2"
        assert (
            vanishing_point is not None
        ), "vanishing_point is required when roi_regions is not None"
        assert len(vanishing_point) == 2, "x, y"

    _check()

    desc = {
        "roi_input": {
            "fp_x": vanishing_point[0] - roi_regions[0],
            "fp_y": vanishing_point[1] - roi_regions[1],
            "width": roi_regions[2] - roi_regions[0],
            "height": roi_regions[3] - roi_regions[1],
        },
        "vanishing_point": [vanishing_point[0], vanishing_point[1]],
        "merge_fn_desc": {"type": merge_fn_type, "params": merge_fn_params},
    }
    return desc


def _get_image_fail_parsing_labels(desc_id):
    """Get image fail parsing labels by the number of classes.

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
    if num_classes == 5:
        labels = [
            {
                ParsingDescClassName: ["normal"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["light_blur"],
                ParsingDescColorMap: [255, 255, 0],
            },
            {
                ParsingDescClassName: ["heavy_blur"],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["light_glare"],
                ParsingDescColorMap: [128, 64, 192],
            },
            {
                ParsingDescClassName: ["heavy_glare"],
                ParsingDescColorMap: [255, 0, 0],
            },
        ]
    elif num_classes == 7:
        labels = [
            {
                ParsingDescClassName: ["normal"],
                ParsingDescColorMap: [64, 192, 0],
            },
            {
                ParsingDescClassName: ["light_blur"],
                ParsingDescColorMap: [255, 255, 0],
            },
            {
                ParsingDescClassName: ["heavy_blur"],
                ParsingDescColorMap: [250, 170, 30],
            },
            {
                ParsingDescClassName: ["light_glare"],
                ParsingDescColorMap: [128, 64, 192],
            },
            {
                ParsingDescClassName: ["heavy_glare"],
                ParsingDescColorMap: [255, 0, 0],
            },
            {
                ParsingDescClassName: ["light_blockage"],
                ParsingDescColorMap: [0, 0, 255],
            },
            {
                ParsingDescClassName: ["heavy_blockage"],
                ParsingDescColorMap: [0, 0, 0],
            },
        ]
    else:
        raise ValueError("unsupported number of classes %s" % num_classes)
    # check again
    assert len(labels) == num_classes
    return labels


_parsing_colors = np.zeros((256, 1, 3), dtype="uint8")
_parsing_colors[0, :, :] = [128, 64, 128]
_parsing_colors[1, :, :] = [244, 35, 232]
_parsing_colors[2, :, :] = [70, 70, 70]
_parsing_colors[3, :, :] = [200, 200, 200]
_parsing_colors[4, :, :] = [190, 153, 153]
_parsing_colors[5, :, :] = [153, 153, 153]
_parsing_colors[6, :, :] = [250, 170, 30]
_parsing_colors[7, :, :] = [220, 220, 0]
_parsing_colors[8, :, :] = [107, 142, 35]
_parsing_colors[9, :, :] = [152, 251, 152]
_parsing_colors[10, :, :] = [70, 130, 180]
_parsing_colors[11, :, :] = [220, 20, 60]
_parsing_colors[12, :, :] = [255, 0, 0]
_parsing_colors[13, :, :] = [0, 0, 142]
_parsing_colors[14, :, :] = [0, 0, 70]
_parsing_colors[15, :, :] = [0, 60, 100]
_parsing_colors[16, :, :] = [0, 80, 100]
_parsing_colors[17, :, :] = [0, 0, 230]
_parsing_colors[18, :, :] = [119, 11, 32]
_parsing_colors[19, :, :] = [128, 192, 0]
_parsing_colors[20, :, :] = [102, 102, 156]
_parsing_colors[21, :, :] = [200, 200, 128]
_parsing_colors[22, :, :] = [0, 192, 200]
_parsing_colors[23, :, :] = [128, 192, 128]
_parsing_colors[24, :, :] = [64, 64, 0]
_parsing_colors[25, :, :] = [192, 64, 0]
_parsing_colors[26, :, :] = [64, 192, 0]
_parsing_colors[27, :, :] = [192, 192, 0]
_parsing_colors[28, :, :] = [64, 64, 128]
_parsing_colors[29, :, :] = [192, 64, 128]
_parsing_colors[30, :, :] = [64, 192, 128]
_parsing_colors[31, :, :] = [192, 192, 128]
_parsing_colors[32, :, :] = [0, 0, 64]
_parsing_colors[33, :, :] = [128, 0, 64]
_parsing_colors[34, :, :] = [0, 128, 64]
_parsing_colors[35, :, :] = [128, 128, 64]
_parsing_colors[36, :, :] = [0, 0, 192]
_parsing_colors[37, :, :] = [128, 0, 192]
_parsing_colors[38, :, :] = [0, 128, 192]
_parsing_colors[39, :, :] = [128, 128, 192]
_parsing_colors[40, :, :] = [64, 0, 64]
_parsing_colors[41, :, :] = [192, 0, 64]
_parsing_colors[42, :, :] = [64, 128, 64]
_parsing_colors[43, :, :] = [192, 128, 64]
_parsing_colors[44, :, :] = [64, 0, 192]
_parsing_colors[45, :, :] = [192, 0, 192]
_parsing_colors[46, :, :] = [64, 128, 192]
_parsing_colors[47, :, :] = [192, 128, 192]
_parsing_colors[48, :, :] = [0, 64, 64]
_parsing_colors[49, :, :] = [128, 64, 64]
_parsing_colors[50, :, :] = [0, 192, 64]
_parsing_colors[51, :, :] = [128, 192, 64]
_parsing_colors[52, :, :] = [0, 64, 192]
_parsing_colors[53, :, :] = [128, 64, 192]
_parsing_colors[54, :, :] = [0, 192, 192]
_parsing_colors[55, :, :] = [128, 192, 192]
_parsing_colors[56, :, :] = [64, 64, 64]
_parsing_colors[57, :, :] = [192, 64, 64]
_parsing_colors[58, :, :] = [64, 192, 64]
_parsing_colors[59, :, :] = [192, 192, 64]
_parsing_colors[60, :, :] = [64, 64, 192]
_parsing_colors[61, :, :] = [192, 64, 192]
_parsing_colors[62, :, :] = [64, 192, 192]
_parsing_colors[63, :, :] = [192, 192, 192]
_parsing_colors[64, :, :] = [32, 0, 0]
_parsing_colors[65, :, :] = [160, 0, 0]
_parsing_colors[66, :, :] = [32, 128, 0]
_parsing_colors[67, :, :] = [160, 128, 0]
_parsing_colors[68, :, :] = [32, 0, 128]
_parsing_colors[69, :, :] = [160, 0, 128]
_parsing_colors[70, :, :] = [32, 128, 128]
_parsing_colors[71, :, :] = [160, 128, 128]
_parsing_colors[72, :, :] = [96, 0, 0]
_parsing_colors[73, :, :] = [224, 0, 0]
_parsing_colors[74, :, :] = [96, 128, 0]
_parsing_colors[75, :, :] = [224, 128, 0]
_parsing_colors[76, :, :] = [96, 0, 128]
_parsing_colors[77, :, :] = [224, 0, 128]
_parsing_colors[78, :, :] = [96, 128, 128]
_parsing_colors[79, :, :] = [224, 128, 128]
_parsing_colors[80, :, :] = [32, 64, 0]
_parsing_colors[81, :, :] = [160, 64, 0]
_parsing_colors[82, :, :] = [32, 192, 0]
_parsing_colors[83, :, :] = [160, 192, 0]
_parsing_colors[84, :, :] = [32, 64, 128]
_parsing_colors[85, :, :] = [160, 64, 128]
_parsing_colors[86, :, :] = [32, 192, 128]
_parsing_colors[87, :, :] = [160, 192, 128]
_parsing_colors[88, :, :] = [96, 64, 0]
_parsing_colors[89, :, :] = [224, 64, 0]
_parsing_colors[90, :, :] = [96, 192, 0]
_parsing_colors[91, :, :] = [224, 192, 0]
_parsing_colors[92, :, :] = [96, 64, 128]
_parsing_colors[93, :, :] = [224, 64, 128]
_parsing_colors[94, :, :] = [96, 192, 128]
_parsing_colors[95, :, :] = [224, 192, 128]
_parsing_colors[96, :, :] = [32, 0, 64]
_parsing_colors[97, :, :] = [160, 0, 64]
_parsing_colors[98, :, :] = [32, 128, 64]
_parsing_colors[99, :, :] = [160, 128, 64]
_parsing_colors[100, :, :] = [32, 0, 192]
_parsing_colors[101, :, :] = [160, 0, 192]
_parsing_colors[102, :, :] = [32, 128, 192]
_parsing_colors[103, :, :] = [160, 128, 192]
_parsing_colors[104, :, :] = [96, 0, 64]
_parsing_colors[105, :, :] = [224, 0, 64]
_parsing_colors[106, :, :] = [96, 128, 64]
_parsing_colors[107, :, :] = [224, 128, 64]
_parsing_colors[108, :, :] = [96, 0, 192]
_parsing_colors[109, :, :] = [224, 0, 192]
_parsing_colors[110, :, :] = [96, 128, 192]
_parsing_colors[111, :, :] = [224, 128, 192]
_parsing_colors[112, :, :] = [32, 64, 64]
_parsing_colors[113, :, :] = [160, 64, 64]
_parsing_colors[114, :, :] = [32, 192, 64]
_parsing_colors[115, :, :] = [160, 192, 64]
_parsing_colors[116, :, :] = [32, 64, 192]
_parsing_colors[117, :, :] = [160, 64, 192]
_parsing_colors[118, :, :] = [32, 192, 192]
_parsing_colors[119, :, :] = [160, 192, 192]
_parsing_colors[120, :, :] = [96, 64, 64]
_parsing_colors[121, :, :] = [224, 64, 64]
_parsing_colors[122, :, :] = [96, 192, 64]
_parsing_colors[123, :, :] = [224, 192, 64]
_parsing_colors[124, :, :] = [96, 64, 192]
_parsing_colors[125, :, :] = [224, 64, 192]
_parsing_colors[126, :, :] = [96, 192, 192]
_parsing_colors[127, :, :] = [224, 192, 192]
_parsing_colors[128, :, :] = [0, 32, 0]
_parsing_colors[129, :, :] = [128, 32, 0]
_parsing_colors[130, :, :] = [0, 160, 0]
_parsing_colors[131, :, :] = [128, 160, 0]
_parsing_colors[132, :, :] = [0, 32, 128]
_parsing_colors[133, :, :] = [128, 32, 128]
_parsing_colors[134, :, :] = [0, 160, 128]
_parsing_colors[135, :, :] = [128, 160, 128]
_parsing_colors[136, :, :] = [64, 32, 0]
_parsing_colors[137, :, :] = [192, 32, 0]
_parsing_colors[138, :, :] = [64, 160, 0]
_parsing_colors[139, :, :] = [192, 160, 0]
_parsing_colors[140, :, :] = [64, 32, 128]
_parsing_colors[141, :, :] = [192, 32, 128]
_parsing_colors[142, :, :] = [64, 160, 128]
_parsing_colors[143, :, :] = [192, 160, 128]
_parsing_colors[144, :, :] = [0, 96, 0]
_parsing_colors[145, :, :] = [128, 96, 0]
_parsing_colors[146, :, :] = [0, 224, 0]
_parsing_colors[147, :, :] = [128, 224, 0]
_parsing_colors[148, :, :] = [0, 96, 128]
_parsing_colors[149, :, :] = [128, 96, 128]
_parsing_colors[150, :, :] = [0, 224, 128]
_parsing_colors[151, :, :] = [128, 224, 128]
_parsing_colors[152, :, :] = [64, 96, 0]
_parsing_colors[153, :, :] = [192, 96, 0]
_parsing_colors[154, :, :] = [64, 224, 0]
_parsing_colors[155, :, :] = [192, 224, 0]
_parsing_colors[156, :, :] = [64, 96, 128]
_parsing_colors[157, :, :] = [192, 96, 128]
_parsing_colors[158, :, :] = [64, 224, 128]
_parsing_colors[159, :, :] = [192, 224, 128]
_parsing_colors[160, :, :] = [0, 32, 64]
_parsing_colors[161, :, :] = [128, 32, 64]
_parsing_colors[162, :, :] = [0, 160, 64]
_parsing_colors[163, :, :] = [128, 160, 64]
_parsing_colors[164, :, :] = [0, 32, 192]
_parsing_colors[165, :, :] = [128, 32, 192]
_parsing_colors[166, :, :] = [0, 160, 192]
_parsing_colors[167, :, :] = [128, 160, 192]
_parsing_colors[168, :, :] = [64, 32, 64]
_parsing_colors[169, :, :] = [192, 32, 64]
_parsing_colors[170, :, :] = [64, 160, 64]
_parsing_colors[171, :, :] = [192, 160, 64]
_parsing_colors[172, :, :] = [64, 32, 192]
_parsing_colors[173, :, :] = [192, 32, 192]
_parsing_colors[174, :, :] = [64, 160, 192]
_parsing_colors[175, :, :] = [192, 160, 192]
_parsing_colors[176, :, :] = [0, 96, 64]
_parsing_colors[177, :, :] = [128, 96, 64]
_parsing_colors[178, :, :] = [0, 224, 64]
_parsing_colors[179, :, :] = [128, 224, 64]
_parsing_colors[180, :, :] = [0, 96, 192]
_parsing_colors[181, :, :] = [128, 96, 192]
_parsing_colors[182, :, :] = [0, 224, 192]
_parsing_colors[183, :, :] = [128, 224, 192]
_parsing_colors[184, :, :] = [64, 96, 64]
_parsing_colors[185, :, :] = [192, 96, 64]
_parsing_colors[186, :, :] = [64, 224, 64]
_parsing_colors[187, :, :] = [192, 224, 64]
_parsing_colors[188, :, :] = [64, 96, 192]
_parsing_colors[189, :, :] = [192, 96, 192]
_parsing_colors[190, :, :] = [64, 224, 192]
_parsing_colors[191, :, :] = [192, 224, 192]
_parsing_colors[192, :, :] = [32, 32, 0]
_parsing_colors[193, :, :] = [160, 32, 0]
_parsing_colors[194, :, :] = [32, 160, 0]
_parsing_colors[195, :, :] = [160, 160, 0]
_parsing_colors[196, :, :] = [32, 32, 128]
_parsing_colors[197, :, :] = [160, 32, 128]
_parsing_colors[198, :, :] = [32, 160, 128]
_parsing_colors[199, :, :] = [160, 160, 128]
_parsing_colors[200, :, :] = [96, 32, 0]
_parsing_colors[201, :, :] = [224, 32, 0]
_parsing_colors[202, :, :] = [96, 160, 0]
_parsing_colors[203, :, :] = [224, 160, 0]
_parsing_colors[204, :, :] = [96, 32, 128]
_parsing_colors[205, :, :] = [224, 32, 128]
_parsing_colors[206, :, :] = [96, 160, 128]
_parsing_colors[207, :, :] = [224, 160, 128]
_parsing_colors[208, :, :] = [32, 96, 0]
_parsing_colors[209, :, :] = [160, 96, 0]
_parsing_colors[210, :, :] = [32, 224, 0]
_parsing_colors[211, :, :] = [160, 224, 0]
_parsing_colors[212, :, :] = [32, 96, 128]
_parsing_colors[213, :, :] = [160, 96, 128]
_parsing_colors[214, :, :] = [32, 224, 128]
_parsing_colors[215, :, :] = [160, 224, 128]
_parsing_colors[216, :, :] = [96, 96, 0]
_parsing_colors[217, :, :] = [224, 96, 0]
_parsing_colors[218, :, :] = [96, 224, 0]
_parsing_colors[219, :, :] = [224, 224, 0]
_parsing_colors[220, :, :] = [96, 96, 128]
_parsing_colors[221, :, :] = [224, 96, 128]
_parsing_colors[222, :, :] = [96, 224, 128]
_parsing_colors[223, :, :] = [224, 224, 128]
_parsing_colors[224, :, :] = [32, 32, 64]
_parsing_colors[225, :, :] = [160, 32, 64]
_parsing_colors[226, :, :] = [32, 160, 64]
_parsing_colors[227, :, :] = [160, 160, 64]
_parsing_colors[228, :, :] = [32, 32, 192]
_parsing_colors[229, :, :] = [160, 32, 192]
_parsing_colors[230, :, :] = [32, 160, 192]
_parsing_colors[231, :, :] = [160, 160, 192]
_parsing_colors[232, :, :] = [96, 32, 64]
_parsing_colors[233, :, :] = [224, 32, 64]
_parsing_colors[234, :, :] = [96, 160, 64]
_parsing_colors[235, :, :] = [224, 160, 64]
_parsing_colors[236, :, :] = [96, 32, 192]
_parsing_colors[237, :, :] = [224, 32, 192]
_parsing_colors[238, :, :] = [96, 160, 192]
_parsing_colors[239, :, :] = [224, 160, 192]
_parsing_colors[240, :, :] = [32, 96, 64]
_parsing_colors[241, :, :] = [160, 96, 64]
_parsing_colors[242, :, :] = [32, 224, 64]
_parsing_colors[243, :, :] = [160, 224, 64]
_parsing_colors[244, :, :] = [32, 96, 192]
_parsing_colors[245, :, :] = [160, 96, 192]
_parsing_colors[246, :, :] = [32, 224, 192]
_parsing_colors[247, :, :] = [160, 224, 192]
_parsing_colors[248, :, :] = [96, 96, 64]
_parsing_colors[249, :, :] = [224, 96, 64]
_parsing_colors[250, :, :] = [96, 224, 64]
_parsing_colors[251, :, :] = [224, 224, 64]
_parsing_colors[252, :, :] = [96, 96, 192]
_parsing_colors[253, :, :] = [224, 96, 192]
_parsing_colors[254, :, :] = [255, 255, 255]
_parsing_colors[255, :, :] = [0, 0, 0]
