from collections import OrderedDict

dst_label = OrderedDict()
dst_label["dashed"] = 1
dst_label["solid"] = 1
dst_label["mixed"] = 1
dst_label["wide_dashed"] = 4
dst_label["wide_solid"] = 1
dst_label["deceleration_lane"] = 1
dst_label["tidal_lane"] = 1
dst_label["Road_teeth"] = 2
dst_label["double_line"] = 3
dst_label["wide_lane"] = 4

dst_label["other"] = 255
dst_label["ignore"] = 255


src_label = OrderedDict()
src_label["wide_lane"] = 4
src_label["Road_teeth"] = 33
src_label["dashed"] = 34
src_label["solid"] = 35
src_label["mixed"] = 36
src_label["wide_dashed"] = 37
src_label["wide_solid"] = 38
src_label["deceleration_lane"] = 39
src_label["tidal_lane"] = 40
src_label["double_line"] = 41
src_label["other"] = 255
src_label["ignore"] = 255


dst_label_map = OrderedDict()
dst_label_map["wide_lane"] = {
    "match_conditions": [{"type": ["wide_dashed"], "ignore": ["no"]}],
    "id": 4,
}
dst_label_map["double_line"] = {
    "match_conditions": [
        {
            "double_line": ["yes"],
            "type": [
                "dashed",
                "solid",
                "mixed",
                "deceleration_lane",
                "tidal_lane",
                "wide_solid",
            ],
            "ignore": ["no"],
        }
    ],
    "id": 3,
}
dst_label_map["curb"] = {
    "match_conditions": [{"type": ["Road_teeth"], "ignore": ["no"]}],
    "id": 2,
}
dst_label_map["lane"] = {
    "match_conditions": [
        {
            "double_line": ["no"],
            "type": [
                "dashed",
                "solid",
                "mixed",
                "deceleration_lane",
                "tidal_lane",
                "wide_solid",
            ],
            "ignore": ["no"],
        }
    ],
    "id": 1,
}
dst_label_map["ignore"] = {
    "match_conditions": [{"ignore": ["yes"]}],
    "id": 255,
}


color_map = OrderedDict()  # 5 cls
color_map["road"] = [0, 0, 0]
color_map["dashed"] = [0, 0, 255]
color_map["Road_teeth"] = [0, 255, 0]
color_map["double_line"] = [255, 255, 0]
color_map["wide_dashed"] = [190, 153, 153]
# lane parsing task 'other' class default is [255, 0, 0]
