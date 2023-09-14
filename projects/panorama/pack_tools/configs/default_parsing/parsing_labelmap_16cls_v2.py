from collections import OrderedDict

from cap.core.proj_spec.parsing import _parsing_colors

src_label = OrderedDict()  # 38 cls
src_label["road"] = 0
src_label["pothole"] = 0
src_label["sidewalk"] = 1
src_label["vegetation"] = 2
src_label["terrain"] = 3
src_label["pole"] = 4
src_label["traffic_sign"] = 5
src_label["Traffic_Sign1"] = 5
src_label["traffic_light"] = 6
src_label["Sign_Line"] = 7
src_label["lane_marking"] = 8
src_label["person"] = 9
src_label["rider"] = 10
src_label["bicycle"] = 11
src_label["motorcycle"] = 12
src_label["tricycle"] = 13
src_label["car"] = 14
src_label["truck"] = 15
src_label["bus"] = 16
src_label["train"] = 17
src_label["building"] = 18
src_label["fence"] = 19
src_label["sky"] = 20
src_label["Traffic_Cone"] = 21
src_label["Bollard"] = 22
src_label["Guide_Post"] = 23
src_label["Crosswalk_Line"] = 24
src_label["Traffic_Arrow"] = 25
src_label["Guide_Line"] = 26
src_label["Stop_Line"] = 27
src_label["Slow_Down_Triangle"] = 28
src_label["Speed_Sign"] = 29
src_label["Diamond"] = 30
src_label["BicycleSign"] = 31
src_label["SpeedBumps"] = 32
# add no_forward_marker
src_label["no_forward_marker"] = 33

# apa sign cls
src_label["parking_line"] = 7
src_label["Parking_space"] = 0
src_label["parking_rod"] = 34
src_label["parking_lock"] = 35
src_label["column"] = 18
# add genObj
src_label["traversable_obstruction"] = 36
src_label["untraversable_obstruction"] = 37
src_label["Road_teeth"] = 255
src_label["mask"] = 255
src_label["other"] = 255


dst_label = OrderedDict()  # 16.2 cls
dst_label["road"] = 0
dst_label["sidewalk"] = 1
dst_label["building"] = 1
dst_label["pothole"] = 0
dst_label["fence"] = 2
dst_label["pole"] = 3
dst_label["traffic_light"] = 4
dst_label["Traffic_Sign1"] = 4
dst_label["traffic_sign"] = 4
dst_label["vegetation"] = 1
dst_label["terrain"] = 1
dst_label["sky"] = 1
dst_label["person"] = 5
dst_label["rider"] = 5
dst_label["car"] = 6
dst_label["truck"] = 6
dst_label["bus"] = 6
dst_label["train"] = 6
dst_label["motorcycle"] = 7
dst_label["bicycle"] = 7
# ----------------
dst_label["tricycle"] = 6
dst_label["lane_marking"] = 8

dst_label["Guide_Post"] = 4
dst_label["Crosswalk_Line"] = 9
dst_label["Traffic_Arrow"] = 10
dst_label["Sign_Line"] = 11
dst_label["Guide_Line"] = 12
dst_label["Traffic_Cone"] = 13
dst_label["Bollard"] = 13
# ------------------
dst_label["Stop_Line"] = 14
dst_label["Slow_Down_Triangle"] = 11
dst_label["Speed_Sign"] = 11

dst_label["Diamond"] = 11
dst_label["BicycleSign"] = 11
dst_label["SpeedBumps"] = 15

# apa sign cls
dst_label["parking_line"] = 11
dst_label["Parking_space"] = 0
dst_label["parking_rod"] = 1
dst_label["parking_lock"] = 1
dst_label["column"] = 1

dst_label["no_forward_marker"] = 1
dst_label["traversable_obstruction"] = 0
dst_label["untraversable_obstruction"] = 1

dst_label["Road_teeth"] = 255
dst_label["mask"] = 255

dst_label["other"] = 255


color_map_idx = OrderedDict()  # 12.5 cls
color_map_idx["road"] = 0
color_map_idx["background"] = 2
color_map_idx["fence"] = 4
color_map_idx["pole"] = 5
color_map_idx["traffic"] = 6
color_map_idx["person"] = 11
color_map_idx["vehicle"] = 13
color_map_idx["two-wheel"] = 14
color_map_idx["lane_marking"] = 3
color_map_idx["crosswalk"] = 26
color_map_idx["traffic_arrow"] = 37
color_map_idx["sign_line"] = 21
color_map_idx["guide_line"] = 54
color_map_idx["cone"] = 48
color_map_idx["stop_line"] = 12
color_map_idx["speed_bump"] = 7

color_map = OrderedDict()  # 16.2 cls
for key, value in color_map_idx.items():
    color_map[key] = _parsing_colors[value, :, :]
