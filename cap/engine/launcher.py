# Copyright (c) Changan Auto. All rights reserved.

# different trainers need different launcher
LaunchMap = {}


def register_launcher(trainer_type, launcher):
    LaunchMap[trainer_type] = launcher


def get_launcher(trainer_type):
    return LaunchMap[trainer_type]


def build_launcher(trainer):
    return LaunchMap[trainer["type"]]
