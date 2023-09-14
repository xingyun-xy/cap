# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.

import pathlib


class RootHelper(object):
    """根目录辅助类.

    提供接口可以快速获取到 CAP 下的一些特定目录.
    """

    # 获取CAP包的根目录地址
    # __file__ : .../CAP/cap/utils/root_helper.py
    # parents  :        2   1     0
    CAP_ROOT: str = str(pathlib.Path(__file__).parents[2])

    @staticmethod
    def get_project_root(name: str) -> str:
        """获取各个项目的根目录.

        Args:
            name: 项目名

        Returns:
            项目名对应的项目的根目录
        """
        root = pathlib.Path(RootHelper.CAP_ROOT).joinpath("projects", name)
        if not root.exists():
            raise FileNotFoundError(f"{root}并不存在, 请检查你指定的项目名是否正确")
        return str(root)
