# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.

import itertools
import logging
import math
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from cap.data.datasets.carp.redis_client import (
    TendisClient,
    multimodal_default_rename_handle,
)
from cap.data.datasets.utils import decode_img
from cap.registry import OBJECT_REGISTRY
from cap.utils.data_search import find_closest_element

logger = logging.getLogger(__name__)


class BaseImageReader(object):
    """BaseImageReader.

    图片 read 的基类, 补充缺失图片的基本方法。子类需要实现 read_seg_images 接口

    Args:
        missing_image_complete_mode : 补全丢失图片的模式，支持如下格式
            - "zero": 用0补全
            - "nearest" 用最近邻的帧补全
            - "left" 用左侧帧补全
            - "right" 用右侧帧补全
        filter_rules : 数据过滤策略，当满足数据过滤的策略时需要删除数据
    """

    def __init__(
        self,
        missing_image_complete_mode: str = "zero",
        filter_rules: Mapping = None,
    ):
        self._missing_image_complete_mode = missing_image_complete_mode
        self.filter_rules = filter_rules

    def read_seg_images(  # type: ignore[no-untyped-def]
        self, *args, **kwargs
    ) -> Optional[List[np.ndarray]]:
        """所有子类应该实现该接口."""
        raise NotImplementedError

    def complete_missing_image(
        self,
        raw_img_lst: Sequence[Optional[np.ndarray]],
        log_utt: str = None,
        logger: logging.Logger = logger,
    ) -> Optional[List[np.ndarray]]:
        if self.is_filtered(raw_img_lst):
            return None
        # complete missing images
        avilable_idx_lst = [
            idx for idx, img in enumerate(raw_img_lst) if img is not None
        ]
        # logging warning missing images
        if len(avilable_idx_lst) != len(raw_img_lst):
            missing_num = len(raw_img_lst) - len(avilable_idx_lst)
            msg = f"{log_utt} has {missing_num}/{len(raw_img_lst)} images missing."  # noqa: E501
            logger.info(msg)
        # zero mode: complete by insert zero array
        if self._missing_image_complete_mode == "zero":
            if not hasattr(self, "eg_img"):
                try:
                    self.eg_img = raw_img_lst[avilable_idx_lst[0]]
                except Exception:
                    self.eg_img = np.zeros((96, 96, 3), dtype=np.uint8)
            ret_img_lst = [
                np.zeros_like(self.eg_img) + 128 if img is None else img
                for img in raw_img_lst
            ]
        # nearest,left,right mode: complete by the could found
        # nearest, left_nearest, right_nearest image.
        # the missing in edge will be complete by the available edge image.
        else:
            if self._missing_image_complete_mode == "nearest":
                left, right = False, False
            elif self._missing_image_complete_mode == "left":
                left, right = True, False
            elif self._missing_image_complete_mode == "right":
                left, right = False, True
            else:
                raise NotImplementedError(
                    "未知的:``missing_image_complete_mode``"
                )

            def _find_ret_image(idx: int) -> np.ndarray:
                if idx in avilable_idx_lst:
                    return raw_img_lst[idx]
                if idx < avilable_idx_lst[0]:
                    ret_idx = avilable_idx_lst[0]
                elif idx > avilable_idx_lst[-1]:
                    ret_idx = avilable_idx_lst[-1]
                else:
                    ret_idx, _ = find_closest_element(  # type: ignore
                        idx,
                        avilable_idx_lst,
                        closest_left=left,
                        closest_right=right,
                    )
                return np.copy(raw_img_lst[ret_idx])

            ret_img_lst = [
                _find_ret_image(idx) for idx, _ in enumerate(raw_img_lst)
            ]
        return ret_img_lst

    def is_filtered(self, img_lst: Sequence[np.ndarray]) -> bool:
        help_lst = [1 if img is not None else None for img in img_lst]
        if self.filter_rules is not None:
            for key, item in self.filter_rules.items():
                if key == "continuous_missing_count":
                    tmp_lst = [
                        len(list(g))
                        for k, g in itertools.groupby(help_lst)
                        if k is None
                    ]
                    max_cnt = 0 if len(tmp_lst) == 0 else max(tmp_lst)
                    if max_cnt > item["threshold"]:
                        return True
                elif key == "missing_ratio":
                    cnt = len([f for f in help_lst if f is None])
                    if float(cnt) / len(help_lst) > item["threshold"]:
                        return True
                elif key == "missing_count":
                    cnt = len([f for f in help_lst if f is None])
                    if cnt > item["threshold"]:
                        return True
                else:
                    logger.error(f"Unknown filter rule: {key}")
                    raise NotImplementedError
        return False


@OBJECT_REGISTRY.register
class TendisJ2MMVideoImageReader(BaseImageReader):

    KEY_HEAD = "Key2Img"

    def __init__(
        self,
        fps: int = 30,
        origin_fps: int = None,
        zfill_num: int = 6,
        missing_image_complete_mode: str = "zero",
        filter_rules: Mapping = None,
        tendis_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super(TendisJ2MMVideoImageReader, self).__init__(
            missing_image_complete_mode=missing_image_complete_mode,
            filter_rules=filter_rules,
        )
        self._origin_fps = fps if origin_fps is None else origin_fps
        self._fps = fps
        self._zfill_num = zfill_num
        self.tendis_kwargs = {} if tendis_kwargs is None else tendis_kwargs
        self._fork()

    def _fork(self):  # type: ignore
        if getattr(self, "db_client", None) is None:
            self.db_client = TendisClient(
                rename_handle=multimodal_default_rename_handle,
                **self.tendis_kwargs,
            )

    def read_seg_images(  # type: ignore[override]
        self,
        utt: str,
        seg_beg: float,
        seg_end: float,
        logger: logging.Logger = logger,
    ) -> Optional[List[np.ndarray]]:
        """read_seg_images.

        读取片段对应的图片列表

        Args:
            utt : 片段对应视频的 utt_id, 格式是: '${batch_name}_${video_id}'
                  示例：'J2mm_batch01_00154_02_03_cam2_000063'
            seg_beg : 片段的开始时间点，单位为秒
            seg_end : 片段的结束时间点，单位为秒

        Returns:
            List[np.ndarray] 返回片段对应的图片列表
        """
        # utt: [0-9]{5}_[0-9]{2}_[0-9]{2}_cam[0-9]*
        # 计算在图片保存fps下片段的图片起止序号
        seg_beg_idx = math.ceil(seg_beg * self._origin_fps)
        seg_end_idx = math.ceil(seg_end * self._origin_fps)
        # 生成当前段log信息的标签
        log_utt = (
            f"{self.KEY_HEAD}_{utt}:"
            + f"{seg_beg_idx :>0{self._zfill_num}d}:"
            + f"{seg_end_idx :>0{self._zfill_num}d}"
        )
        # 初始化图片序列的key值
        key_lst = [
            f"{self.KEY_HEAD}_{utt}_{idx :>0{self._zfill_num}d}"
            for idx in range(seg_beg_idx, seg_end_idx)
        ]
        # 获取和解码图片
        raw_buff_lst = self.db_client.mget(key_lst)
        raw_img_lst = [
            None if buff is None else decode_img(buff) for buff in raw_buff_lst
        ]
        check_iter = zip(raw_buff_lst, raw_img_lst)
        if any((buff is not None and img is None) for buff, img in check_iter):
            logger.warning(
                f"{log_utt} maybe use `mxnet.recordio.pack_img` format,"
                "please check and rewrite to ``cv2.imencode`` format"
            )
        # 将图片列表冲构成目标fps
        if self._fps != self._origin_fps:
            seg_beg_idx = math.ceil(seg_beg * self._fps)
            seg_end_idx = math.ceil(seg_end * self._fps)
            target_count = seg_end_idx - seg_beg_idx
            raw_img_lst = [
                raw_img_lst[int(idx)]
                for idx in np.linspace(
                    0, len(raw_img_lst), target_count, endpoint=False
                )
            ]
        ret_img_lst = self.complete_missing_image(
            raw_img_lst,
            log_utt=log_utt,
            logger=logger,
        )
        return ret_img_lst

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["db_client"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()
