# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.

import logging
import math
from typing import Tuple

import h5py
import numpy as np

from cap.registry import OBJECT_REGISTRY

SUBTYPE_2_DTYPE = {
    "PCM_16": "int16",
    "PCM_S8": "int8",
    "PCM_32": "float32",
    "DOUBLE": "float64",
}


def convert_waveform_to_float(
    data: np.ndarray, dtype: str = "float32"
) -> np.ndarray:
    """convert_waveform_to_float.

    将waveform转换成float类型.由于waveform读取出来可能是int16或者int8类型的数据。
    该接口将其转换为 float 类型的数组


    Args:
        data: 需要处理的waveform数组，当数组是浮点数组时不处理。
              当数组是int型数组时，所属int型的取值空间转换到[-1, 1]之间的浮点型
        dtype: 目标 float 数组，支持np.floating的所有float类型.默认是："float32".
    """
    assert np.issubdtype(dtype, np.floating), "dtype is not an ``np.floating``"
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(dtype)
    elif np.issubdtype(data.dtype, np.signedinteger):
        iinfo = np.iinfo(data.dtype)
        data = data.astype(dtype) / (-iinfo.min)
    else:
        raise NotImplementedError(f"{data.dtype} is not support now!")
    return data


class BaseWaveformReader(object):
    def read_seg_audio(  # type: ignore[no-untyped-def]
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, int]:
        """所有子类都应该实现该接口."""
        raise NotImplementedError


@OBJECT_REGISTRY.register
class HDF5WaveformReader(BaseWaveformReader):

    group = "audio"
    _SUPPORT_DTYPE = ["int16", "int32", "float32", "float64"]

    def __init__(
        self,
        hdf5_file: str,
        ret_dtype: str = "float32",
        channel_first: bool = True,
    ):
        self.hdf5_file = hdf5_file
        assert (
            ret_dtype in HDF5WaveformReader._SUPPORT_DTYPE
        ), f"Only support {HDF5WaveformReader._SUPPORT_DTYPE}, but get ``{ret_dtype}``"  # noqa: E501
        self.ret_dtype = ret_dtype
        self.channel_first = channel_first
        self._fork()

    def _fork(self):  # type: ignore
        if getattr(self, "hdf5_fileno", None) is None:
            self.hdf5_fileno = h5py.File(self.hdf5_file, "r")

    def read_seg_audio(  # type: ignore[override]
        self, utt: str, seg_beg: float, seg_end: float
    ) -> Tuple[np.ndarray, int]:
        logging.debug(f"seg_info: {seg_beg} {seg_end}")
        # 获取数据
        seg_beg_sec = math.floor(seg_beg)
        seg_end_sec = math.ceil(seg_end)
        logging.debug(f"sec index: {seg_beg_sec} {seg_end_sec}")
        waveform = self.hdf5_fileno[self.group][utt][seg_beg_sec:seg_end_sec]
        # 读取音频信息并变形
        attrs = self.hdf5_fileno[self.group][utt].attrs
        channels = attrs["channels"]
        samplerate = attrs["samplerate"]
        waveform = [wf.reshape((-1, channels)) for wf in waveform]
        waveform = np.concatenate(waveform, axis=0)
        # 截取数据
        seg_beg_idx = int(samplerate * (seg_beg - seg_beg_sec))
        duration = attrs["duration"]
        seg_end_sec = min(duration, seg_end_sec)
        if seg_end_sec < seg_end:
            msg = f"{utt}:{seg_beg}:{seg_end} with duration {duration} maybe error label"  # noqa: E501
            logging.warning(msg)
            seg_end_sec = seg_end
        _seg_end_idx = int(samplerate * (seg_end_sec - seg_end))
        seg_end_idx = waveform.shape[0] - _seg_end_idx
        logging.debug(f"seg index: {seg_beg_idx} {seg_end_idx}")
        waveform = waveform[seg_beg_idx:seg_end_idx]
        # 转成目标dtype
        if self.ret_dtype in ["float32", "float64"]:
            waveform = convert_waveform_to_float(waveform, self.ret_dtype)
        # 转换 channel 的维度位置
        if self.channel_first:
            waveform = waveform.transpose(1, 0)
        return waveform, samplerate

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["hdf5_fileno"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()
