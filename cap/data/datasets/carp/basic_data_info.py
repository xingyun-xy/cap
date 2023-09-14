# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.


import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from easydict import EasyDict

from cap.data.datasets.carp.redis_client import (
    TendisClient,
    multimodal_default_rename_handle,
)
from cap.registry import OBJECT_REGISTRY


class BasicDataInfo(object):
    """BasicDataInfo.

    鸡尾酒(多模)算法数据的基础标签信息，是对json标签抽象类。详细的json结构参见：


    Args:
        json_line: 标签信息对应的一行json字符串

    """

    def __init__(self, json_line: str):
        self.data_info = EasyDict(json.loads(json_line))

    def get_audio_length(self) -> Optional[float]:
        for key in self.data_info.audio.keys():
            return float(self.data_info.audio[key].duration)
        return None

    def get_batch_name(self) -> str:
        return self.data_info.basic.batch

    def get_wav_path_list(self) -> List[str]:
        wav_path_list = []
        for key in self.data_info.audio.keys():
            wav_path_list.append(self.data_info.audio[key].wav_path)

        return wav_path_list

    def get_wav_path(self, micid: str) -> str:
        return self.data_info.audio[micid].wav_path

    def get_video_id(self) -> Optional[str]:
        for key in self.data_info.audio.keys():
            wav_name = self.data_info.audio[key].wav_name
            video_id = "_".join(wav_name.split("_")[:-1])
            return video_id
        return None

    def get_uttid(self) -> List[str]:
        uttids = []
        for key in self.data_info.audio.keys():
            uttids.append(self.data_info.audio[key].uttid)
        return uttids

    def get_spk_id(self) -> str:
        return self.data_info.speaker.spkid

    def get_vad_segment_list(self) -> List[Tuple[float, float]]:
        vad_segment_list = []
        for vad_dict in self.data_info.label.sentence:
            vad_start = vad_dict["vad_beg"]
            vad_end = vad_dict["vad_end"]
            vad_segment_list.append((vad_start, vad_end))
        return vad_segment_list

    def get_label_list(self) -> List[Tuple[float, float, str]]:
        label_list = []
        for vad_dict in self.data_info.label.sentence:
            vad_start = vad_dict["vad_beg"]
            vad_end = vad_dict["vad_end"]
            label = vad_dict["text"]
            label_list.append((vad_start, vad_end, label))
        return label_list

    def get_sentence_content(self) -> List[Dict[str, Union[str, float]]]:
        return self.data_info.label.sentence

    def get_sentence_id(self) -> str:
        """get_sentence_id.

        这段json数据的唯一值
        错误输出: J2MM_develop_speech_batch04-04412_37_01_mic1
        正确输出: J2MM_develop_speech_batch04-04412_37_01
        """
        batch_name = self.get_batch_name()
        video_id = self.get_video_id()
        return f"{batch_name}-{video_id}"

    def get_seat_role(self) -> Optional[str]:
        role = self.data_info.scene.attrs.spk_pos
        if role == "1L":
            return "pilot"
        elif role == "1R":
            return "copilot"
        else:
            NotImplementedError
        return None

    def get_gender(self) -> str:
        return self.data_info.speaker.gender

    def get_available_cam_ids(self) -> List[str]:
        return list(self.data_info.video.keys())

    def get_available_mic_ids(self) -> List[str]:
        return list(self.data_info.audio.keys())

    def get_video_info(self, cam_id: str) -> str:
        return self.data_info.video[cam_id]


@OBJECT_REGISTRY.register
class TendisBasicInfoReader(object):

    KEY_HEAD = "Key2Label"

    def __init__(
        self,
        tendis_kwargs: Optional[Mapping[str, Any]] = None,
    ):  # type: ignore[no-untyped-def]
        self.tendis_kwargs = {} if tendis_kwargs is None else tendis_kwargs
        self._fork()

    def _fork(self) -> None:
        if getattr(self, "db_client", None) is None:
            self.db_client = TendisClient(
                rename_handle=multimodal_default_rename_handle,
                **self.tendis_kwargs,
            )

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["db_client"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()

    def read_basic_data_info(self, snt_utt: str) -> Optional[BasicDataInfo]:
        """read_basic_data_info.

        从Tendis中, 根据 snt_utt 读取 basic_data_info

        Args:
            snt_utt: 一条json样本的位移ID，示例如下：
            J2MM_develop_speech_batch01-00000_00_00
        """
        key = f"{self.KEY_HEAD}_{snt_utt}"
        buff = self.db_client.get(key)
        if buff is None:
            logging.warning(f"There's no basic_info of [{snt_utt}]")
            return None
        data_info = BasicDataInfo(buff.decode())
        return data_info
