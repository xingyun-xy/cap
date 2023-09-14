# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.


import logging
import random

import torch
from torch.utils import data

from cap.data.datasets.carp.audio import BaseWaveformReader
from cap.data.datasets.carp.basic_data_info import TendisBasicInfoReader
from cap.data.datasets.carp.image import BaseImageReader
from cap.data.datasets.carp.info import BaseInfoList, UttParser
from cap.data.datasets.pack_type.mxrecord import MXRecord
from cap.registry import OBJECT_REGISTRY

try:
    import mxnet as mx
except ImportError:
    mx = None


@OBJECT_REGISTRY.register
class CocktailDatasetV0(data.Dataset):

    SUPPORT_MODE = ["train", "val", "test"]

    def __init__(
        self,
        image_reader: BaseImageReader,
        audio_reader: BaseWaveformReader,
        info_list: BaseInfoList,
        basic_info_reader: TendisBasicInfoReader,
        mode: str = "train",
        transforms=None,
    ):
        self.image_reader = image_reader
        self.audio_reader = audio_reader
        self.info_list = info_list
        self.basic_info_reader = basic_info_reader
        self.utt_parser = UttParser()
        assert (
            mode in self.SUPPORT_MODE
        ), f"mode should be in {self.SUPPORT_MODE}, but get {mode}"
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index) -> dict:
        info = self.info_list[index]
        logging.debug(f"Info: {info}")
        # 利用 info 信息解析其他数据
        matched = self.utt_parser.match(info["seg_utt"])
        logging.debug(f"matched: {matched}")
        snt_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"
        basic_info = self.basic_info_reader.read_basic_data_info(snt_utt)
        cams = basic_info.get_available_cam_ids()
        logging.debug(f"cams: {cams}")
        cam = random.choice(cams) if self.mode == "train" else cams[0]
        # 获取图片序列
        video_utt = (
            f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}"
        )
        logging.debug(f"video_utt: {video_utt}")
        images = self.image_reader.read_seg_images(
            video_utt, info["seg_beg"], info["seg_end"]
        )
        logging.debug(f"images: {len(images)} {images[0].shape}")
        # 获取音频序列
        mics = basic_info.get_available_mic_ids()
        logging.debug(f"mics: {mics}")
        mic = random.choice(mics) if self.mode == "train" else mics[0]
        audio_utt = (
            f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{mic}"
        )
        logging.debug(f"audio_utt: {audio_utt}")
        waveform, samplerate = self.audio_reader.read_seg_audio(
            audio_utt, info["seg_beg"], info["seg_end"]
        )
        logging.debug(f"audio: {waveform.shape}, {samplerate}")
        # 获取label
        logging.debug(f"text: {info['text']}")
        # 构造返回的数据dict
        key = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}_{mic}_{matched['index']}"  # noqa: E501
        data = {
            "key": key,
            "images": images,
            "waveform": (torch.tensor(waveform), samplerate),
            "label": info["text"],
        }
        # 对数据进行各种处理
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.info_list)


@OBJECT_REGISTRY.register
class CocktailCmdDataset(data.Dataset):

    SUPPORT_MODE = ["train", "val", "test"]

    def __init__(
        self,
        image_reader: BaseImageReader,
        audio_reader: MXRecord,
        info_list: BaseInfoList,
        basic_info_reader: TendisBasicInfoReader,
        mode: str = "train",
        transforms=None,
    ):
        if mx is None:
            raise ModuleNotFoundError(
                "mxnet is required by CocktailCmdDataset"
            )

        self.image_reader = image_reader
        self.audio_reader = audio_reader
        self.info_list = info_list
        self.basic_info_reader = basic_info_reader
        self.utt_parser = UttParser()
        assert (
            mode in self.SUPPORT_MODE
        ), f"mode should be in {self.SUPPORT_MODE}, but get {mode}"
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index) -> dict:
        info = self.info_list[index]
        logging.debug(f"Info: {info}")
        # 利用 info 信息解析其他数据
        matched = self.utt_parser.match(info["seg_utt"])
        logging.debug(f"matched: {matched}")
        snt_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"
        basic_info = self.basic_info_reader.read_basic_data_info(snt_utt)
        cams = basic_info.get_available_cam_ids()
        logging.debug(f"cams: {cams}")
        cam = random.choice(cams) if self.mode == "train" else cams[0]
        # 获取图片序列
        video_utt = (
            f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}"
        )
        logging.debug(f"video_utt: {video_utt}")
        images = self.image_reader.read_seg_images(
            video_utt, info["seg_beg"], info["seg_end"]
        )
        logging.debug(f"images: {len(images)} {images[0].shape}")
        # 获取音频序列
        rec_idx = random.choice(info["rec_idx"])
        s = self.audio_reader.read(rec_idx)
        audio = mx.nd.load_from_str(s).asnumpy()
        audio = torch.from_numpy(audio)
        logging.debug(f"audio_fbank: {audio.shape}")
        # 获取label
        logging.debug(f"text: {info['text']}")
        # 构造返回的数据dict
        key = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}_{matched['mic']}_{matched['index']}"  # noqa: E501
        data = {
            "key": key,
            "images": images,
            "audio": audio,
            "tokens": info["tokens"],
        }
        # 对数据进行各种处理
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.info_list)
