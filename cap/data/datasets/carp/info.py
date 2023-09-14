# Copyright (c) Changan Auto, All rights reserved.

import logging
import pathlib
import re
from typing import Any, AnyStr, Dict, Match, Optional, Tuple, Union

from tqdm import tqdm

from cap.data.datasets.carp.parse_utils import (
    parse_id_map,
    parse_monophone,
    parse_video_info,
)
from cap.registry import OBJECT_REGISTRY
from cap.utils.filesystem import get_filesystem

logger = logging.getLogger(__name__)


class BaseInfoList(object):
    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@OBJECT_REGISTRY.register
class MMASRInfoList(BaseInfoList):
    """多模ASR的Info文件列表类.

    Args:
        path: info文件的路径, 支持本地,hdfs,bucket路径.
        left_limit: 左边界限制,即小于该时长的会被过滤. 单位秒, 默认是 0.1 .
        right_limit: 右边界限制,即大于该时长的会被过滤. 单位秒，默认是 6.0 .

    Raises:
        FileNotFoundError: 当path的路径不存在时返回该异常
    """

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        left_limit: float = 0.1,
        right_limit: float = 6.0,
    ):
        self._info_path = path
        self._left_limit = left_limit
        self._right_limit = right_limit
        self._fork()

    def _fork(self):  # type: ignore
        fs = get_filesystem(self._info_path)
        with fs.open(self._info_path, "r", encoding="utf-8") as fr:

            def callback(line: str) -> Optional[Dict[str, Any]]:
                # 内部解析数据的接口
                utt, *text, beg, end = line.strip().split()
                ret = {
                    "seg_utt": utt,
                    "seg_beg": float(beg),
                    "seg_end": float(end),
                    "text": "".join(text),
                }
                duration = ret["seg_end"] - ret["seg_beg"]
                if duration <= 0.0:
                    msg = f"错误的INFO行(时长为负): {line}"
                elif duration < self._left_limit:
                    msg = f"错误的INFO行(<{self._left_limit}): {line}"
                elif duration > self._right_limit:
                    msg = f"错误的info行(>{self._right_limit}): {line}"
                else:
                    msg = None
                if msg is not None:
                    logger.warning(msg)
                    return None
                return ret

            info_iter = (callback(line) for line in fr)
            self.infos = [
                ret
                for ret in tqdm(info_iter, ncols=120, desc="Info Load...")
                if ret is not None
            ]

    def __getitem__(self, index: int) -> dict:
        return self.infos[index]

    def __len__(self) -> int:
        return len(self.infos)

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["infos"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"path=.../{self._info_path.name}"
        repr_str += f", left_limit={self._left_limit :.3f}"
        repr_str += f", right_limit={self._right_limit :.3f}"
        repr_str += f", lens={len(self)}"
        return repr_str


@OBJECT_REGISTRY.register
class MMCMDInfoList(BaseInfoList):
    """MMCMDInfoList.

    Args:
        info_path: path for video_info.txt
        monophone_path: path for monophone.txt
        id_map_path: path for id.map

    """

    def __init__(self, info_path: str, monophone_path: str, id_map_path: str):
        self._info_path = info_path
        self._monophone_path = monophone_path
        self._id_map_path = id_map_path
        self._fork()

    def _fork(self):  # type: ignore
        utt_to_video_info = parse_video_info(self._info_path)
        utt_to_monophone = parse_monophone(self._monophone_path)
        utt_to_idxs = parse_id_map(self._id_map_path)
        infos = []
        filtered_count = 0
        for utt, idxs in utt_to_idxs.items():
            if utt not in utt_to_video_info:
                filtered_count += len(idxs)
                continue
            if utt not in utt_to_monophone:
                filtered_count += len(idxs)
                continue

            info = utt_to_video_info[utt]
            info["tokens"] = utt_to_monophone[utt]["monophone"]
            info["rec_idx"] = idxs
            infos.append(info)
        self.infos = infos

        logging.warning(f"Total Filtered : {filtered_count}")

    def __getitem__(self, index: int) -> dict:
        return self.infos[index]

    def __len__(self) -> int:
        return len(self.infos)

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["infos"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()


class UttParser(object):

    _DATASET_BATCH_PATTERNS = {
        "standard": re.compile(
            "(J2MM|S202DA|IPC_halo2|IPC_halo0|DATATANG|FYZ|BYD|QR).*?_(batch[0-9]{2})"  # noqa: E501
        ),
        "HAOKAN": re.compile("(batch[0-9]{4})"),
    }

    _UTT_PASER_REGEXS = {
        "J2MM": "(?P<dataset>J2MM.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>[0-9]{5}_[0-9]{2}_[0-9]{2})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "IPC_halo0": "(?P<dataset>IPC_halo0.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((od)|(ho)|[0-9])[0-9]+_[0-9]{2}_[0-9]+)(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "S202DA": "(?P<dataset>S202DA.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>[0-9]{5}_[0-9]{2}_[0-9]{2})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "DATATANG": "(?P<dataset>DATATANG)_(?P<batch>batch[0-9]{2})-(?P<snt>G[0-9]{4}_[0-9]{2}_[0-9]{4})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "HAOKAN": "(?P<batch>batch[0-9]{4})-(?P<snt>11[0-9]{4})(_(?P<index>[0-9]*))?",  # noqa: E501
        "IPC_halo2": "(?P<dataset>IPC_halo2)_(?P<batch>batch[0-9]{2})_(?P<snt>[0-9]{5}_[0-9]{2}_[0-9]{2})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "FYZ": "(?P<dataset>FYZ.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((batch[0-9]{2}_)*[A-Z][0-9]{4}_[0-9]{2}_[0-9]{2}|[A-Z][0-9]{4}))(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "BYD": "(?P<dataset>BYD.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((od)|(ho)|[0-9])[0-9]+_[0-9]{2}_[0-9]+)(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "QR": "(?P<dataset>QR.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((od)|(ho))[0-9]{6}_[0-9]{2}_[0-9]{2})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
    }

    def __init__(self):  # type: ignore[no-untyped-def]
        self.parsers = {}

    def match(self, utt: str) -> Optional[Match[AnyStr]]:
        dataset, batch = self.parse_dataset_and_batch(utt)
        if dataset not in self.parsers:
            self.parsers[dataset] = re.compile(self._UTT_PASER_REGEXS[dataset])
        matched = self.parsers[dataset].match(utt)
        if matched is None:
            logging.warning(f"[{utt}] can't be matched.")
        return matched

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["parsers"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self.parsers = {}

    @classmethod
    def parse_dataset_and_batch(cls, info_key: str) -> Tuple[str, str]:
        dataset, batch = "unknown", "unknown"
        for key, pattern in cls._DATASET_BATCH_PATTERNS.items():
            db_and_batch = pattern.findall(info_key)
            if len(db_and_batch) != 1:
                continue
            if key == "HAOKAN":
                dataset, batch = "HAOKAN", db_and_batch[0]
                break
            else:
                dataset, batch = db_and_batch[0]
                break
        return dataset, batch
