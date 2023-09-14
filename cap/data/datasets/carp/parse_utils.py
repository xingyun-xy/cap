# Copyright (c) Changan Auto, All rights reserved.


import logging

from tqdm import tqdm

from cap.utils.filesystem import get_filesystem


def parse_video_info(path: str):
    fs = get_filesystem(path)
    ret = {}
    with fs.open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), ncols=120, desc="Viedo Info Load..."):
            utt, *text, beg, end = line.strip().split()
            info = {
                "seg_utt": utt,
                "seg_beg": float(beg),
                "seg_end": float(end),
                "text": "".join(text),
            }
            if info["seg_end"] - info["seg_beg"] < 0.1:  # type: ignore
                logging.warning(f"ErrorInfo: {line}")
                continue
            ret[utt] = info
    return ret


def parse_monophone(path: str):
    fs = get_filesystem(path)
    ret = {}
    with fs.open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), ncols=120, desc="Monophone Load..."):
            utt, *monophones = line.strip().split()
            info = {
                "seg_utt": utt,
                "monophone": monophones,
            }
            ret[utt] = info
    return ret


def parse_id_map(path: str):
    fs = get_filesystem(path)
    ret = {}
    with fs.open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), ncols=120, desc="ID Map Load..."):
            line = line.strip()
            if not line:
                continue
            rec_idx, utt = line.split()
            if utt not in ret:
                ret[utt] = []
            ret[utt].append(rec_idx)
    return ret
