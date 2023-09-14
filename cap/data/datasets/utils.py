# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.

import cv2
import numpy as np


def encode_img(
    img: np.ndarray, quality: int = 95, img_fmt: str = ".jpg"
) -> bytes:
    """encode_img.

    对图片进行编码返回编码后的字符串


    Args:
        img : 图片的矩阵, 形状是 HxWxC 的
        quality : 压缩图像的质量，默认为95%
        img_fmt : 压缩图片的格式，默认为 ".jpg"

    Returns:
        str 图片压缩后的buf str
    """
    jpg_formats = [".JPG", ".JPEG"]
    png_formats = [".PNG"]
    encode_params = None
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, "failed to encode image"
    return buf


def decode_img(s: bytes, iscolor: int = -1) -> np.ndarray:
    """decode_img.

    对图片压缩后的buff str进行解码, 得到解压后的图片数组

    Args:
        s: 图片压缩后的bytes字符串
        iscolor: 是否是彩色图片. 默认是 -1.

    Returns:
        np.ndarray 解码后的图片数组
    """
    img = np.frombuffer(s, dtype=np.uint8)
    img = cv2.imdecode(img, iscolor)
    return img
