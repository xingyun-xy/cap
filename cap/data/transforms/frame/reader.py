# Copyright (c) Changan Auto. All rights reserved.

from capbc.utils.import_util import check_import

try:
    from pyramid_resizer import pyramid_resizer
except ImportError:
    pyramid_resizer = None

from cap.registry import OBJECT_REGISTRY

__all__ = ["ImgBufToYUV444", "YUVTurboJPEGDecoder"]


@OBJECT_REGISTRY.register
class YUVTurboJPEGDecoder(object):
    """YUV JPEG Decoder.

    Args:
        to_string: Whether to decode jpeg into yuvi420_buf.
        If to_string is `False`, will return (Y, U, V) in np.ndarray type.
    """

    def __init__(self, to_string: bool = True):
        check_import(pyramid_resizer, "pyramid_resizer")
        self.to_string = to_string

    def __call__(self, data):
        img_bytes = data["img_buf"]
        if self.to_string:
            (
                yuv42x,
                img_w,
                img_h,
                sample_type,
            ) = pyramid_resizer.imdecode_yuv42x(img_bytes, decode=False)
            if sample_type == "yuvi420":
                yuvi420_buf = yuv42x
            elif sample_type == "yuv422p":
                yuvi420_buf = pyramid_resizer.yuv422p_str2yuvi420_str(
                    yuv42x, img_w, img_h
                )
            else:
                raise NotImplementedError(
                    "Sample type only support `yuvi420` and `yuv422p`."
                )

            data["img_buf"] = yuvi420_buf
            data["img_height"], data["img_width"] = img_h, img_w
            data["color_space"] = "yuv"
            return data
        else:
            y_img, u_img, v_img = pyramid_resizer.imdecode_yuv42x(
                img_bytes, decode=True
            )
            return y_img, u_img, v_img


@OBJECT_REGISTRY.register
class ImgBufToYUV444(object):
    def __call__(self, data):
        data["img"] = pyramid_resizer.yuvi420_str2yuv444_np(
            data["img_buf"], data["img_width"], data["img_height"]
        )
        data["img_shape"] = data["img"].shape

        return data
