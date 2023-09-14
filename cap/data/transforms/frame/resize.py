# Copyright (c) Changan Auto. All rights reserved.

import math

import cv2
import numpy as np
from capbc.utils.import_util import check_import

from cap.registry import OBJECT_REGISTRY

try:
    from pyramid_resizer import pyramid_resizer
except ImportError:
    pyramid_resizer = None

__all__ = ["BPUPyramidResizer", "CV2AdptiveResolutionInput"]


@OBJECT_REGISTRY.register
class BPUPyramidResizer(object):
    """BPU Pyramid Resizer.

    Args:
        scale_wh: Pyramid resize scale. Valid scale is 0.5^n, n=0,1,2,3,4,5.
        pyramid_type: Pyramid type, "ips" or "ipu".
    """

    def __init__(self, scale_wh: tuple, pyramid_type: str):
        super().__init__()
        check_import(pyramid_resizer, "pyramid_resizer")

        assert pyramid_type in [
            "ips",
            "ipu",
        ], f"`pyramid_type` should be 'ips' or 'ipu', but get {pyramid_type}."

        self.scale_wh = tuple(scale_wh)
        self.pyramid_type = pyramid_type
        self.pyramid = self._get_pyramid_class(self.pyramid_type)

    def _get_pyramid_class(self, pyramid_type):
        if pyramid_type == "ips":
            pyramid = pyramid_resizer.IPSPyramid()
        elif pyramid_type == "ipu":
            pyramid = pyramid_resizer.IpuPyramid()
        else:
            raise NotImplementedError(
                "`pyramid_type` only support `ips` and `ipu`."
            )
        return pyramid

    def __getstate__(self):
        state = self.__dict__.copy()
        state["pyramid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.pyramid = self._get_pyramid_class(self.pyramid_type)

    def __call__(self, data):

        yuvi420_buf = data["img_buf"]
        img_hw = (data["img_height"], data["img_width"])
        resize_hw = (
            int(img_hw[0] * self.scale_wh[1]),
            int(img_hw[1] * self.scale_wh[0]),
        )

        if self.pyramid_type == "ips":
            self.pyramid.build_pyramid(yuvi420_buf, img_hw[1], img_hw[0], 1)
            layer_id = self._get_pyramid_layer_id(img_hw[1], resize_hw[1])
            yuv420sp = self.pyramid.get_image_data_yuv420(layer_id)
            yuvi420_buf = self._yuv420sp_to_yuvi420(yuv420sp, resize_hw)
        elif self.pyramid_type == "ipu":
            yuv420sp = self.pyramid.get_resize_420sp_from_i420bytes(
                yuvi420_buf, tuple(img_hw), tuple(resize_hw)
            )
            h_padding = int(resize_hw[0])
            w_padding = int(len(yuv420sp) / 1.5 / resize_hw[0])
            padding_hw = (h_padding, w_padding)
            yuv420sp_np = np.fromstring(yuv420sp, dtype="uint8").reshape(
                (-1, w_padding)
            )
            yuvi420_buf = self._yuv420sp_to_yuvi420(
                yuv420sp_np, tuple(padding_hw)
            )
            yuvi420_buf = self._unpad_image_yuvi420(
                yuvi420_buf, tuple(padding_hw), tuple(resize_hw)
            )
        data["img_buf"] = yuvi420_buf
        data["img_height"], data["img_width"] = resize_hw

        return data

    def _yuv420sp_to_yuvi420(self, yuv420sp, img_hw, to_string=True):
        y_img = yuv420sp[: img_hw[0], :]
        u_img = yuv420sp[img_hw[0] :, ::2]
        v_img = yuv420sp[img_hw[0] :, 1::2]
        if to_string:
            return y_img.tostring() + u_img.tostring() + v_img.tostring()
        else:
            return y_img, u_img, v_img

    def _get_pyramid_layer_id(self, src_w, dst_w):
        layer = int(math.log(src_w / dst_w, 2.0))
        assert layer in [0, 1, 2, 3, 4, 5, 6]
        return layer

    def _unpad_image_yuvi420(self, img_yuvi420, src_hw, dst_hw):
        img_y_str = img_yuvi420[: src_hw[0] * src_hw[1]]
        u_offset = int(src_hw[0] * src_hw[1] + int(src_hw[0] * src_hw[1] / 4))
        img_u_str = img_yuvi420[src_hw[0] * src_hw[1] : u_offset]
        img_v_str = img_yuvi420[u_offset:]

        img_y = np.fromstring(img_y_str, dtype="uint8").reshape(src_hw)
        src_uv_hw = (int(src_hw[0] / 2), int(src_hw[1] / 2))
        img_u = np.fromstring(img_u_str, dtype="uint8").reshape(src_uv_hw)
        img_v = np.fromstring(img_v_str, dtype="uint8").reshape(src_uv_hw)
        img_y_crop = img_y[: dst_hw[0], : dst_hw[1]]
        img_u_crop = img_u[: int(dst_hw[0] / 2), : int(dst_hw[1] / 2)]
        img_v_crop = img_v[: int(dst_hw[0] / 2), : int(dst_hw[1] / 2)]

        return (
            img_y_crop.tostring()
            + img_u_crop.tostring()
            + img_v_crop.tostring()
        )

    def inverse_transform(self, obj):
        obj.rescale(1 / self.scale_wh[0], 1 / self.scale_wh[1])
        return obj


@OBJECT_REGISTRY.register
class CV2AdptiveResolutionInput(object):
    """
    CV2 resizer transformer.

    Parameters
    ----------
    model_input_hw : tuple of int,
        The size of model input data.
    scale_type : str, default="MIN"
        The way to transfrom scale. Possivle value:
        "W": scale = w_scale = float(target_wh[0])/img_wh[0]
        "H": scale = h_scale = float(target_wh[1])/img_wh[1]
        "MIN": scale = min(w_scale, h_scale)
        "MAX": scale = max(w_scale, h_scale)
    """

    def __init__(self, model_input_hw, scale_type="MIN"):
        super().__init__()
        self.model_input_hw = model_input_hw

        assert scale_type in ["W", "H", "MIN", "MAX"]
        self.scale_type = scale_type
        self.transform_meta = {}

    def _get_scale(self, model_input_hw, img_hw, scale_type):

        h_scale = float(model_input_hw[0]) / img_hw[0]
        w_scale = float(model_input_hw[1]) / img_hw[1]
        if scale_type == "W":
            res_scale = w_scale
        elif scale_type == "H":
            res_scale = h_scale
        elif scale_type == "MIN":
            res_scale = min(w_scale, h_scale)
        elif scale_type == "MAX":
            res_scale = max(w_scale, h_scale)
        else:
            raise ValueError("Unknow scale_type:{}".format(scale_type))

        scale_side_idx = 0 if res_scale == h_scale else 1
        return res_scale, scale_side_idx

    def _cal_trans_param(self, img_hw):

        reszie_scale, scale_side_idx = self._get_scale(
            self.model_input_hw, img_hw, self.scale_type
        )
        padd_size_idx = 1 - scale_side_idx
        padding_len = (
            int(self.model_input_hw[padd_size_idx] / reszie_scale)
            - img_hw[padd_size_idx]
        )  # noqa

        padding_side = padd_size_idx
        scale_wh = [reszie_scale, reszie_scale]

        return padding_len, padding_side, scale_wh

    def _crop_img(self, bgr_img, crop_hw):
        img_h = bgr_img.shape[0]
        img_w = bgr_img.shape[1]

        return bgr_img[: img_h - crop_hw[0], : img_w - crop_hw[1], :]

    def _cv2_padding_zero(self, bgr_img, padding_len):
        assert len(padding_len) == 4
        padding_left = padding_len[0]
        padding_top = padding_len[1]
        padding_right = padding_len[2]
        padding_bottom = padding_len[3]

        padding_param = (
            (padding_top, padding_bottom),  # height
            (padding_left, padding_right),  # width
            (0, 0),
        )
        padding_img = np.pad(bgr_img, padding_param, "constant")
        return padding_img

    def __call__(self, img_meta):

        bgr_img, org_img_hw = img_meta["img"], img_meta["img_shape"]

        if img_meta["layout"] == "chw":
            c, h, w = org_img_hw
            org_img_hw = (h, w, c)
        # 获取变换参数
        padding_len, padding_side, scale_wh = self._cal_trans_param(org_img_hw)
        if padding_len >= 0:
            padding_param = [0, 0, 0, 0]
            padding_param[-(padding_side + 1)] = padding_len
            padding_img = self._cv2_padding_zero(bgr_img, padding_param)
        else:
            crop_hw = [0, 0]
            crop_hw[padding_side] = -padding_len
            padding_img = self._crop_img(bgr_img, crop_hw)
        padding_img_hw = padding_img.shape[:2]

        resize_img_hw = [
            int(self.model_input_hw[0]),
            int(self.model_input_hw[1]),
        ]

        resize_img = cv2.resize(
            padding_img, (resize_img_hw[1], resize_img_hw[0])
        )

        transform_meta = {
            "original_img_hw": org_img_hw,
            "padding_img_hw": padding_img_hw,
            "transform_hw": resize_img_hw,
            "scale_wh": scale_wh,
            "padding_len": padding_len,
            "padding_side": padding_side,
        }

        self.transform_meta = transform_meta

        img_meta.update(
            {
                "img": resize_img,
                "img_height": resize_img_hw[0],
                "img_width": resize_img_hw[1],
            }
        )

        return img_meta

    def inverse_transform(self, results):

        if self.transform_meta:
            assert isinstance(results, tuple)

            new_results = []
            for obj in results:
                obj.rescale(
                    1 / self.transform_meta["scale_wh"][0],
                    1 / self.transform_meta["scale_wh"][1],
                )
                new_results.append(obj)
            results = tuple(new_results)
        return results
