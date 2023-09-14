import json
import os
from glob import glob

import numpy as np
from capbc.utils.import_util import check_import

# try:
#     from pyramid_resizer.image_cvt_utils import convert_nv12_to_yuv444_uint8
# except ImportError:
#     convert_nv12_to_yuv444_uint8 = None

from torch.utils.data import Dataset

from cap.registry import OBJECT_REGISTRY


def read_yuv_with_shape(yuv_file, im_hw):
    frame_len = im_hw[0] * im_hw[1] * 3 // 2
    with open(yuv_file, "rb") as f:
        raw = f.read(frame_len)
    yuv = np.frombuffer(raw, dtype=np.uint8)
    yuv = yuv.reshape(im_hw[0] * 3 // 2, im_hw[1])
    return yuv


def load_calib_and_dist_coeffs(calib_path):
    with open(calib_path, "r") as rf:
        calib_dict = json.load(rf)

    fu, fv = calib_dict["focal_u"], calib_dict["focal_v"]
    cu, cv = calib_dict["center_u"], calib_dict["center_v"]
    calib = np.array(
        [
            [fu, 0, cu, 0],
            [0, fv, cv, 0],
            [0, 0, 1, 0],
        ]
    )
    distCoeffs = np.array(calib_dict["distort"])

    return calib, distCoeffs


def convert_nv12_to_yuv444_uint8(img_data, img_shape):
    if len(img_shape) == 2: #For convenience, some times args dosen't contain # of channels
        img_shape = (img_shape[0], img_shape[1], 3)
    assert len(img_shape) == 3, 'Len of image_shape should be 2 (h x w) or 3 (h x w x c)'
    nv12_data = img_data.flatten()
    uv_start_idx = img_shape[0] * img_shape[1]
    nv12_y_data = nv12_data.flatten()[0: uv_start_idx]
    u_id = np.arange(len(nv12_data))
    u_id = np.bitwise_and(u_id >= uv_start_idx, u_id % 2 == uv_start_idx % 2)
    nv12_u_data = nv12_data.flatten()[u_id]
    v_id = np.arange(len(nv12_data))
    v_id = np.bitwise_and(v_id >= uv_start_idx, v_id % 2 != uv_start_idx % 2)
    nv12_v_data = nv12_data.flatten()[v_id]
    # truncate YUV data as int8
    nv12_y_data = nv12_y_data.astype(np.uint8)
    nv12_u_data = nv12_u_data.astype(np.uint8)
    nv12_v_data = nv12_v_data.astype(np.uint8)
    # reformat data as nv12
    yuv444_res = np.zeros(img_shape, dtype=np.uint8)
    # yuv444_res = np.zeros(img_shape, dtype=np.uint8)
    for h in range(img_shape[0]):
        # centralize yuv 444 data for inference framework
        for w in range(img_shape[1]):
            yuv444_res[h][w][0] = (nv12_y_data[h * img_shape[1] + w]).astype(np.uint8)
            yuv444_res[h][w][1] = (nv12_u_data[int(h / 2) * int(img_shape[1] / 2) + int(w / 2)]).astype(np.uint8)
            yuv444_res[h][w][2] = (nv12_v_data[int(h / 2) * int(img_shape[1] / 2) + int(w / 2)]).astype(np.uint8)
    return yuv444_res


@OBJECT_REGISTRY.register
class YUVFrames(Dataset):
    def __init__(
        self,
        img_dir,
        im_hw,
        calib_path=None,
        transforms=None,
    ):

        # check_import(convert_nv12_to_yuv444_uint8, "pyramid_resizer")

        self.img_paths = glob(os.path.join(img_dir, "*.yuv"))
        self.im_hw = im_hw

        if calib_path is not None:
            calib, dist_coeffs = load_calib_and_dist_coeffs(calib_path)
            self.calib_dict = {"calib": calib, "distCoeffs": dist_coeffs}
        else:
            self.calib_dict = None

        self.transform = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        nv12 = read_yuv_with_shape(img_path, self.im_hw)

        yuv = convert_nv12_to_yuv444_uint8(nv12, self.im_hw)
        data = {
            "img": yuv.astype(np.float32).transpose(2, 0, 1),
            "img_id": img_path.split("/")[-1].split(".")[0],
        }

        if self.calib_dict is not None:
            data.update(self.calib_dict)

        data = data if self.transform is None else self.transform(data)
        return data

    def __len__(self):
        return len(self.img_paths)
