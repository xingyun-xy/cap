# Copyright (c) Changan Auto. All rights reserved.

import copy
import os
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

from cap.data.transforms.real3d import (
    draw_heatmap,
    get_gaussian2D,
    get_reg_map,
)
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from .classification import BgrToYuv444

__all__ = [
    "Resize3DV",
    "Crop3DV",
    "ToTensor3DV",
    "Normalize3DV",
    "PrepareDepthPose",
    "SelectDataByIdx",
    "StackData",
    "PrepareDataBEV",
    "Bev3dTargetGenerator",
    "BevDiscreteTargetGenerator",
    "ConvertReal3dTo3DV",
    "Bev3dTargetGeneratorV2",
    "BevSegTargetGenerator",
    "BevSegAnnoGenerator",
]

PIL_INTERP_CODES = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
}


@OBJECT_REGISTRY.register
class Collect3DV(object):
    """Selecting data what we really need from all frames.

    Args:
        load_data_types list(str): which type data to load. Depth training
            only need gt_depths; Pose training need intrinsics, obj_mask.
        img_idxs list(int): frame indexs of input images.
        gt_seg_idxs list(int): frame indexs of gt seg to load.
        gt_depth_idxs list(int): frame indexs of gt depth to load.
        gt_bev_seg_idxs list(int): frame indexs of gt bev seg to load.
        gt_bev_3d_idxs list(int): frame indexs of gt bev 3d to load.
        gt_bev_motionflow_idxs list(int): frame indexs bev
            motion flow to load.
        gt_bev_discobj_idx (int): frame index of gt bev discrete object
            (e.g., crosswalk, road arrow, etc.) to load
        obj_mask_idx int: frame index of object mask to load.
        timestamp_idx int: timestamp index to load

    """

    def __init__(
        self,
        load_data_types,
        img_idxs,
        gt_seg_idxs=(0,),
        gt_depth_idx=1,
        gt_bev_seg_idxs=(0,),
        gt_bev_3d_idx=0,
        gt_bev_motionflow_idxs=(0,),
        gt_bev_discobj_idx=0,
        gt_om_idx=0,
        img_paths_idx=0,
        obj_mask_idx=1,
        timestamp_idx=0,
        occlusion_idxs=(0,),
    ):
        self.load_data_types = load_data_types
        self.img_idxs = img_idxs
        self.img_paths_idx = img_paths_idx
        self.gt_seg_idxs = gt_seg_idxs
        self.gt_depth_idx = gt_depth_idx
        self.gt_bev_seg_idxs = gt_bev_seg_idxs
        self.gt_bev_3d_idx = gt_bev_3d_idx
        self.gt_bev_motionflow_idxs = gt_bev_motionflow_idxs
        self.gt_om_idx = gt_om_idx
        self.gt_bev_discobj_idx = gt_bev_discobj_idx
        self.obj_mask_idx = obj_mask_idx
        self.timestamp_idx = timestamp_idx
        self.occlusion_idxs = occlusion_idxs

    def _squeeze_list(
        self,
        list_data,
    ):
        if isinstance(list_data, Sequence) and len(list_data) == 1:
            return self._squeeze_list(list_data[0])
        else:
            return list_data

    def __call__(self, data_dict: Mapping):
        frames = data_dict.pop("frames")

        data_dict["pil_imgs"] = [frames[idx].img() for idx in self.img_idxs]
        if "gt_seg" in self.load_data_types:
            data_dict["gt_seg"] = self._squeeze_list(
                [frames[idx].seg() for idx in self.gt_seg_idxs]
            )
        if "gt_depth" in self.load_data_types:
            data_dict["gt_depth"] = frames[self.gt_depth_idx].depth()

        if "gt_bev_seg" in self.load_data_types:
            data_dict["gt_bev_seg"] = self._squeeze_list(
                [frames[idx].bev_seg() for idx in self.gt_bev_seg_idxs]
            )
        if "occlusion" in self.load_data_types:
            data_dict["occlusion"] = self._squeeze_list(
                [frames[idx].bev_occlusion() for idx in self.occlusion_idxs]
            )
        if "gt_bev_seg_anno" in self.load_data_types:
            data_dict["gt_bev_static_anno"] = self._squeeze_list(
                [
                    frames[idx].read_static_anno()
                    for idx in self.gt_bev_seg_idxs
                ]
            )
        if "gt_bev_3d" in self.load_data_types:
            data_dict["gt_bev_3d"] = frames[self.gt_bev_3d_idx].bev_3d()

        if "gt_bev_motion_flow" in self.load_data_types:
            data_dict["gt_bev_motion_flow"] = self._squeeze_list(
                [
                    frames[idx].bev_motion_flow()
                    for idx in self.gt_bev_motionflow_idxs
                ]
            )

        if "gt_bev_discrete_obj" in self.load_data_types:
            data_dict["gt_bev_discrete_obj"] = frames[
                self.gt_bev_discobj_idx
            ].bev_discrete_obj()

        if "gt_online_mapping" in self.load_data_types:
            data_dict["gt_online_mapping"] = frames[self.gt_om_idx].om()

        # load object mask of t-1 in front view, only for 2.5d task
        if "obj_mask" in self.load_data_types:
            data_dict["obj_mask"] = frames[self.obj_mask_idx].front_seg()

        # only load timestamp of t
        if "timestamp" in self.load_data_types:
            data_dict["timestamp"] = frames[self.timestamp_idx].timestamp()

        # only return pack_dir of frame 0
        if "pack_dir" in self.load_data_types:
            data_dict["pack_dir"] = frames[0].pack_dir

        # only return img_paths of frame 0
        if "img_paths" in self.load_data_types:
            data_dict["img_paths"] = frames[0].img_paths

        return data_dict

    def __repr__(self):
        return "Collect3DV"


@OBJECT_REGISTRY.register
class Resize3DV(object):
    """Resize PIL Images to the given size and modify intrinsics.

    Args:
        size (sequence): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
    interpolation (str, optional):Desired interpolation. Default is 'nearest'.
    resize_depth (bool): whether resize gt depth.
    """

    def __init__(
        self,
        size: Union[Sequence, Sequence[Sequence]],
        interpolation: str = "nearest",
        resize_depth=True,
    ):
        self.size = size if isinstance(size[0], Sequence) else [size]
        assert interpolation in PIL_INTERP_CODES
        self.interpolation = interpolation
        self.resize_depth = resize_depth

    def _resize(self, data: Union[Image.Image, Sequence], size, interpolation):
        if isinstance(data, Sequence):
            assert len(data) == len(size)
            return [
                self._resize(data_i, size, interpolation)
                for data_i, size in zip(data, size)
            ]
        else:
            return F.resize(data, size, interpolation)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'
        if len(data["pil_imgs"][0]) != len(self.size):
            duplicate = len(data["pil_imgs"][0]) / len(self.size)
            self.size = self.size * int(duplicate)

        data["pil_imgs"] = [
            self._resize(
                pil_img, self.size, PIL_INTERP_CODES[self.interpolation]
            )
            for pil_img in data["pil_imgs"]
        ]
        # in-place modify on data['size'] will affect self.size,
        # use copy.deepcopy fix it
        data["size"] = copy.deepcopy(self.size)

        if "gt_seg" in data:
            data["gt_seg"] = self._resize(
                data["gt_seg"], self.size, PIL_INTERP_CODES["nearest"]
            )

        if "gt_depth" in data and self.resize_depth:
            data["gt_depth"] = self._resize(
                data["gt_depth"], self.size, PIL_INTERP_CODES["nearest"]
            )

        if "color_imgs" in data:
            data["color_imgs"] = [
                self._resize(
                    color_img, self.size, PIL_INTERP_CODES[self.interpolation]
                )
                for color_img in data["color_imgs"]
            ]

        if "front_mask" in data:
            data["front_mask"] = self._resize(
                data["front_mask"], self.size[0], PIL_INTERP_CODES["nearest"]
            )

        if "obj_mask" in data:
            data["obj_mask"] = self._resize(
                data["obj_mask"], self.size[0], PIL_INTERP_CODES["nearest"]
            )

        if "intrinsics" in data:
            # scale intrinsics matrix
            data["intrinsics"][0, :] *= self.size[0][1]
            data["intrinsics"][1, :] *= self.size[0][0]

        if "homography" in data:
            # modify homography matrix
            for i in range(len(self.size)):
                scale_mat = np.eye(3, dtype="float32")
                scale_mat[0, 0] = self.size[i][1]
                scale_mat[1, 1] = self.size[i][0]
                data["homography"][i] = scale_mat @ data["homography"][i]
        return data

    def __repr__(self):
        return "Resize3DV"


@OBJECT_REGISTRY.register
class BevSegTargetGenerator(object):
    """Generate gt_bev_seg from gt_bev_static_anno to calculate bevseg loss\
    or calculate MIOU metric.

    Args:
        vcs_range (sequence): vcs range.(order is (bottom,right,top,left))
        vcs_origin_coord (sequence): The position of Car in vcs coordinate.
        line_width (int, optional): Thickness of bevseg midline elements.
                    such as roadedges,solid_lanes
        bev_size (sequence):bev_size
    """

    def __init__(
        self,
        vcs_range: Sequence[float],
        vcs_origin_coord: Sequence[float],
        line_width: int = 2,
        bev_size: Sequence[int] = (512, 512),
    ):
        self.bev_size = bev_size
        self.labels = {
            "roadedges": (1, 1, 1),
            "roadarrows": (2, 2, 2),
            "solid_lanes": (3, 3, 3),
            "stoplines": (4, 4, 4),
            "crosswalks": (5, 5, 5),
            "ignores": (255, 255, 255),
        }
        scope_H = vcs_range[2] - vcs_range[0]
        scope_W = vcs_range[3] - vcs_range[1]
        bev_height, bev_width = bev_size
        self.vcs2bev = np.array(
            [
                0.0,
                -float(bev_width) / scope_W,
                vcs_origin_coord[1],
                -float(bev_height) / scope_H,
                0.0,
                vcs_origin_coord[0],
                0,
                0,
                1,
            ],
            dtype=np.float,
        ).reshape((3, 3))
        self.line_width = line_width

    def __call__(self, data):
        if "gt_bev_static_anno" not in data.keys():
            """When 'gt_bev_static_anno' not in data, load gt_bev_seg from bev_seg_rec.
            Otherwise, generate gt_bev_seg from bev_seg_lmdb file
            """
            assert "gt_bev_seg" in data.keys()
            return data
        bev_seg_dot = data.pop("gt_bev_static_anno")
        bev_height, bev_width = self.bev_size
        bev_map = np.zeros((bev_height, bev_width, 3), dtype="uint8")

        # draw bevseg img
        for (category, edges) in bev_seg_dot.items():
            if edges is not None:
                if category in ["crosswalks", "roadarrows", "ignores"]:
                    # draw plane element
                    for edge in edges:
                        if edge:
                            output_list = []
                            for _current_edge in edge:
                                [x1, y1, x2, y2] = _current_edge
                                pt1 = np.squeeze(
                                    self.vcs2bev
                                    @ np.array(
                                        [x1, y1, 1.0], dtype=np.float
                                    ).reshape((3, 1))
                                )
                                pt2 = np.squeeze(
                                    self.vcs2bev
                                    @ np.array(
                                        [x2, y2, 1.0], dtype=np.float
                                    ).reshape((3, 1))
                                )
                                output_list.append(
                                    [int(pt1[0] + 0.5), int(pt1[1] + 0.5)]
                                )
                                output_list.append(
                                    [int(pt2[0] + 0.5), int(pt2[1] + 0.5)]
                                )
                            edge = np.array([output_list])
                            cv2.fillConvexPoly(
                                bev_map, edge, self.labels[category]
                            )
                elif category in ["roadedges", "solid_lanes", "stoplines"]:
                    # draw line element
                    for edge in edges:
                        for _current_edge in edge:
                            [x1, y1, x2, y2] = _current_edge
                            pt1 = np.squeeze(
                                self.vcs2bev
                                @ np.array(
                                    [x1, y1, 1.0], dtype=np.float
                                ).reshape((3, 1))
                            )
                            pt2 = np.squeeze(
                                self.vcs2bev
                                @ np.array(
                                    [x2, y2, 1.0], dtype=np.float
                                ).reshape((3, 1))
                            )
                            bev_map = cv2.line(
                                bev_map,
                                (int(pt1[0] + 0.5), int(pt1[1] + 0.5)),
                                (int(pt2[0] + 0.5), int(pt2[1] + 0.5)),
                                self.labels[category],
                                self.line_width,
                            )
        data["gt_bev_seg"] = Image.fromarray(bev_map[:, :, 0]).convert("I")
        return data

    def __repr__(self):
        return "BevSegTargetGenerator"


@OBJECT_REGISTRY.register
class BevSegAnnoGenerator(object):
    """Generate gt_bev_seg_anno from gt_bev_static_anno to\
    calculate BevSegInstanceEval metric.

    Args:
        vcs_range (sequence): vcs range.(order is (bottom,right,top,left))
        max_anno_num (int, optional): Maximum number of each bevseg annotation
    """

    def __init__(
        self,
        vcs_range: Sequence[float],  # (bottom, right, top, left)
        max_anno_num: int = 20000,
    ):
        self.vcs_range = vcs_range
        self.max_anno_num = max_anno_num

    def get_arrange(self, x1, y1, x2, y2):
        # generate lines at 0.1m intervals from endpoints
        output_list = []
        if abs(x2 - x1) < 0.1:
            output_list.append([x1, y1])
            output_list.append([x2, y2])
            return output_list
        else:
            if x1 > x2:
                x_list = np.arange(x2, x1, 0.1).tolist()
                x_list.append(x1)
            else:
                x_list = np.arange(x1, x2, 0.1).tolist()
                x_list.append(x2)
            x_list = [float(format(x, ".1f")) for x in x_list]
            x_list = [_item for _item in x_list if (_item * 10) % 2 != 0]
            k = (y1 - y2) / (x1 - x2)
            b = y1 - x1 * k
            y_list = [k * x + b for x in x_list]
            output_list = [[x, y] for (x, y) in zip(x_list, y_list)]
            return output_list

    def convert_points(self, input_arrays):
        """Convert arrays to the set of points needed for evaluation.

        If some category less than 20,000 points in a category,
            fill it to 20,000 by (1000.0,1000.0)
        """
        out_set = set()
        for input_array in list(input_arrays):
            for [x1, y1, x2, y2] in list(input_array):
                current_list = self.get_arrange(x1, y1, x2, y2)
                for current_dots in current_list:
                    current_dots[0] = float(format(current_dots[0], ".2f"))
                    current_dots[1] = float(format(current_dots[1], ".2f"))
                    if (
                        current_dots[0] < self.vcs_range[2]
                        and current_dots[0] > self.vcs_range[0]
                        and current_dots[1] < self.vcs_range[3]
                        and current_dots[1] > self.vcs_range[1]
                    ):
                        out_set.add(
                            str(current_dots[0]) + "," + str(current_dots[1])
                        )
        out = [one.split(",") for one in list(out_set)]
        out = [[float(one[0]), float(one[1])] for one in out]
        assert len(out) < self.max_anno_num
        out.extend([[1000.0, 1000.0]] * (self.max_anno_num - len(out)))
        return out

    def get_validation_anno_target(self, input_dict):
        output_dict = {}
        for (category, edges) in input_dict.items():
            if edges is not None:
                edges_new = self.convert_points(edges)
                output_dict[category] = edges_new
        return output_dict

    def __call__(self, data):

        data["gt_bev_seg_anno"] = self.get_validation_anno_target(
            data["gt_bev_static_anno"]
        )
        return data

    def __repr__(self):
        return "BevSegAnnoGenerator"


@OBJECT_REGISTRY.register
class Crop3DV(object):  # noqa: D205,D400
    """Crop the given list of image at specified location/output size
     and modify intrinsics/homography matrix. \

     The image can be a PIL Image or a Tensor,
     in which case it is expected to have [..., H, W] shape,
     where ... means an arbitrary number of leading dimensions.

    Args:
    heights (sequence,int): Height of the crop boxs in each view.
    widths (sequence,int): Width of the crop boxs in each view.
    top (sequence, int): Vertical component of the top left corner \
        of the crop box in each view. Setting None means random crop.
    left (sequence, int): Horizontal component of the top left corner \
        of the crop box in each view. Setting None means random crop.
    """

    def __init__(
        self,
        height: Union[Sequence, int],
        width: Union[Sequence, int],
        top: Union[Sequence, int],
        left: Union[Sequence, int],
    ):

        self.top = top if isinstance(top, Sequence) else [top]
        self.left = left if isinstance(left, Sequence) else [left]
        self.height = height if isinstance(height, Sequence) else [height]
        self.width = width if isinstance(width, Sequence) else [width]

    def _crop(
        self, data: Union[Image.Image, Sequence], top, left, height, width
    ):
        if isinstance(data, Sequence):
            assert (
                len(data) == len(top) == len(left) == len(height) == len(width)
            ), "please check length of data and length of argument, must be same"  # noqa
            return [
                self._crop(_, t, l, h, w)
                for _, t, l, h, w in zip(data, top, left, height, width)
            ]
        else:
            return F.crop(data, top, left, height, width)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'

        nums = len(data["size"])
        assert (
            len(data["pil_imgs"][0])
            == len(self.top)
            == len(self.left)
            == len(self.height)
            == len(self.width)
        )

        for i in range(nums):
            assert data["size"][i][0] >= self.height[i]
            assert data["size"][i][1] >= self.width[i]

        # generate a random integer if self.top is None
        top = [
            np.random.randint(data["size"][i][0] - self.height[i] + 1)
            if self.top[i] is None
            else self.top[i]
            for i in range(nums)
        ]
        left = [
            np.random.randint(data["size"][i][1] - self.width[i] + 1)
            if self.left[i] is None
            else self.left[i]
            for i in range(nums)
        ]

        data["pil_imgs"] = [
            self._crop(pil_img, top, left, self.height, self.width)
            for pil_img in data["pil_imgs"]
        ]

        if "gt_seg" in data:
            data["gt_seg"] = self._crop(
                data["gt_seg"], top, left, self.height, self.width
            )

        if "obj_mask" in data:
            data["obj_mask"] = self._crop(
                data["obj_mask"],
                top[0],
                left[0],
                self.height[0],
                self.width[0],
            )

        if "gt_depth" in data:
            for i in range(len(data["gt_depth"])):
                depth_w, depth_h = data["gt_depth"][i].size
                scale_h, scale_w = (
                    depth_h // data["size"][i][0],
                    depth_w // data["size"][i][1],
                )
                data["gt_depth"][i] = self._crop(
                    data["gt_depth"][i],
                    top[i] * scale_h,
                    left[i] * scale_w,
                    self.height[i] * scale_h,
                    self.width[i] * scale_w,
                )

        if "color_imgs" in data:
            data["color_imgs"] = [
                self._crop(color_img, top, left, self.height, self.width)
                for color_img in data["color_imgs"]
            ]

        if "front_mask" in data:
            data["front_mask"] = self._crop(
                data["front_mask"],
                top[0],
                left[0],
                self.height[0],
                self.width[0],
            )

        if "intrinsics" in data:
            # modify intrinsics matrix
            data["intrinsics"][0, 2] -= left[0]
            data["intrinsics"][1, 2] -= top[0]

        if "homography" in data:
            # modify homography matrix
            for i in range(len(top)):
                trans_mat = np.eye(3, dtype="float32")
                trans_mat[0, 2] = -left[i]
                trans_mat[1, 2] = -top[i]
                data["homography"][i] = trans_mat @ data["homography"][i]
        return data

    def __repr__(self):
        return "Crop3DV"


@OBJECT_REGISTRY.register
class ResizeHomo(object):  # noqa: D205,D400
    """Modify homography matrix with specified scale.

    Args:
        H_persp_view_scale (int, None): perspective view scale factor of
            homography matrix.
        H_bev_view_scale (int, None): BEV scale factor of homography matrix.

    NOTE:
    If perspective view`s size of original homography matrix is (h_pers,w_pers),  # noqa
    BEV`s size of original homography matrix is (x_bev,y_bev),
    now you want to change perspective view`s size to (h_pers*scale_p,w_pers*scale_p),  # noqa
    and change BEV`s size to (x_bev*scale_b, y_bev*scale_b),
    please set H_persp_view_scale to scale_p and H_bev_view_scale to scale_b.


    """

    def __init__(
        self,
        H_persp_view_scale: Optional[float] = 1.0,
        H_bev_view_scale: Optional[float] = 1.0,
    ):
        self.H_persp_view_scale = H_persp_view_scale
        self.H_bev_view_scale = H_bev_view_scale

    def __call__(self, data):
        if "homography" in data:
            # modify homography matrix

            scale_mat = np.eye(3, dtype="float32")
            scale_mat[0, 0] = self.H_persp_view_scale
            scale_mat[1, 1] = self.H_persp_view_scale

            bev_scale_mat = np.eye(3, dtype="float32")
            bev_scale_mat[0, 0] = self.H_bev_view_scale
            bev_scale_mat[1, 1] = self.H_bev_view_scale

            data["homography"] = (
                scale_mat @ data["homography"] @ np.linalg.inv(bev_scale_mat)
            )  # noqa

        return data


@OBJECT_REGISTRY.register
class Pad3DV(object):  # noqa: D205,D400
    """Pad the given list of image and modify intrinsics/homography matrix.

     The image can be a PIL Image or a Tensor,
     in which case it is expected to have [..., H, W] shape,
     where ... means an arbitrary number of leading dimensions.

    Args:
        paddings (sequence): Padding on each border of multi view data.
            NOTE: usage of padding is same as F.pad().
            NOTE: make sure the length of paddings must be equal to length of imgs.  # noqa
        img_fill (int): Pixel fill value for image.
        gt_seg_fill (int) : Pixel fill value for gt seg.
        gt_depth_fill (int) : Pixel fill value for gt depth.
    """

    def __init__(
        self,
        paddings: Union[int, Sequence],
        img_fill: int = 0,
        gt_seg_fill: int = -1,
        gt_depth_fill: int = 0,
    ):
        self.paddings = paddings
        self.img_fill = img_fill
        self.gt_seg_fill = gt_seg_fill
        self.gt_depth_fill = gt_depth_fill

    def _pad(
        self,
        data: Union[Image.Image, Sequence],
        padding,
        fill,
    ):
        if isinstance(data, Sequence):
            assert len(data) == len(padding)
            return [self._pad(_, p, fill) for _, p in zip(data, padding)]
        else:
            return F.pad(data, padding, fill)

    def __call__(self, data: Mapping):

        assert len(data["size"]) == len(self.paddings)

        assert "pil_imgs" in data, 'input data must has "pil_imgs"'

        data["pil_imgs"] = [
            self._pad(pil_img, self.paddings, fill=self.img_fill)
            for pil_img in data["pil_imgs"]
        ]

        if "gt_seg" in data:
            data["gt_seg"] = self._pad(
                data["gt_seg"], self.paddings, fill=self.gt_seg_fill
            )

        if "obj_mask" in data:
            data["obj_mask"] = self._pad(
                data["obj_mask"], self.paddings[0], fill=0
            )  # noqa

        if "gt_depth" in data:
            data["gt_depth"] = self._pad(
                data["gt_depth"], self.paddings, fill=self.gt_depth_fill
            )

        if "color_imgs" in data:
            data["color_imgs"] = [
                self._pad(color_img, self.paddings, self.img_fill)
                for color_img in data["color_imgs"]
            ]

        if "front_mask" in data:
            data["front_mask"] = self._pad(
                data["front_mask"], self.paddings[0], fill=0
            )  # noqa

        if "intrinsics" in data:
            # modify intrinsics matrix
            front_padding = self.paddings[0]
            left = (
                front_padding[0]
                if isinstance(front_padding, Sequence)
                else front_padding
            )
            top = (
                front_padding[1]
                if isinstance(front_padding, Sequence)
                else front_padding
            )

            data["intrinsics"][0, 2] += top
            data["intrinsics"][1, 2] += left

        if "homography" in data:
            # modify homography matrix
            for i, padding in enumerate(self.paddings):
                # 1.padding could be a single int is used to pad all borders.
                # 2.padding could be a sequence of length 2 means
                #   left/right and top/bottom respectively.
                # 3.padding could be a sequence of length 4 means
                #   left, top, right and bottom.

                left = padding[0] if isinstance(padding, Sequence) else padding
                top = padding[1] if isinstance(padding, Sequence) else padding
                trans_mat = np.eye(3, dtype="float32")
                trans_mat[0, 2] = left
                trans_mat[1, 2] = top
                data["homography"][i] = trans_mat @ data["homography"][i]

        return data

    def __repr__(self):
        return "Pad3DV"


@OBJECT_REGISTRY.register
class ClassRemap(object):
    """Remap segmentation gt with specific remap dict.

    Args:
        remap_dict dict[str: dict]: remap dict.

    """

    def __init__(self, remap_dict: Optional[Mapping] = None):
        self.remap_dict = remap_dict if remap_dict else {}

    def _remap(self, data, remap_dict):
        if isinstance(data, Sequence):
            return [self._remap(_, remap_dict) for _ in data]
        else:
            return np.vectorize(remap_dict.get)(np.array(data))

    def __call__(self, data):

        for key_name, each_map in self.remap_dict.items():
            assert key_name in data
            data[key_name] = self._remap(data[key_name], each_map)

        return data

    def __repr__(self):
        return "ClassRemap"


@OBJECT_REGISTRY.register
class ToTensor3DV(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to torch.tensor.

    Args:
    to_yuv (bool): wheather convert to yuv444 format durning quanti training.
    """

    def __init__(self, to_yuv=False, with_color_imgs=True):
        self.to_yuv = to_yuv
        self.with_color_imgs = with_color_imgs
        self.pil2yuv = torchvision.transforms.Compose(
            [torchvision.transforms.PILToTensor(), BgrToYuv444(rgb_input=True)]
        )

    def _to_tensor(self, data: Union[Image.Image, Sequence], as_numpy=False):
        # convert a pillow img to tensor(normlizae to [0,1])
        # F.to_tensor will normlize a pillow img to [0,1] defaultly,
        # depth and seg do not need normlization so wo convert depth and seg
        # data to numpy first.
        if isinstance(data, Sequence):
            return [self._to_tensor(_) for _ in data]
        else:
            return F.to_tensor(np.array(data) if as_numpy else data)

    def _to_yuv_tensor(self, data):
        # step1.convert a pil img to tensor(do not normlization)
        # step2.convart a rgb tensor to yuv tensor

        if isinstance(data, Sequence):
            return [self._to_yuv_tensor(_) for _ in data]
        else:
            return self.pil2yuv(data)

    def _undistort_coord(self, inputsize, intrinsic, distor_coeff):
        # scale intrinsics matrix
        intrinsic = intrinsic.copy()
        # intrinsic[0, :] *= inputsize[1]     #width
        # intrinsic[1, :] *= inputsize[0]     # high

        mapx, mapy = cv2.initUndistortRectifyMap(
            intrinsic,
            distor_coeff,
            None,
            intrinsic,
            (inputsize[1], inputsize[0]),
            cv2.CV_32FC1,
        )

        mapx = (mapx - inputsize[1] * 0.5) / (inputsize[1] * 0.5)
        mapy = (mapy - inputsize[0] * 0.5) / (inputsize[0] * 0.5)

        img_ud_coord = np.concatenate(
            [mapx[:, :, np.newaxis], mapy[:, :, np.newaxis]], axis=2
        )
        return img_ud_coord

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'
        if "color_imgs" in data:
            data["color_imgs"] = self._to_tensor(data["color_imgs"])
            # generate undistorted image to calcu project loss that are normalized to [0,1] # noqa
            data["ud_coord"] = self._undistort_coord(
                data["color_imgs"][0][0].shape[1:3],
                data["intrinsics"],
                data["distortcoef"],
            )
            data["ud_coord"] = torch.from_numpy(data["ud_coord"])
            # generate undistorted coordinates for depth and resflow
        else:
            data["color_imgs"] = self._to_tensor(data["pil_imgs"])

        if self.to_yuv:
            data["imgs"] = self._to_yuv_tensor(data["pil_imgs"])
            # generate input yuv img for quanti training
        else:
            data["imgs"] = self._to_tensor(data["pil_imgs"])
            # generate input rgb img normalized to [0,1] for float training

        if "gt_seg" in data:
            data["gt_seg"] = self._to_tensor(data["gt_seg"], as_numpy=True)

        if "obj_mask" in data:
            data["obj_mask"] = self._to_tensor(data["obj_mask"], as_numpy=True)

        if "gt_depth" in data:
            data["gt_depth"] = self._to_tensor(data["gt_depth"], as_numpy=True)

        if "front_mask" in data:
            data["front_mask"] = self._to_tensor(
                data["front_mask"], as_numpy=True
            )

        if "intrinsics" in data:
            data["intrinsics"] = torch.from_numpy(data["intrinsics"])

        if "distortcoef" in data:
            data["distortcoef"] = torch.from_numpy(data["distortcoef"])

        if "homography" in data:
            data["homography"] = torch.from_numpy(data["homography"])

        if "homo_offset" in data:
            data["homo_offset"] = torch.from_numpy(data["homo_offset"])

        if "gt_bev_seg" in data:
            data["gt_bev_seg"] = self._to_tensor(
                data["gt_bev_seg"], as_numpy=True
            )

        if "occlusion" in data:
            data["occlusion"] = self._to_tensor(
                data["occlusion"], as_numpy=True
            )

        if "gt_bev_discrete_obj" in data:
            for k in data["gt_bev_discrete_obj"].keys():
                data["gt_bev_discrete_obj"][k] = self._to_tensor(
                    data["gt_bev_discrete_obj"][k], as_numpy=True
                )
            assert "annos_bev_discrete_obj" in data
            for k in data["annos_bev_discrete_obj"].keys():
                data["annos_bev_discrete_obj"][k] = torch.from_numpy(
                    data["annos_bev_discrete_obj"][k]
                )

        if "gt_bev_seg_anno" in data:
            _gt_bev_seg_anno = {}
            for k in data["gt_bev_seg_anno"].keys():
                if data["gt_bev_seg_anno"][k] is not None:
                    _gt_bev_seg_anno[k] = torch.from_numpy(
                        np.array(data["gt_bev_seg_anno"][k])
                    )
            if "occlusion" in data:
                # occlusion need to be B*H*W in bevseg instance eval
                _gt_bev_seg_anno["occlusion"] = copy.deepcopy(
                    torch.squeeze(data["occlusion"])
                )
            data["gt_bev_seg_anno"] = _gt_bev_seg_anno

        if "om_target" in data:
            for k, v in data["om_target"].items():
                data["om_target"][k] = torch.from_numpy(v)

        if "timestamp" in data:
            data["timestamp"] = torch.from_numpy(data["timestamp"])

        if "front_bool_idx" in data:
            data["front_bool_idx"] = torch.from_numpy(data["front_bool_idx"])

        if "axisangle" in data:
            data["axisangle"] = torch.from_numpy(data["axisangle"])
            data["translation"] = torch.from_numpy(data["translation"])

        if not self.with_color_imgs:
            data.pop("color_imgs")
        data.pop("pil_imgs")
        return data

    def __repr__(self):
        return "ToTensor3DV"


@OBJECT_REGISTRY.register
class Normalize3DV(object):  # noqa: D205,D400
    """Normalize a tensor image with mean and standard deviation,and
       scale gt depth with a specified coefficient if need.

    Args:
    mean (float or sequence of float): Sequence of means for each channel.
    std (float or sequence of float): Sequence of std for each channel.
    depth_scale (float): coefficient to scale depth.
    """

    def __init__(self, mean, std, depth_scale=1.0):
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def _normalize(self, data: Union[Image.Image, Sequence], mean, std):
        if isinstance(data, Sequence):
            return [self._normalize(_, mean, std) for _ in data]
        else:
            return F.normalize(data, mean, std)

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["imgs"] = self._normalize(data["imgs"], self.mean, self.std)

        if "gt_depth" in data:
            data["gt_depth"] = [
                depth * self.depth_scale for depth in data["gt_depth"]
            ]
        return data

    def __repr__(self):
        return "Normalize3DV"


@OBJECT_REGISTRY.register
class MotionFlowGenerator(object):
    """Generate gt bev motion flow map from 3d bbox annotation of two frames.

    Args:
    bev_size (sequence of float): bev size.(order is (h,w))
    vcs_range (sequence of float): vcs range.(order is (bottom,right,top,left))
    ego_loc (sequence offloat): location of ego car in bev pixel coordinate.
        (order is (h,w))
    """

    def __init__(
        self,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        ego_loc: Sequence[int],
    ):
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x
        self.ego_loc = ego_loc

    def pad_to_square(self, array, pad_value=0):
        """Pad the array to square array.

        Args:
        kernel (ndarray): the input numpy array
        pad_value (int): value used for pad
        """
        h, w = array.shape[0:2]
        pad_dim = max(h, w) + w // 2
        before_1 = after_1 = int((pad_dim - h) // 2)
        before_2 = after_2 = int((pad_dim - w) // 2)
        padded_array = np.pad(
            array,
            ((before_1, after_1), (before_2, after_2), (0, 0)),  # noqa
            "constant",
            constant_values=pad_value,
        )
        return padded_array

    def rotate(self, image, angle, center=None, scale=1.0):
        """Rotate the input img or array.

        Args:
        image: (ndarray) input iamge
        angle: (float) rotate angle, degree
        center: (tuple) center to rotate
        scale: (float) transform scale

        """
        (h, w) = image.shape[:2]
        max_dim = max((h, w))
        output_dim = (max_dim, max_dim)

        if center is None:
            center = tuple((np.array(image.shape[:2]) // 2).astype("float"))

        # Perform the rotation
        # NOTE: The angle is degree
        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(image, M, output_dim, flags=cv2.INTER_NEAREST)

        return rotated

    def __call__(self, data):
        """Generate gt bev motion flow map.

        Args:
            data (dict): Type is ndarray
                The dict contains at leaset annotations

        Returns (dict): Type is ndarray

        """
        assert "gt_bev_motion_flow" in data
        annotations = data.pop("gt_bev_motion_flow")

        bev_motion_flow = np.zeros((*self.bev_size, 3), dtype=np.float32)

        cur_frame, pre_frame = annotations
        cur_ego_location = cur_frame.pop("ego_location")
        cur_ego_yaw = cur_frame.pop("ego_yaw")

        cur_seq_center = SeqCenter(
            pos_x=cur_ego_location[0],
            pos_y=cur_ego_location[1],
            yaw=cur_ego_yaw,
        )

        obs_d_yaw = []
        pre_obs_location = []
        obs_w_bev, obs_h_bev, obs_location, obs_yaw_vcs = [], [], [], []
        for track_id, obs_data in cur_frame.items():
            if track_id not in pre_frame:
                continue
            obs_d_yaw.append(
                obs_data["obs_yaw"] - pre_frame[track_id]["obs_yaw"]
            )

            obs_w_bev.append(
                obs_data["obs_dimension"][0] / self.m_perpixel[1]
            )  # pixel
            obs_h_bev.append(
                obs_data["obs_dimension"][1] / self.m_perpixel[0]
            )  # pixel

            obs_yaw_vcs.append(obs_data["obs_yaw"] - cur_ego_yaw)
            obs_location.append(np.array(obs_data["obs_location"]))
            pre_obs_location.append(
                np.array(pre_frame[track_id]["obs_location"])
            )

        obs_num = len(obs_location)
        obs_vcs_coord = TdtCoordHelper.global_phy_to_local_phy(
            np.stack(obs_location + pre_obs_location),
            cur_seq_center,
        )

        obs_vcs_ct = obs_vcs_coord[:obs_num]
        pre_obs_vcs_ct = obs_vcs_coord[obs_num:]

        obs_dxy_vcs = obs_vcs_ct - pre_obs_vcs_ct
        obs_dx_vcs, obs_dy_vcs = obs_dxy_vcs[:, 0], obs_dxy_vcs[:, 1]

        # convert vcs to bev
        obs_bev_ct = (
            -obs_vcs_ct / np.array(self.m_perpixel) + np.array(self.ego_loc)
        )[:, [1, 0]]

        for bev_ct, w, h, yaw, d_x, d_y, d_yaw, in zip(
            obs_bev_ct,
            obs_w_bev,
            obs_h_bev,
            obs_yaw_vcs,
            obs_dx_vcs,
            obs_dy_vcs,
            obs_d_yaw,
        ):
            if (
                0 <= bev_ct[0] < self.bev_size[1]
                and 0 <= bev_ct[1] < self.bev_size[0]
            ):
                reg_map = get_reg_map((int(w), int(h)), (d_x, d_y, d_yaw))
                reg_map = self.pad_to_square(reg_map, 0)
                # rotate the insert hm
                reg_map = self.rotate(reg_map, np.rad2deg(yaw))
                draw_heatmap(bev_motion_flow, reg_map, bev_ct, op="overwrite")
        data["gt_bev_motion_flow"] = bev_motion_flow
        return data

    def __repr__(self):
        return "MotionFlowGenerator"


@OBJECT_REGISTRY.register
class Bev3dTargetGenerator(object):
    """Generate gound truth labels for bev_3d.

    Args:
        num_classes (int): Number of classes
        bev_size (sequence of int): bev size.(order is (h,w))
        vcs_range (sequence of float): vcs range.(order is
            (bottom,right,top,left))
        cls_hm_kernel (dict): the gaussian kernel size of bev
            heatmap for each category.(e.g. {cls1: 3})
        category2id_map (dict): A mapping from raw category (str)
            to training category_id which starts from 0
        cls_dimension (np.ndarray): the average dimension of each category,
            each column stands for a class.
        max_objs (int): Maximum number of objects used in the training and
            inference. This number should be large enough
        enable_ignore (bool): Whether to use ignore_mask. If true, the
            ignored object will be drew in the bev3d_ignore_mask heatmap
            and not participate in the loss calculation, else the object
            will be regarded as the background.

    """

    def __init__(
        self,
        num_classes: int,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        cls_hm_kernel: Mapping,
        category2id_map: Mapping,
        cls_dimension: np.ndarray = None,
        max_objs: int = 100,
        enable_ignore: bool = True,
    ):
        self.num_classes = num_classes
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.max_objs = max_objs
        self.cls_dimension = cls_dimension
        self.category2id_map = category2id_map
        self.enable_ignore = enable_ignore

        self.m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x

        self.cls2kernel = {}
        for category, kernel_size in cls_hm_kernel.items():
            self.cls2kernel[category] = np.array(
                [kernel_size, kernel_size], dtype=np.float32
            )

    def get_ctoff_map(self, wh, center):
        """Calculate the regression map of bev center offset.

        Args:
            wh (int): the inserted kernel size.(order (w,h))
            center (float): object center in bev.(order u, w)

        Returns:
            (ndarray): bev center offset regression map
        """

        w, h = wh
        radius = (w // 2, h // 2)
        n, m = radius
        center_int = (int(center[0]), int(center[1]))
        center_offset = np.array(center) - np.array(center_int)
        x, y = center_int

        y_grid = np.arange(y - m, y + m + 1)
        x_grid = np.arange(x - n, x + n + 1)

        y_reg = center[1] - y_grid
        x_reg = center[0] - x_grid

        y_reg[m] = center_offset[1]
        x_reg[n] = center_offset[0]
        xv, yv = np.meshgrid(x_reg, y_reg)
        ct_off_reg_map = np.concatenate(
            [xv[:, :, np.newaxis], yv[:, :, np.newaxis]], axis=-1
        )
        return ct_off_reg_map

    def __call__(self, data):
        """Generate bev_3d labels.

        Args:
            data (dict): Type is ndarray
                The dict contains at leaset annotations

        Returns (dict): Type is ndarray

        """
        assert "gt_bev_3d" in data
        annotations = data.pop("gt_bev_3d")

        # build bev3d gt heatmaps
        bev3d_hm = np.zeros(
            (*self.bev_size, self.num_classes), dtype=np.float32
        )
        bev3d_dim = np.zeros((*self.bev_size, 3), dtype=np.float32)  # h, w, l
        bev3d_rot = np.zeros((*self.bev_size, 2), dtype=np.float32)  # cos sin
        bev3d_ct_offset = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # bev center offest
        bev3d_loc_z = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_weight_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_point_pos_mask = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_ignore_mask = np.zeros((self.bev_size), dtype=np.float32)

        # used for eval
        vcs_loc_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_dim_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_rot_z_ = np.zeros((self.max_objs), dtype=np.float32)
        vcs_cls_ = np.zeros((self.max_objs), dtype=np.float32) - 99
        vcs_ignore_ = np.zeros((self.max_objs), dtype=np.bool)
        vcs_visible_ = np.zeros((self.max_objs), dtype=np.float32)

        count_id = 0
        for ann_idx, anno in enumerate(annotations):  # noqa [B007]
            if count_id > (self.max_objs):
                break
            cls_id = int(self.category2id_map[anno["label"]])
            if cls_id <= -99 or (
                not self.enable_ignore and anno.get("ignore", False)
            ):
                continue
            vcs_loc = anno["location"]
            bev_ct = (
                ((self.vcs_range[3] - vcs_loc[1]) / self.m_perpixel[1]),
                ((self.vcs_range[2] - vcs_loc[0]) / self.m_perpixel[0]),
            )  # (u, v)
            bev_ct_int = (int(bev_ct[0]), int(bev_ct[1]))

            if (
                0 <= bev_ct_int[0] < self.bev_size[1]
                and 0 <= bev_ct_int[1] < self.bev_size[0]
            ):

                vcs_loc_[count_id] = vcs_loc
                vcs_cls_[count_id] = cls_id
                vcs_dim_[count_id] = anno["dimension"]
                # transfer the yaw to value [-pi,pi]
                vcs_yaw = np.arctan2(np.sin(anno["yaw"]), np.cos(anno["yaw"]))
                vcs_rot_z_[count_id] = vcs_yaw
                if anno.get("ignore", False):
                    vcs_ignore_[count_id] = 1
                # Visibility reflects the occlusion degree of the target
                if anno.get("visibility", False):
                    vcs_visible_[count_id] = anno["visibility"]

                # draw bev3d heatmap
                insert_bev_hm = get_gaussian2D(
                    self.cls2kernel[cls_id], alpha=1
                )
                insert_bev_hm_wh = insert_bev_hm.shape[:2][::-1]
                if anno.get("ignore", False):
                    insert_ignore_mask = np.ones_like(insert_bev_hm)
                else:
                    insert_ignore_mask = np.zeros_like(insert_bev_hm)
                if self.cls_dimension is not None:
                    ori_dim = np.array(anno["dimension"])
                    avg_dim = self.cls_dimension[cls_id]
                    residual_dim = np.log(ori_dim / avg_dim)
                    ann_dim = list(residual_dim)
                else:
                    ann_dim = anno["dimension"]
                insert_bev_reg_map_list = [
                    get_reg_map(insert_bev_hm_wh, ann_dim),
                    get_reg_map(
                        insert_bev_hm_wh, (np.cos(vcs_yaw), np.sin(vcs_yaw))
                    ),
                    get_reg_map(insert_bev_hm_wh, vcs_loc[-1]),
                    self.get_ctoff_map(insert_bev_hm_wh, bev_ct),
                ]
                bev_reg_map_list = [
                    bev3d_dim,
                    bev3d_rot,
                    bev3d_loc_z,
                    bev3d_ct_offset,
                ]
                draw_heatmap(bev3d_hm[:, :, cls_id], insert_bev_hm, bev_ct_int)
                draw_heatmap(
                    bev3d_weight_hm,
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )
                draw_heatmap(bev3d_ignore_mask, insert_ignore_mask, bev_ct_int)

                bev3d_point_pos_mask[bev_ct_int[1], bev_ct_int[0]] = 1
                count_id += 1

        gt_bev_3d = {
            "bev3d_hm": bev3d_hm,
            "bev3d_dim": bev3d_dim,
            "bev3d_rot": bev3d_rot,
            "bev3d_ct_offset": bev3d_ct_offset,
            "bev3d_loc_z": bev3d_loc_z[:, :, np.newaxis],
            "bev3d_weight_hm": bev3d_weight_hm[:, :, np.newaxis],
            "bev3d_point_pos_mask": bev3d_point_pos_mask[:, :, np.newaxis],
            "bev3d_ignore_mask": bev3d_ignore_mask[:, :, np.newaxis],
        }

        annos_bev_3d = {
            "vcs_loc_": vcs_loc_,
            "vcs_cls_": vcs_cls_,
            "vcs_rot_z_": vcs_rot_z_,
            "vcs_dim_": vcs_dim_,
            "vcs_ignore_": vcs_ignore_,
            "vcs_visible_": vcs_visible_,
        }

        data["gt_bev_3d"] = gt_bev_3d
        data["annos_bev_3d"] = annos_bev_3d
        return data

    def __repr__(self):
        return "Bev3dTargetGenerator"


@OBJECT_REGISTRY.register
class BevDiscreteTargetGenerator(object):
    """Generate ground truth labels for bev discrete objects.

    Discrete objects includes arrow, crosswalk, junction, etc.
    (TODO: Integrate BevDiscreteTargetGenerator into
    Bev3dTargetGenerator, ben.hu, xiangyu.li)

    Args:
        num_classes (int): Number of classes
        bev_size (sequence of int): bev size.(order is (h,w))
        vcs_range (sequence of float): vcs range.(order is
            (bottom,right,top,left))
        category2id_map (dict): A mapping from raw category (str)
            to training category_id which starts from 0
        name2label (dict):  A mapping from multimodal lmdb data name to label.
        max_objs (int): Maximum number of objects used in the training and
            inference. This number should be large enough
        vcs_bbox_area_thresh (float): filter the bbox area less than
            this number. default is set to 0.
    """

    def __init__(
        self,
        num_classes: int,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        category2id_map: Mapping,
        name2label: Mapping,
        max_objs: int = 100,
        vcs_bbox_area_thresh: float = 0.0,
    ):
        self.num_classes = num_classes
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.category2id_map = category2id_map
        self.max_objs = max_objs
        self.vcs_bbox_area_thresh = vcs_bbox_area_thresh
        self.name2label = name2label

        self.m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x

    def get_vertices_from_vcs_box(self, wh, ct, yaw):
        """Get vertices of bounding box with format (w,h,cx,cy, yaw) in vcs.

        Args:
            wh: [w, h] in vcs along yaw direction and vertical to yaw.
            ct: [cx, cy], coordinate of center of bounding box in vcs.
            yaw: yaw of bounding box in vcs.

        Returns: coordinate (x, y) in vcs, 4 vertices of a bbox.

        """
        w, h = wh
        ctx, cty = ct

        p0 = [0.5 * w, 0.5 * h, 1]
        p1 = [0.5 * w, -0.5 * h, 1]
        p2 = [-0.5 * w, -0.5 * h, 1]
        p3 = [-0.5 * w, 0.5 * h, 1]
        points = np.array([p0, p1, p2, p3])
        R = np.array(
            [
                [np.cos(yaw), np.sin(yaw), ctx],
                [-np.sin(yaw), np.cos(yaw), cty],
                [0, 0, 1],
            ]
        )
        points = R.dot(points.T).T
        return points[:, :2]

    def vcs2bev_coord(self, pt):
        """Convert point pt(x, y) in vcs to bev u-v coordinate.

        Args:
            pt: [x, y] coordinate of point in vcs

        Returns: (u, v) in bev u-v coordinate

        """
        u = int((self.vcs_range[3] - pt[1]) / self.m_perpixel[1])
        v = int((self.vcs_range[2] - pt[0]) / self.m_perpixel[0])
        return [u, v]

    def get_rotated_gaussian2D(self, wh, yaw, alpha=0.54, sigma=None):
        """Create Gaussian heatmap with rotation.

        Args:
            wh: [w, h] in vcs along yaw direction and vertical to yaw.
            yaw: yaw of bounding box in vcs.
            alpha: heatmap scaling.
            sigma: Gaussian kernel standard deviation.

        Returns: heatmap of bbox.

        """
        radius = (np.array([*wh]) / 2 * alpha).astype("int32")
        if sigma is None:
            sigma = np.array(
                [[(radius[0] * 2 + 1) / 6, 0], [0, (radius[1] * 2 + 1) / 6]]
            )
        else:
            sigma = np.array(sigma)
        var = sigma * sigma

        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        var_rotated = np.matmul(R, np.matmul(var, R.transpose()))

        points = self.get_vertices_from_vcs_box(wh, (0, 0), yaw)
        w_rotated = points[:, 0].max() - points[:, 0].min()
        h_rotated = points[:, 1].max() - points[:, 1].min()

        rw, rh = int(w_rotated / 2), int(h_rotated / 2)
        x, y = np.ogrid[-rw : rw + 1, -rh : rh + 1]
        xy_grid = np.array(np.meshgrid(x, y), dtype=np.float32).transpose(
            1, 2, 0
        )

        heatmap = np.exp(
            -np.sum(
                np.matmul(xy_grid, np.linalg.inv(var_rotated)) * xy_grid,
                axis=-1,
            )
            * 0.5
        )
        heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
        assert heatmap.max() == 1, "heatmap value max must be 1!"
        return heatmap

    def get_annotations(self, lines, label, timestamp):
        """Create annotation from lmdb data.

        Args:
            lines: multiple sides of the target box.
            label: label of the target.
            timestamp: the timestamp when the target was generated.

        Return:
            anno: the annotations of the target.

        """
        pt_csv = np.array(lines).reshape((-1, 2)).astype(np.float32)
        (cx, cy), (l, w), yaw = cv2.minAreaRect(pt_csv)
        if l < w:
            length, width = w, l
            if yaw > 0:
                yaw = -(90 - yaw)
            else:
                yaw = 90 + yaw
        else:
            length, width = l, w

        yaw = yaw * np.pi / 180
        length = np.clip(length, 1, 1000)
        width = np.clip(width, 0.5, 1000)
        anno = {
            "dimension": [length, width],
            "yaw": yaw,
            "location": [cx, cy],
            "area": length * width,
            "label": label,
            "ignore": False,
            "timestamp": timestamp,
        }
        return anno

    def __call__(self, data):
        """Generate bev discrete object labels, e.g., arrow, crosswalk, etc.

        Args:
            data (dict): Type is ndarray
                The dict contains at least annotations

        Returns (dict): Type is ndarray

        """
        assert "gt_bev_discrete_obj" in data
        objs = data.pop("gt_bev_discrete_obj")
        timestamp = str(int(data["timestamp"][0] * 1000))
        annotations = []
        for name, label in self.name2label.items():
            for obj in objs[name]:
                anno = self.get_annotations(obj, label, timestamp)
                annotations.append(anno)

        bev_discobj_hm = np.zeros(
            (*self.bev_size, self.num_classes), dtype=np.float32
        )
        bev_discobj_wh = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # w, h
        bev_discobj_rot = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # cos, sin
        bev_discobj_weight_hm = np.zeros((self.bev_size), dtype=np.float32)

        # used for eval
        vcs_discobj_loc = np.zeros((self.max_objs, 2), dtype=np.float32)
        vcs_discobj_dim = np.zeros((self.max_objs, 2), dtype=np.float32)
        vcs_discobj_cls = np.zeros((self.max_objs), dtype=np.float32) - 1
        vcs_discobj_valid = np.zeros((self.max_objs,), dtype=np.float32)
        vcs_discobj_yaw = np.zeros((self.max_objs,), dtype=np.float32)

        count_id = 0
        for anno_idx, anno in enumerate(annotations):  # noqa [B007]
            if count_id >= self.max_objs:
                break
            if anno["area"] < self.vcs_bbox_area_thresh or anno.get(
                "ignore", False
            ):
                continue

            cls_id = int(self.category2id_map[anno["label"]])
            vcs_yaw = anno["yaw"]  # in rad
            vcs_dim = anno["dimension"]  # (w, h), w along yaw direction
            vcs_loc = anno["location"]  # (x, y) in vcs
            bev_ct_int = self.vcs2bev_coord(vcs_loc)
            bev_dim = [
                vcs_dim[0] / self.m_perpixel[0],
                vcs_dim[1] / self.m_perpixel[1],
            ]
            if (
                0 <= bev_ct_int[0] < self.bev_size[1]
                and 0 <= bev_ct_int[1] < self.bev_size[0]
            ):
                insert_bev_hm = self.get_rotated_gaussian2D(
                    bev_dim, vcs_yaw + np.pi / 2, alpha=1.0
                )
                insert_bev_wh = insert_bev_hm.shape[:2][::-1]
                insert_bev_reg_map_list = [
                    get_reg_map(insert_bev_wh, vcs_dim),
                    get_reg_map(
                        insert_bev_wh, (np.cos(vcs_yaw), np.sin(vcs_yaw))
                    ),
                ]
                bev_reg_map_list = [
                    bev_discobj_wh,
                    bev_discobj_rot,
                ]
                draw_heatmap(
                    bev_discobj_hm[:, :, cls_id], insert_bev_hm, bev_ct_int
                )
                draw_heatmap(
                    bev_discobj_weight_hm,
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )

                vcs_discobj_loc[count_id] = vcs_loc  # (x, y) in vcs
                vcs_discobj_dim[count_id] = vcs_dim
                vcs_discobj_valid[count_id] = 1
                vcs_discobj_cls[count_id] = cls_id
                vcs_discobj_yaw[count_id] = vcs_yaw

                count_id += 1

        gt_bev_discrete_obj = {
            "bev_discobj_hm": bev_discobj_hm,
            "bev_discobj_wh": bev_discobj_wh,
            "bev_discobj_rot": bev_discobj_rot,
            "bev_discobj_weight_hm": bev_discobj_weight_hm,
        }

        annos_bev_discrete_obj = {
            "vcs_discobj_loc": vcs_discobj_loc,
            "vcs_discobj_wh": vcs_discobj_dim,
            "vcs_discobj_valid": vcs_discobj_valid,
            "vcs_discobj_cls": vcs_discobj_cls,
            "vcs_discobj_yaw": vcs_discobj_yaw,
        }
        data["gt_bev_discrete_obj"] = gt_bev_discrete_obj
        data["annos_bev_discrete_obj"] = annos_bev_discrete_obj
        return data

    def __repr__(self):
        return "BevDiscreteTargetGenerator"


@OBJECT_REGISTRY.register
class PrepareDepthPose(object):
    """Build input data for 2.5D task.

    e.g stack mulit view datas on batch axis. convert tensor type and so on.

    Args:
        with_extra_img (bool): whther return extra img.
            set True when depth/pose training and False for validation.
        with_color_img (bool): whther return color img.
            Used for depth/pose training.
        input_sequence_length (int): img number input to backbone, 2 means
            two img input to backbone(order is (t-1,t)), 1 means single img
            input to backbone.
    """

    def __init__(
        self,
        with_extra_img: bool = False,
        with_color_img: bool = False,
        input_sequence_length: int = 1,
    ) -> None:

        self.with_extra_img = with_extra_img
        self.with_color_img = with_color_img
        assert input_sequence_length in (1, 2)
        self.input_sequence_length = input_sequence_length

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["view"] = "front"
        cat_data = [torch.stack(_) for _ in data["imgs"]]

        if self.with_extra_img:
            # durning train process, need three frames,
            # suppose the order in data['imgs'] is [t,t-1,t-2]
            # and t-1 frame is current frame(
            # it means we will cal depth loss for t-1 frame)
            data["extra_img"] = [cat_data[0], cat_data[2]]  # t,t-2
            if self.input_sequence_length == 1:
                data["img"] = [cat_data[1]]  # t-1
            else:
                data["img"] = [cat_data[2], cat_data[1]]  # t-2,t-1
        else:
            # durning val process,only need two frames,
            # suppose the order in data['imgs'] is [t-1,t-2]
            data["extra_img"] = [cat_data[1]]  # t-2
            if self.input_sequence_length == 1:
                data["img"] = [cat_data[0]]  # t-1
            else:
                data["img"] = [cat_data[1], cat_data[0]]  # t-2,t-1

        if self.with_color_img:
            color_cat_data = []
            for color_img_i in data["color_imgs"]:
                color_cat_data.append(torch.stack(color_img_i))
            data["color_imgs"] = color_cat_data

        data.pop("imgs")

        if "gt_depth" in data:
            data["gt_depth"] = torch.stack(data["gt_depth"])

        if "front_mask" in data:
            data["front_mask"] = data["front_mask"].float()

        if "obj_mask" in data:
            data["obj_mask"] = (
                (data["obj_mask"] <= 17) * (data["obj_mask"] >= 9)
            ).float()
        # class idx between 9 and 17 means dynamic class
        # (auto 33class parsing model). e.g. bus, car, person and so on.

        if "size" in data:
            data.pop("size")

        return data


@OBJECT_REGISTRY.register
class SelectDataByIdx(object):
    """Select the specified data from the input data by idx.

    Args:
        select_idxs (list(int)): the index list to select.
        input_key (str): data key in input dict to select.
        output_key (str): data key to store data selected.
            NOTE: output_key and input_key is same means in-place modification.

    """

    def __init__(
        self,
        select_idxs: Sequence[int],
        input_key: str = "imgs",
        output_key: str = "img",
    ):
        self.select_idxs = select_idxs
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, data: Mapping):
        assert self.input_key in data
        assert isinstance(data[self.input_key], Sequence)
        data[self.output_key] = [
            data[self.input_key][idx] for idx in self.select_idxs
        ]
        return data

    def __repr__(self):
        return "SelectDataByIdx"


@OBJECT_REGISTRY.register
class StackData(object):
    """Stack specific data in input dict.

    Args:
        data_keys (str/list(str)): a key list to stack.

    """

    def __init__(self, data_keys: Union[str, Sequence[str]]):
        self.data_keys = _as_list(data_keys)

    def __call__(self, data: Mapping):
        for key in self.data_keys:
            if key not in data:
                continue
            assert isinstance(data[key], Sequence)
            if isinstance(data[key][0], Sequence):
                data[key] = [torch.stack(_) for _ in data[key]]
            else:
                data[key] = torch.stack(data[key])
        return data

    def __repr__(self):
        return "StackData"


@OBJECT_REGISTRY.register
class PrepareDataBEV(object):
    """Build input data for BEV task.

    e.g stack mulit view datas on batch axis. convert tensor type and so on.

    Args:
        organize_data_type (str): specify which method to organize data.
            e.g.
            "front" means all view datas will store in key of "img" and
            pass one single backbone(front backbone).
            "front_side" means front view data will store in key of "img"
            and pass front backbone. Side view datas will store in key of
            "side_img" and pass side backbone.
            "round" means all view datas will store in "round_img" and
            round backbone(fisheye backbone).
            "front_side_round" means front view data will store in key of "img"
            and pass front backbone. Side view datas will store in key of
            "side_img" and pass side backbone, and Round View datas datas
            will store in key of "round_img" and pass round backbone.
        single_frame (bool): Whether it is single frame input, default to True.
    """

    def __init__(
        self,
        organize_data_type: str,
        single_frame: bool = True,
    ) -> None:
        assert organize_data_type in [
            "front",
            "front_side",
            "round",
            "front_side_round",
        ], f"do not support type of {organize_data_type}"
        self.organize_data_type = organize_data_type
        self.single_frame = single_frame

    def __call__(self, data: Mapping):
        assert "imgs" in data, 'input data must has "imgs"'
        data["view"] = self.organize_data_type

        if self.organize_data_type == "front_side":

            front_cat_data = [torch.stack(_[0:1]) for _ in data["imgs"]]
            side_cat_data = [torch.stack(_[1:]) for _ in data["imgs"]]
            data["img"] = [front_cat_data[0]]  # t
            data["side_img"] = [side_cat_data[0]]  # t
            if not self.single_frame:
                data["img"].insert(0, front_cat_data[1])  # t-1,t
                data["side_img"].insert(0, side_cat_data[1])  # t-1,t

            if "gt_seg" in data:
                data["front_gt_seg"] = torch.stack(data["gt_seg"][0:1]).long()
                data["side_gt_seg"] = torch.stack(data["gt_seg"][1:]).long()
                data.pop("gt_seg")

        elif self.organize_data_type == "front":
            cat_data = [torch.stack(_) for _ in data["imgs"]]
            data["img"] = [cat_data[0]]  # t
            if not self.single_frame:
                data["img"].insert(0, cat_data[1])  # t-1,t

            if "gt_seg" in data:
                data["gt_seg"] = torch.stack(data["gt_seg"]).long()
        elif self.organize_data_type == "round":
            cat_data = [torch.stack(_) for _ in data["imgs"]]
            data["round_img"] = [cat_data[0]]  # t-1,t
            if not self.single_frame:
                data["round_img"].insert(0, cat_data[1])  # t-1,t

            if "gt_seg" in data:
                data["gt_seg"] = torch.stack(data["gt_seg"]).long()
        elif self.organize_data_type == "front_side_round":
            front_cat_data = [
                torch.stack(_[0:1]) for _ in data["imgs"]
            ]  # 10v [1,5,4] front,side,round
            side_cat_data = [torch.stack(_[1:6]) for _ in data["imgs"]]
            round_cat_data = [torch.stack(_[6:]) for _ in data["imgs"]]
            data["img"] = [front_cat_data[0]]  # t
            data["side_img"] = [side_cat_data[0]]  # t
            data["round_img"] = [round_cat_data[0]]  # t
            if not self.single_frame:
                data["img"].insert(0, front_cat_data[1])  # t-1,t
                data["side_img"].insert(0, side_cat_data[1])  # t-1,t
                data["round_img"].insert(0, round_cat_data[1])  # t-1,t
            if "gt_seg" in data:
                data["front_gt_seg"] = torch.stack(data["gt_seg"][0:1]).long()
                data["side_gt_seg"] = torch.stack(data["gt_seg"][1:6]).long()
                data["round_gt_seg"] = torch.stack(data["gt_seg"][6:]).long()
                data.pop("gt_seg")
        data.pop("imgs")

        if "gt_bev_seg" in data:
            data["gt_bev_seg"] = data["gt_bev_seg"].long()

        if "occlusion" in data:
            data["occlusion"] = data["occlusion"].long()

        if "gt_bev_3d" in data:
            for k in data["gt_bev_3d"].keys():
                data["gt_bev_3d"][k] = torch.from_numpy(
                    data["gt_bev_3d"][k].transpose(2, 0, 1)
                )
            assert "annos_bev_3d" in data
            for k in data["annos_bev_3d"].keys():
                data["annos_bev_3d"][k] = torch.from_numpy(
                    data["annos_bev_3d"][k]
                )

        if "size" in data:
            data.pop("size")

        return data


@OBJECT_REGISTRY.register
class ConvertReal3dTo3DV(object):
    """Convert real3d annotations to gt_bev_3d.

    Args:
        repeat_times (int): The repeat times of the single image.
        category_id_dict (dict): The category mapping dict for label exchange.
    """

    def __init__(self, category_id_dict):
        super().__init__()
        self.category_id_dict = category_id_dict

    def __call__(self, data_dict: Mapping):
        anno_multi_view = data_dict.pop("annotations")
        data_dict.pop("num_classes")

        gt_bev_3d = []

        for anno_each_view in anno_multi_view:
            for anno in anno_each_view:
                if "in_camera" in anno:
                    anno.update(anno.pop("in_camera"))
                gt_object = {}
                gt_object["dimension"] = anno["dim"]
                gt_object["location"] = anno["location"]
                gt_object["score"] = 1
                gt_object["category_id"] = anno["category_id"]
                gt_object["label"] = self.category_id_dict[anno["category_id"]]
                gt_object["ignore"] = anno["ignore"]
                gt_object["rotation_y"] = anno["rotation_y"]
                gt_object["image_id"] = anno["image_id"]
                gt_bev_3d.append(gt_object)

        data_dict["gt_bev_3d"] = gt_bev_3d
        # collect pack_dir for eval result
        data_dict["pack_dir"] = data_dict["image_name"][0].split("__")[0]

        return data_dict

    def __repr__(self):
        return "ConvertReal3dTo3DV"


@OBJECT_REGISTRY.register
class Bev3dTargetGeneratorV2(Bev3dTargetGenerator):
    """Parsing real3d annotations and generate gound truth labels for bev_3d.

    Args:
        lidar2cam_path (str): path of lidar2camera matrix npy file.
        lidar2vcs_extrinsic (ndarray): A 3*3 matrix of the extrinsic.
        sub_dirs (Sequence[str]): sub directory name of each view.

    """

    def __init__(
        self,
        lidar2cam_path: str,
        lidar2vcs_extrinsic: Mapping,
        sub_dirs: Sequence[str],
        **kwargs,
    ):
        super(Bev3dTargetGeneratorV2, self).__init__(**kwargs)
        self.lidar2cam_path = lidar2cam_path
        self.lidar2vcs_extrinsic = lidar2vcs_extrinsic
        self.sub_dirs = sub_dirs

        self.m_perpixel = (
            abs(self.vcs_range[2] - self.vcs_range[0]) / self.bev_size[0],
            abs(self.vcs_range[3] - self.vcs_range[1]) / self.bev_size[1],
        )  # bev coord y, x

        # laod extrinsics (lidar2cam), return list
        self.lidar2cam_array = self._load_lidar2cam_extrinsic()
        self.RT_Cam_To_Lidar = [
            np.linalg.inv(_lidar2cam) for _lidar2cam in self.lidar2cam_array
        ]
        self.RT_LIDAR_To_VCS = lidar2vcs_extrinsic

    def _load_lidar2cam_extrinsic(self):
        """Load lidar2camera extrinsics for each view and retuan as list."""
        lidar2cam_array = []
        for sub_dir in self.sub_dirs:
            extrinsic_file = os.path.join(
                self.lidar2cam_path, "lidar_" + sub_dir + ".npy"
            )
            assert os.path.exists(extrinsic_file)
            extrinsic = np.load(extrinsic_file)
            lidar2cam_array.append(extrinsic)
        return lidar2cam_array

    def __call__(self, data):
        """Generate bev_3d labels.

        Args:
            data (dict): Type is ndarray
                The dict contains at leaset annotations

        Returns (dict): Type is ndarray

        """
        assert "gt_bev_3d" in data
        annotations = data.pop("gt_bev_3d")

        # build bev3d gt heatmaps
        bev3d_hm = np.zeros(
            (*self.bev_size, self.num_classes), dtype=np.float32
        )
        bev3d_dim = np.zeros((*self.bev_size, 3), dtype=np.float32)  # h, w, l
        bev3d_rot = np.zeros((*self.bev_size, 2), dtype=np.float32)  # cos sin
        bev3d_ct_offset = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # bev center offest
        bev3d_loc_z = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_weight_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_point_pos_mask = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_ignore_mask = np.zeros((self.bev_size), dtype=np.float32)

        # used for eval
        vcs_loc_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_dim_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_rot_z_ = np.zeros((self.max_objs), dtype=np.float32)
        vcs_cls_ = np.zeros((self.max_objs), dtype=np.float32) - 99
        vcs_ignore_ = np.zeros((self.max_objs), dtype=np.bool)
        vcs_visible_ = np.zeros((self.max_objs), dtype=np.float32)

        image_ids = []

        count_id = 0
        for ann_idx, anno in enumerate(annotations):  # noqa [B007]
            image_ids.append(anno["image_id"])
            if count_id > (self.max_objs):
                break
            cls_id = anno["label"]
            if cls_id <= -99 or (
                not self.enable_ignore and anno.get("ignore", False)
            ):
                continue
            anno["location"][1] -= (
                anno["dimension"][0] / 2
            )  # fix the loc's y axis in camera
            sub_dir = data["image_name"][
                data["image_id"].index(anno["image_id"])
            ].split("__")[1]
            if "_0820" in sub_dir:
                sub_dir = sub_dir.split("_0820")[0]
            view_idx = self.sub_dirs.index(sub_dir)
            vcs_loc = (
                self.RT_LIDAR_To_VCS
                @ self.RT_Cam_To_Lidar[view_idx]
                @ (anno["location"] + [1.0])
            )[:3]
            bev_ct = (
                ((self.vcs_range[3] - vcs_loc[1]) / self.m_perpixel[1]),
                ((self.vcs_range[2] - vcs_loc[0]) / self.m_perpixel[0]),
            )  # (u, v)
            bev_ct_int = (int(bev_ct[0]), int(bev_ct[1]))

            if (
                0 <= bev_ct_int[0] < self.bev_size[1]
                and 0 <= bev_ct_int[1] < self.bev_size[0]
            ):
                if anno.get("visibility", False):
                    vcs_visible_[count_id] = anno["visibility"]

                valid_vcs_loc = vcs_loc_[:count_id]
                distance = (
                    np.sum((vcs_loc[:2] - valid_vcs_loc[:, :2]) ** 2, axis=1)
                    ** 0.5
                )
                # Filter redundant bboxes with a distance less than 0.1
                # caused by projection deviation.
                if (distance < 0.1).any():
                    continue
                vcs_loc_[count_id] = vcs_loc
                vcs_cls_[count_id] = cls_id
                vcs_dim_[count_id] = anno["dimension"]
                # transfer the yaw to value [-pi,pi]
                cam_rot_z = anno["rotation_y"]
                cam_rot_z_vec = np.array(
                    [np.cos(cam_rot_z), 0, -np.sin(cam_rot_z)]
                )
                rot_cam2vcs = (
                    self.RT_LIDAR_To_VCS @ self.RT_Cam_To_Lidar[view_idx]
                )[:3, :3]
                vcs_rot_z_vec = rot_cam2vcs @ cam_rot_z_vec
                vcs_rot_z = np.arctan2(vcs_rot_z_vec[1], vcs_rot_z_vec[0])
                vcs_rot_z_[count_id] = vcs_rot_z
                if anno.get("ignore", False):
                    vcs_ignore_[count_id] = 1

                # draw bev3d heatmap
                insert_bev_hm = get_gaussian2D(
                    self.cls2kernel[cls_id], alpha=1
                )
                insert_bev_hm_wh = insert_bev_hm.shape[:2][::-1]
                if anno.get("ignore", False):
                    insert_ignore_mask = np.ones_like(insert_bev_hm)
                else:
                    insert_ignore_mask = np.zeros_like(insert_bev_hm)
                if self.cls_dimension is not None:
                    ori_dim = np.array(anno["dimension"])
                    avg_dim = self.cls_dimension[cls_id]
                    residual_dim = np.log(ori_dim / avg_dim)
                    ann_dim = list(residual_dim)
                else:
                    ann_dim = anno["dimension"]
                insert_bev_reg_map_list = [
                    get_reg_map(insert_bev_hm_wh, ann_dim),
                    get_reg_map(
                        # insert_bev_hm_wh, (np.cos(vcs_yaw), np.sin(vcs_yaw))
                        insert_bev_hm_wh,
                        (np.cos(vcs_rot_z), np.sin(vcs_rot_z)),
                    ),
                    get_reg_map(insert_bev_hm_wh, vcs_loc[-1]),
                    self.get_ctoff_map(insert_bev_hm_wh, bev_ct),
                ]
                bev_reg_map_list = [
                    bev3d_dim,
                    bev3d_rot,
                    bev3d_loc_z,
                    bev3d_ct_offset,
                ]
                draw_heatmap(bev3d_hm[:, :, cls_id], insert_bev_hm, bev_ct_int)
                draw_heatmap(
                    bev3d_weight_hm,
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )
                draw_heatmap(bev3d_ignore_mask, insert_ignore_mask, bev_ct_int)

                bev3d_point_pos_mask[bev_ct_int[1], bev_ct_int[0]] = 1
                count_id += 1

        gt_bev_3d = {
            "bev3d_hm": bev3d_hm,
            "bev3d_dim": bev3d_dim,
            "bev3d_rot": bev3d_rot,
            "bev3d_ct_offset": bev3d_ct_offset,
            "bev3d_loc_z": bev3d_loc_z[:, :, np.newaxis],
            "bev3d_weight_hm": bev3d_weight_hm[:, :, np.newaxis],
            "bev3d_point_pos_mask": bev3d_point_pos_mask[:, :, np.newaxis],
            "bev3d_ignore_mask": bev3d_ignore_mask[:, :, np.newaxis],
        }

        annos_bev_3d = {
            "vcs_loc_": vcs_loc_,
            "vcs_cls_": vcs_cls_,
            "vcs_rot_z_": vcs_rot_z_,
            "vcs_dim_": vcs_dim_,
            "vcs_ignore_": vcs_ignore_,
            "vcs_visible_": vcs_visible_,
        }

        data["gt_bev_3d"] = gt_bev_3d
        data["annos_bev_3d"] = annos_bev_3d
        data.pop("image_id")
        data.pop("image_name")

        return data

    def __repr__(self):
        return "Bev3dTargetGeneratorV2"


# used for fsd pack inference
@OBJECT_REGISTRY.register
class HomoAdaption(object):  # noqa: D205,D400
    """Modify homography matrix of each view and convert to tensor.
        only use for pack inference.
    Args:
        homo_scale_each_view (list): homography matrix scale factor.
    """

    def __init__(self, homo_scale_each_view: Sequence):
        self.homo_scale_each_view = homo_scale_each_view

    def __call__(self, data):
        if "homography" in data:
            # modify homography matrix
            assert len(data["homography"]) == len(self.homo_scale_each_view)
            for i, (h, scale) in enumerate(
                zip(data["homography"], self.homo_scale_each_view)
            ):
                scale_mat = np.eye(3, dtype="float32")
                scale_mat[0, 0] = scale
                scale_mat[1, 1] = scale
                data["homography"][i] = scale_mat @ h
            data["homography"] = np.stack(data["homography"], axis=0)
            data["homography"] = torch.from_numpy(data["homography"])
        if "homo_offset" in data:
            data["homo_offset"] = np.stack(data["homo_offset"], axis=0)
            data["homo_offset"] = torch.from_numpy(data["homo_offset"])
        return data
