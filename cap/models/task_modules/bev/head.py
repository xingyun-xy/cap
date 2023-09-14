# Copyright (c) Changan Auto. All rights reserved.
import random
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import changan_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
)
from changan_plugin_pytorch.quantization.stubs import QuantStub as HQuantStub
from torch.quantization import QConfig, QuantStub

from cap.models.base_modules import ConvModule2d, SeparableConvModule2d
from cap.models.task_modules.real3d import Real3DHead
from cap.models.utils import _take_features
from cap.models.weight_init import bias_init_with_prob, normal_init
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list, is_list_of_type

grid_qconfig = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=qinfo("qint16").min,
        quant_max=qinfo("qint16").max,
        dtype="qint16",
        saturate=True,
    ),
    weight=None,
)

__all__ = ["BEVFusionModule", "BEV3DHead"]


def get_random_idx(prob: float, n: int):
    """
    Generate a random index in [0,1,...n-1] with a probability.

    Args:
        prob (float): probability.
        n (int): max index.
    Returns:
        a randoms index.
    """
    flag = np.random.binomial(1, prob, 1).item()
    idx = random.randint(0, n - 1) if flag else None
    return idx


class SpatialTransfomer(nn.Module):  # noqa: D205,D400
    """
    Layer which transform feature from one view
    to other view with homography matrix.
    NOTE:
    when block_warp_padding is None, this module will do grid_sample directly.
    and block_warp_padding padding is not None, workflow will as below:
    1) calculate homography offset with homography matrix.
    2) crop homography offset with block_warp_padding parameter.
    3) do grid_sample.
    4) pad the warp result with block_warp_padding parameter.

    Args:
        height (int): height of grid.
        width (dict): width of grid.
        grid_quant_scale (float): quant scale of grid.
        mode (str): mode for grid_sample.
        padding_mode (str): padding_mode for grid_sample.
        eps (float): a small value to avoid overflow.
        use_horizon_grid_sample (boolen): whether use grid_sample op
            in changan plugin.
        block_warp_padding: order is (left,right,up,bottom).

    """

    def __init__(
        self,
        height: int,
        width: int,
        grid_quant_scale: Optional[float] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        eps: float = 1e-7,
        use_horizon_grid_sample: bool = True,
        block_warp_padding: Sequence[int] = None,
    ):
        super(SpatialTransfomer, self).__init__()

        self.height = height
        self.width = width
        self.mode = mode
        self.padding_mode = padding_mode
        self.eps = eps

        meshgrid = np.meshgrid(range(width), range(height), indexing="xy")
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)  # (2,h,w)

        ones = np.ones((1, height, width), dtype="float32")
        pix_coords = np.concatenate([id_coords, ones], axis=0).reshape(
            (1, 3, -1)
        )

        self.pix_coords = nn.Parameter(
            torch.from_numpy(pix_coords), requires_grad=False
        )

        self.block_warp_padding = block_warp_padding
        if block_warp_padding:
            assert len(block_warp_padding) == 4
            assert all([_ >= 0 for _ in block_warp_padding])
            assert (
                block_warp_padding[0] < width and block_warp_padding[1] < width
            )
            assert (
                block_warp_padding[2] < height
                and block_warp_padding[3] < height
            )
            self.zero_pad = nn.ZeroPad2d(padding=block_warp_padding)

        self.grid_sample = None
        self.grid_quant_stub = None
        if use_horizon_grid_sample:
            self.grid_sample = hnn.GridSample(
                mode=mode, padding_mode=padding_mode
            )
            self.grid_quant_stub = HQuantStub(scale=grid_quant_scale)

    def set_qconfig(self):
        self.grid_quant_stub.qconfig = grid_qconfig

    def forward(
        self,
        feature: Union[torch.Tensor, QTensor],
        transformation: torch.Tensor,
    ):
        """
        Forward method.

        Args:
            feature : a tensor,shape is (n,c,h,w)
            transformation : a tensor,shape is (n,3,3)
        """
        _, _, h, w = feature.shape

        cam_points = torch.matmul(transformation, self.pix_coords)

        # convert to float32 in case of float16
        # without below will cause bug in amp mode
        cam_points = cam_points.to(self.pix_coords.dtype)
        # donot convert to fp32 in qat training
        if type(feature) == torch.Tensor:
            feature = feature.to(self.pix_coords.dtype)

        new_pix_coords = cam_points[:, :2, :] / (
            cam_points[:, 2, :].unsqueeze(1) + self.eps
        )
        if self.grid_sample is not None:
            valid_points_x = (new_pix_coords[:, 0] - w / 2).abs() < (w / 2)
            valid_points_y = (new_pix_coords[:, 1] - h / 2).abs() < (h / 2)
            valid_points = (valid_points_x * valid_points_y).view(
                -1, 1, self.height, self.width
            )

            new_pix_coords = new_pix_coords - self.pix_coords[:, :2, :]
        new_pix_coords = new_pix_coords.view(-1, 2, self.height, self.width)
        new_pix_coords = new_pix_coords.permute(0, 2, 3, 1)

        if self.grid_sample is not None:
            if self.block_warp_padding:
                pad_l, pad_r, pad_u, pad_b = self.block_warp_padding
                new_pix_coords = new_pix_coords[
                    :,
                    pad_u : self.height - pad_b,
                    pad_l : self.width - pad_r,
                    :,
                ]
                new_pix_coords[:, :, :, 0] += pad_l
                new_pix_coords[:, :, :, 1] += pad_u
                new_pix_coords = self.grid_quant_stub(new_pix_coords)
                homography_feat = self.grid_sample(feature, new_pix_coords)
                homography_feat = self.zero_pad(homography_feat)
            else:
                new_pix_coords = self.grid_quant_stub(new_pix_coords)
                homography_feat = self.grid_sample(feature, new_pix_coords)

        else:
            new_pix_coords[..., 0] /= w - 1
            new_pix_coords[..., 1] /= h - 1
            new_pix_coords = (new_pix_coords - 0.5) * 2
            valid_points = (
                new_pix_coords.abs().max(dim=-1)[0].unsqueeze(1) <= 1
            )
            homography_feat = F.grid_sample(
                feature,
                new_pix_coords,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=False,
            )
        return homography_feat, valid_points.float()


class SpatialTransfomerFixedOffset(nn.Module):  # noqa: D205,D400
    """
    Layer which transform feature from one view
    to other view with fixed homography offset.

    Args:
        height (int): height of grid.
        width (dict): width of grid.
        grid_quant_scale (float): quant scale of grid.
        homography (torch.tensor) homography matrix.
        homo_offset (torch.tensor) homo_offset matrix.
        mode (str): mode for grid_sample.
        padding_mode (str): padding_mode for grid_sample.

    """

    def __init__(
        self,
        height: int,
        width: int,
        homography: Optional[torch.Tensor] = None,
        homo_offset: Optional[torch.Tensor] = None,
        grid_quant_scale: Optional[float] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        eps: float = 1e-7,
    ):
        super(SpatialTransfomerFixedOffset, self).__init__()

        self.height = height
        self.width = width
        self.mode = mode
        self.padding_mode = padding_mode
        self.eps = eps

        if homo_offset is not None:
            self.offset = nn.Parameter(homo_offset, requires_grad=False)
        elif homography is not None:
            self.offset = nn.Parameter(
                self.generate_homo_offset(homography), requires_grad=False
            )
        else:
            raise ValueError(
                "only one of (homo_offset,homography) should be None"
            )
        self.grid_sample = hnn.GridSample(mode=mode, padding_mode=padding_mode)
        self.grid_quant_stub = HQuantStub(scale=grid_quant_scale)

    def generate_homo_offset(self, homography):
        # NOTE: Calculating homo_offset on cpu here but gpu durning training.
        # The calculation results of the two methods are slightly different,
        # but basically do not affect the final result.
        y, x = torch.meshgrid(
            torch.arange(self.height), torch.arange(self.width)
        )
        id_coords = torch.stack([x, y], dim=0).float()
        ones = torch.ones((1, self.height, self.width)).float()
        pix_coords = torch.cat([id_coords, ones], dim=0).reshape((1, 3, -1))

        cam_points = torch.matmul(homography, pix_coords)

        new_pix_coords = cam_points[:, :2, :] / (
            cam_points[:, 2, :].unsqueeze(1) + self.eps
        )
        homo_offset = new_pix_coords - pix_coords[:, :2, :]
        homo_offset = homo_offset.view(-1, 2, self.height, self.width)
        homo_offset = homo_offset.permute(0, 2, 3, 1)
        return homo_offset

    def forward(self, feature: Union[torch.Tensor, QTensor]):
        """
        Foward method.

        Args:
            feature : a tensor,shape is (n,c,h,w)
        """
        homography_feat = self.grid_sample(
            feature, self.grid_quant_stub(self.offset)
        )
        return homography_feat

    def set_qconfig(self):
        self.grid_quant_stub.qconfig = grid_qconfig


class SpatialTransfomerWithOffset(nn.Module):  # noqa: D205,D400
    """
    Layer which transform feature from one view
    to other view with homography offset.
    NOTE:
    when block_warp_padding is None, this module will do grid_sample directly.
    and when block_warp_padding is not None, workflow will as below:
    1) crop homography offset with block_warp_padding parameter.
    2) do grid_sample.
    3) pad the warp result with block_warp_padding parameter.

    Args:
        height (int): height of grid.
        width (dict): width of grid.
        grid_quant_scale (float): quant scale of grid.
        mode (str): mode for grid_sample.
        padding_mode (str): padding_mode for grid_sample.
        block_warp_padding: order is (left,right,up,bottom).
        compile_model: compile model or not. only for int_infer step.
    """

    def __init__(
        self,
        height: int,
        width: int,
        grid_quant_scale: Optional[float] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        block_warp_padding: Sequence[int] = None,
        compile_model: bool = False,
    ):
        super(SpatialTransfomerWithOffset, self).__init__()
        self.height = height
        self.width = width
        self.grid_sample = hnn.GridSample(mode=mode, padding_mode=padding_mode)
        self.grid_quant_stub = HQuantStub(scale=grid_quant_scale)
        self.compile_model = compile_model
        self.block_warp_padding = block_warp_padding
        if block_warp_padding:
            assert len(block_warp_padding) == 4
            assert all([_ >= 0 for _ in block_warp_padding])
            assert (
                block_warp_padding[0] < width and block_warp_padding[1] < width
            )
            assert (
                block_warp_padding[2] < height
                and block_warp_padding[3] < height
            )
            self.zero_pad = nn.ZeroPad2d(padding=block_warp_padding)

    def forward(
        self, feature: Union[torch.Tensor, QTensor], offset: torch.Tensor
    ):
        """
        Foward method.

        Args:
            feature : a tensor,shape is (n,c,h,w)
            offset : a tensor,shape is (n,h',w',2)
        """
        if not self.compile_model and self.block_warp_padding:
            pad_l, pad_r, pad_u, pad_b = self.block_warp_padding
            offset = offset[
                :,
                pad_u : self.height - pad_b,
                pad_l : self.width - pad_r,
                :,
            ]
            offset[:, :, :, 0] += pad_l
            offset[:, :, :, 1] += pad_u
        homography_feat = self.grid_sample(
            feature, self.grid_quant_stub(offset)
        )
        if self.block_warp_padding:
            homography_feat = self.zero_pad(homography_feat)

        return homography_feat

    def set_qconfig(self):
        self.grid_quant_stub.qconfig = grid_qconfig


@OBJECT_REGISTRY.register
class RandomRotation(nn.Module):
    """
    Random rotate data durning BEV training.

    Args:
        height (int): height of grid.
        grid_quant_scale (float): quanti scale of grid.
        width (dict): width of grid.
        angles (list): rotation angle
        mode (str): mode for grid_sample.
        padding_mode (str): padding_mode for grid_sample.
        use_horizon_grid_sample (boolen): whether use grid_sample op
            in changan plugin.

    """

    def __init__(
        self,
        height: int,
        width: int,
        grid_quant_scale: float,
        angles: Sequence = (0),
        mode: str = "bilinear",
        padding_mode: str = "border",
        eps: float = 1e-7,
        use_horizon_grid_sample: bool = True,
    ):
        super(RandomRotation, self).__init__()
        self.angle_num = len(angles)

        rot_mats = []
        trans_mat1 = (
            np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
            .reshape(3, 3)
            .astype("float32")
        )
        trans_mat2 = (
            np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]])
            .reshape(3, 3)
            .astype("float32")
        )
        for angle in angles:
            assert angle in [
                0,
                90,
                180,
                270,
            ], "rotation angle must in [0, 90, 180, 270]"
            r = (
                np.array(
                    [
                        [
                            np.cos(np.deg2rad(angle)),
                            -np.sin(np.deg2rad(angle)),
                            0,
                        ],
                        [
                            np.sin(np.deg2rad(angle)),
                            np.cos(np.deg2rad(angle)),
                            0,
                        ],
                        [0, 0, 1],
                    ]
                )
                .reshape(3, 3)
                .astype("float32")
            )
            rot_mats.append(trans_mat2 @ r @ trans_mat1)

        rot_mats = np.stack(rot_mats)
        self.rotation_mat = nn.Parameter(
            torch.from_numpy(rot_mats), requires_grad=False
        )

        self.st = SpatialTransfomer(
            height,
            width,
            grid_quant_scale=grid_quant_scale,
            mode=mode,
            padding_mode=padding_mode,
            eps=eps,
            use_horizon_grid_sample=use_horizon_grid_sample,
        )

    def get_rot_mat(self, nums):
        idx = np.random.randint(0, self.angle_num, size=nums)
        return self.rotation_mat[idx]

    def forward(self, datas: Sequence):
        """
        Forward method.

        Args:
            datas (list[tensor]): a list tensor to rotation.
        Returns:
            result: (list[tensor]): a list tensor after rotation.
            rot_mat: (tensor): a rotation matrix.

        """
        bs = datas[0].shape[0]
        rot_mat = self.get_rot_mat(bs)  # (b,3,3)
        result = []
        for data in datas:
            result.append(self.st(data, rot_mat)[0])
        return result, rot_mat

    def set_qconfig(self):
        self.st.set_qconfig()


@OBJECT_REGISTRY.register
class BEVFusionModule(nn.Module):
    r"""
    Input multi-views data and output fused BEV data.

    In general, there are the following situations:
    1.Training stage(float or qat): homography matrix as input to the model.
    Please configure as follows::

        use_homo_offset=False
        homographys=None
        homo_offset=None
        compile_model=False

    2.Training stage(float or qat): homography offset as input to the model.
    Please configure as follows::

        use_homo_offset=True
        homographys=None
        homo_offset=None
        compile_model=False

    3.Int_infer stage: homography offset as model`s parameter and we provide homography matrix.
    Please configure as follows::

        use_homo_offset=False
        homographys='provide homography matrix'
        homo_offset=None
        compile_model=True

    4.Int_infer stage: homography offset as model`s parameter and we provide homography offset.
    Please configure as follows::

        use_homo_offset=False
        homographys=None
        homo_offset='provide homography offset'
        compile_model=True

    5.Int_infer stage: homography offset will be used as input to the model.
    Please configure as follows::

        use_homo_offset=True
        homographys=None
        homo_offset=None
        compile_model=True

    Args:
        ipm_output_size (Tuple): output size of ipm, (height, width).
        views (int): view numbers.
        grid_quant_scale (float): quanti scale of grid for grid_sample.
            NOTE: this value is very important for qat training, must set
            properly.
        use_homo_offset (bool) : whether to use homo_offset
            If True, homography offset will as input to the model.
        homographys (torch.tensor): homography matrix of each view.usually be used in compiling process.
            NOTE: Setting homographys means we will use it to calculate homo_offset and save homo_offset
            as model`s parameter. So do not setting homographys in training stage.
        homo_offset (torch.tensor): homo_offset matrix of each view.usually be used in compiling process.
            NOTE: Setting homo_offset means we will save homo_offset as model`s parameter.
            So do not setting homographys in training stage.
        compile_model (bool): Whether compiling model. Compiling model means we only process single batch.
        random_rotation_cfg (bool): config of random rotaton module which
            will apply random rotation for bev input.
        bev_fusion_input_name (str): input key name of bev fusion.
        bev_fusion_out_name (str): out key name of bev fusion.

    """

    def __init__(
        self,
        ipm_output_size: Tuple,
        views: Union[int, Sequence],
        grid_quant_scale: float,
        use_homo_offset: Optional[bool] = False,
        homographys: Optional[torch.Tensor] = None,
        homo_offset: Optional[torch.Tensor] = None,
        compile_model: bool = False,
        random_rotation_cfg: Optional[Mapping] = None,
        drop_view_prob: float = 0.0,
        bev_fusion_input_name: str = "bev_fusion_input",
        bev_fusion_out_name: str = "bev_fusion_out",
        block_warp_padding=None,
        **kwargs,
    ):
        super(BEVFusionModule, self).__init__(**kwargs)
        self.views = _as_list(views)
        self.bev_fusion_input_name = bev_fusion_input_name
        self.bev_fusion_out_name = bev_fusion_out_name
        self.homographys = homographys
        self.homo_offset = homo_offset
        self.use_homo_offset = use_homo_offset
        self.spatial_transformers = nn.ModuleList()
        self.quant = nn.ModuleList()
        self.views_num = sum(_as_list(self.views))
        self.compile_model = compile_model
        self.ipm_output_size = ipm_output_size
        assert (
            is_list_of_type(ipm_output_size, int) and len(ipm_output_size) == 2
        )

        if compile_model:
            assert (
                homographys is not None
                or homo_offset is not None
                or use_homo_offset
            )
        else:
            assert (
                homographys is None
            ), "do not setting homographys in training stage"
            assert (
                homo_offset is None
            ), "do not setting homo_offset in training stage"

        if homographys is not None:
            expected_shape = (self.views_num, 3, 3)
            assert (
                homographys.shape == expected_shape
            ), f"shape of homography  provided is not valid.\
            expected: {expected_shape}, now: {homographys.shape}"
        if homo_offset is not None:
            expected_shape = (self.views_num, 512, 512, 2)
            assert (
                homo_offset.shape == expected_shape
            ), f"shape of homo_offset  provided is not valid.\
            expected: {expected_shape}, now: {homographys.shape}"
        for i in range(self.views_num):
            if homographys is None and homo_offset is None:
                if use_homo_offset:
                    st_i = SpatialTransfomerWithOffset(
                        height=ipm_output_size[0],
                        width=ipm_output_size[1],
                        grid_quant_scale=grid_quant_scale,
                        mode="bilinear",
                        padding_mode="zeros",
                        block_warp_padding=block_warp_padding[i]
                        if block_warp_padding is not None
                        else None,
                        compile_model=compile_model,
                    )
                else:
                    st_i = SpatialTransfomer(
                        height=ipm_output_size[0],
                        width=ipm_output_size[1],
                        grid_quant_scale=grid_quant_scale,
                        mode="bilinear",
                        padding_mode="zeros",
                        use_horizon_grid_sample=True,
                        block_warp_padding=block_warp_padding[i]
                        if block_warp_padding is not None
                        else None,
                    )
            else:
                st_i = SpatialTransfomerFixedOffset(
                    height=ipm_output_size[0],
                    grid_quant_scale=grid_quant_scale,
                    width=ipm_output_size[1],
                    homography=None
                    if homographys is None
                    else homographys[i : i + 1],
                    homo_offset=None
                    if homo_offset is None
                    else homo_offset[i : i + 1],
                    mode="bilinear",
                    padding_mode="zeros",
                )
            self.spatial_transformers.append(st_i)
            self.quant.append(QuantStub())

        self.drop_view_prob = drop_view_prob

        self.random_rotation = random_rotation_cfg

        self.adds = nn.ModuleList()
        for _i in range(self.views_num - 1):
            self.adds.append(nn.quantized.FloatFunctional())
        # we need cap multi batch data durning training
        self.cap = nn.quantized.FloatFunctional()

    def dropout_view(self, views_input, drop_view_idx=None):
        if drop_view_idx is not None:
            views_input = list(views_input)
            drop_data = views_input[drop_view_idx]
            if type(drop_data) == torch.Tensor:
                drop_data *= 0
            elif type(drop_data) == QTensor:
                drop_data = QTensor(
                    drop_data.data * 0,
                    drop_data.scale.clone(),
                    drop_data.dtype,
                )
            else:
                raise TypeError("donot support the type to dropout")
            views_input[drop_view_idx] = drop_data
        return views_input

    def parse_bev_data(
        self,
        data: Union[torch.Tensor, QTensor],
        homography_mat: Optional[Sequence[torch.Tensor]] = None,
        homo_offset: Optional[Sequence[torch.Tensor]] = None,
        drop_view_idxs: Sequence[Optional[int]] = (None),
    ):
        """
        Parse bev data.

        Args:

            data (tensor): shape is [(n*views1,c,h,w),(n*views1,c,h,w)]
            homography_mat (tensor): homograpy matrix shape is (n,views,3,3)
            homo_offset (tensor): homo_offset  shape is (n,views,512,512,2)
            drop_view_idxs (list): dropout view idx of each sample.
                None means do not dropout.

        Returns:
            bev_input (tensor): shape is (n,c,h,w)

        """
        if self.use_homo_offset:
            homo_offset = torch.split(homo_offset, self.views_num, dim=0)
        bev_batch = []
        data_batch_list = [
            torch.split(cur_data, cur_view, dim=0)
            for cur_data, cur_view in zip(data, self.views)
        ]

        # data_batch_list: [[(views1,c,h,w),(views1,c,h,w),...],
        #                   [(views2,c,h,w),(views2,c,h,w),...]..]
        for batch_idx, cur_data_batch_list in enumerate(zip(*data_batch_list)):
            # cur_data_batch_list:[(views1,c,h,w),(views2,c,h,w)]
            # NOTE: torch.split return list in float step but tuple in qat step
            data_batch_split = []
            for cur_data_batch in cur_data_batch_list:
                data_batch_split = data_batch_split + list(
                    torch.split(cur_data_batch, 1, dim=0)
                )

            # data_batch_split: [(1,c,h,w),(1,c,h,w)...]
            if self.use_homo_offset:
                cur_bev = self.get_bev_input(
                    data_batch_split,
                    drop_view_idxs[batch_idx],
                    homography_offset=homo_offset[batch_idx],
                )
            else:
                cur_bev = self.get_bev_input(
                    data_batch_split,
                    drop_view_idxs[batch_idx],
                    homography_mat=homography_mat[batch_idx],
                )
            bev_batch.append(cur_bev)
        if len(bev_batch) > 1:
            return self.cap.cap(bev_batch, dim=0)  # (n,c,h,w)
        else:
            return bev_batch[0]

    def get_bev_input(
        self,
        multi_view_data,
        drop_view_idx=None,
        homography_mat=None,
        homography_offset=None,
    ):
        """
        Get bev input with stage1 output feature and homography_mat or homography_offset.

        Args:
            multi_view_data (list of tensor): each tensor`s shape
                in list is (views,c,h,w).
            homography_mat (tensor): homograpy matrix whose shape
                is (views,3,3).
            homography_offset (tensor/list tensor): homograpy offset(shape is (views,h,w,2)).  # noqa
            drop_view_idxs (int): dropout view idx,
                None means do not dropout.

        Returns:
            bev_input (tensor): shape is (1,c,h,w)

        """
        bevs = []
        for view_idx, each_view in enumerate(multi_view_data):
            if self.homographys is None and self.homo_offset is None:
                # homo matrix or homo offset as input
                if self.use_homo_offset:
                    # homo offset as input
                    bevs.append(
                        self.spatial_transformers[view_idx](
                            self.quant[view_idx](each_view),
                            _as_list(
                                homography_offset[view_idx : view_idx + 1]
                            )[0],
                        )
                    )
                else:
                    # homo matrix as input
                    bevs.append(
                        self.spatial_transformers[view_idx](
                            self.quant[view_idx](each_view),
                            homography_mat[view_idx : view_idx + 1],
                        )[0]
                    )
            else:
                # homo offset has been saved in self.spatial_transformers
                bevs.append(
                    self.spatial_transformers[view_idx](
                        self.quant[view_idx](each_view)
                    )
                )

        bevs = self.dropout_view(bevs, drop_view_idx=drop_view_idx)
        cur_bev = bevs[0]
        for view_idx, one_view in enumerate(bevs[1:]):
            cur_bev = self.adds[view_idx].add(cur_bev, one_view)
        return cur_bev

    def forward(self, data: Mapping):
        res = OrderedDict()
        if not self.compile_model:
            # we need to process multi-batch data durning training stage.
            stage1_out = _as_list(data[self.bev_fusion_input_name])
            bev_batchsize = stage1_out[0].shape[0] // self.views[0]
            drop_view_idxs = [None] * bev_batchsize
            if self.drop_view_prob > 0 and self.training:
                drop_view_idxs = [
                    get_random_idx(self.drop_view_prob, n=self.views_num)
                    for _ in range(bev_batchsize)
                ]
            if self.use_homo_offset:
                # homography offset as input to model.
                bev_input = self.parse_bev_data(
                    stage1_out,
                    homo_offset=data["homo_offset"],
                    drop_view_idxs=drop_view_idxs,
                )
            else:
                # homography matrix as input to model.
                bev_input = self.parse_bev_data(
                    stage1_out,
                    homography_mat=data["homography"],
                    drop_view_idxs=drop_view_idxs,
                )
        else:
            # druning compile model stage, only process single batch.
            if self.use_homo_offset:
                # homography offset as input to model.
                bev_input = self.get_bev_input(
                    data[self.bev_fusion_input_name],
                    homography_offset=data["homo_offset"],
                )
            else:
                # homography offset as model` parameters.
                bev_input = self.get_bev_input(
                    data[self.bev_fusion_input_name]
                )

        rot_mat = None
        if self.random_rotation is not None and self.training:
            bev_input, rot_mat = self.random_rotation([bev_input])
            bev_input = bev_input[0]
        res["bev_rot_mat"] = rot_mat

        res[self.bev_fusion_out_name] = bev_input
        return res

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.random_rotation is not None:
            self.random_rotation.set_qconfig()
        for module in list(self.spatial_transformers):
            if module is None:
                continue
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class BEV3DHead(Real3DHead):
    """Bev3DHead module.

    Args:
        in_channels (dict): A list of to indicates the input channels of the
            block.
        head_channels (dict): A dictionary contains output heads and
            corresponding channels.
        feature_name: (str): Name of features from backbone(or neck)
            in input dict.
        last_conv_kernel_size (int): Kernel size of last conv, default is 1.
        prior_heatmap_init (bool): Whether initialize conv, bias of heatmap
            module according to given values, default is True.
        forward_frame_idx (int): when input multi frame features,
            select which frame idx to forward, default is 0.
            e.g., for bev3d training, backbone will output features of
            frame t and t+1, and bev3d head only output result for t,
            so we should set forward_frame_idx to 0.
    """

    def __init__(
        self,
        in_channels: List[int],
        head_channels: Dict,
        feature_name: str,
        last_conv_kernel_size: int = 1,
        prior_heatmap_init: bool = True,
        forward_frame_idx: int = 0,
        **kwargs,
    ):
        super(BEV3DHead, self).__init__(
            in_channels=in_channels,
            head_channels=head_channels,
            **kwargs,
        )

        if last_conv_kernel_size > 1:
            for name in self.out_block_names:
                if self.sep_conv:
                    block = []
                    for _i in range(self.stack):
                        sep_block = SeparableConvModule2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            pw_norm_layer=nn.BatchNorm2d(in_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                        block.append(sep_block)
                    block.append(
                        nn.Conv2d(
                            in_channels,
                            head_channels[name],
                            last_conv_kernel_size,
                            1,
                            (last_conv_kernel_size - 1) // 2,
                            groups=1,
                            bias=False,
                        )
                    )
                    block = nn.Sequential(*block)
                else:
                    block = nn.Sequential(
                        ConvModule2d(
                            in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                            norm_layer=nn.BatchNorm2d(in_channels),
                            act_layer=nn.ReLU(inplace=True),
                        ),
                        nn.Conv2d(
                            in_channels,
                            head_channels[name],
                            last_conv_kernel_size,
                            1,
                            (last_conv_kernel_size - 1) // 2,
                            groups=1,
                            bias="hm" in name,
                        ),
                    )

                if "hm" in name and prior_heatmap_init:
                    bias = bias_init_with_prob(0.01)
                    normal_init(block[-1], std=0.01, bias=bias)
                setattr(self, "out_block_{}".format(name), block)

        self.feature_name = feature_name
        self.forward_frame_idx = forward_frame_idx

    def forward(
        self, data: Union[Mapping, Sequence]
    ) -> Mapping[str, torch.Tensor]:
        """Forward head layers."""
        input_features = (
            data[self.feature_name][self.forward_frame_idx]
            if isinstance(data, Mapping)
            else data
        )

        feat = _take_features(
            input_features, self.in_strides, self.out_strides
        )[0]
        feat = self.head_block(feat)

        out = OrderedDict()
        for name in self.out_block_names:
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            out[name] = self.dequant(block(feat))
        return out
