# Copyright (c) Changan Auto. All rights reserved.

from typing import Mapping, Sequence, Union

import torch
import torch.nn.functional as F

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list

__all__ = ["ElevationPostprocess"]


def set_id_grid_batch(img: torch.Tensor):
    """Generate meshgrid.

    Args:
        img: shape (b, c, h, w).
    """
    b, _, h, w = img.shape
    i_range = (
        torch.arange(0, h, dtype=img.dtype)
        .reshape((1, h, 1))
        .expand((b, 1, h, w))
        .to(img.device)
    )
    j_range = (
        torch.arange(0, w, dtype=img.dtype)
        .reshape((1, 1, w))
        .expand((b, 1, h, w))
        .to(img.device)
    )
    ones = torch.ones((b, 1, h, w), dtype=img.dtype).to(img.device)
    pixel_coords = torch.cat([j_range, i_range, ones], dim=1)
    return pixel_coords


def calculate_depth_height_from_gamma(
    gamma: torch.Tensor,
    N: torch.Tensor,
    camH: torch.Tensor,
    K_inv: torch.Tensor,
):
    """Calculate depth, height from gamma map.

    Args:
        gamma: gamma map.
        N: ground norm.
        camH: camera height.
        K_inv: inverse camera intrinsics.
    """
    pixel_coords_nml = set_id_grid_batch(gamma)
    b, _, h, w = pixel_coords_nml.shape
    N_T = N.permute((0, 2, 1))  # b*3*1 -> b*1*3
    pixel_coords_nml = pixel_coords_nml.reshape((b, 3, h * w))
    tmp1 = torch.bmm(K_inv, pixel_coords_nml)
    tmp2 = torch.bmm(N_T, tmp1)
    tmp3 = gamma - tmp2.reshape((b, 1, h, w))
    tmp3 = tmp3 + 1e-3
    Depth = torch.div(camH, tmp3.reshape((b, h * w)))
    Depth = Depth.reshape((b, 1, h, w))
    Height = gamma * Depth
    return Depth, Height


@OBJECT_REGISTRY.register
class ElevationPostprocess(torch.nn.Module):
    """Postprocess elevationnet output.

    Using ground norm and camera height to convert the output
    gamma map to depth and height.

    Args:
        gamma_scale: gamma scale, the usually setting is 1000.
        gt_size: ground true gamma size, the usually setting is (512, 960).
        output_name: output name of elevationnet.
        postprocess_frame_idx: the frame idx to process.
    """

    def __init__(
        self,
        gamma_scale: float,
        gt_size: Sequence,
        output_name: str,
        postprocess_frame_idx: int,
    ):
        super(ElevationPostprocess, self).__init__()
        self.gamma_scale = gamma_scale
        self.gt_size = gt_size
        self.output_name = output_name
        self.postprocess_frame_idx = _as_list(postprocess_frame_idx)[0]

    def generate_depth_height(
        self,
        pred: Union[torch.Tensor, Sequence[torch.Tensor]],
        label: Mapping,
    ):
        gt_size = self.gt_size
        out = (
            F.interpolate(
                pred[
                    "%s_frame%s"
                    % (self.output_name, str(self.postprocess_frame_idx))
                ][0],
                size=gt_size,
                mode="bilinear",
            )
            / self.gamma_scale
        )

        # trans gamma -> depth, height
        intrinsics = label["intrinsics"]
        intrinsics_scaled = intrinsics.clone()
        downscale = label["color_imgs"][0].shape[2] / gt_size[0]
        intrinsics_scaled[:, 0:2] /= downscale
        intrinsics_inverse = torch.inverse(intrinsics_scaled)

        camera_height = label["camera_high"][self.postprocess_frame_idx]
        ground_norm = label["ground_norm"][self.postprocess_frame_idx]
        depth, height = calculate_depth_height_from_gamma(
            out, ground_norm, camera_height, intrinsics_inverse
        )
        return depth, height

    def forward(
        self, pred: Union[torch.Tensor, Sequence[torch.Tensor]], label: Mapping
    ):
        depth, height = self.generate_depth_height(pred, label)
        pred[
            "%s_frame%s" % ("pred_depth", str(self.postprocess_frame_idx))
        ] = depth
        pred[
            "%s_frame%s" % ("pred_height", str(self.postprocess_frame_idx))
        ] = height
        return pred
