import torch
from torch import Tensor
import torch.nn.functional as F
import warnings


class GridSample(torch.nn.Module):
    """
    Given an input and a flow-field grid, computes the output using
    input values and pixel locations from grid.

    Note that the grid required by this function is DIFFERENT from
    torch.nn.functional.grid_sample !!!

    Args:
        mode (str, optional): Interpolation mode to calculate output values.
            Only "bilinear" is supported now.
            Defaults to "bilinear".
        padding_mode (str, optional): Padding mode for outside grid values.
            Only "zeros" and "border" is supported now.
            Defaults to "border".
        align_corners ([type], optional): Since the grid format is
            different with torch.nn.functional.grid_sample, this param
            does not have any effect now.
            Defaults to None.
    """

    def __init__(
        self, mode="bilinear", padding_mode="zeros", align_corners=None
    ):
        super(GridSample, self).__init__()

        assert (
            mode == "bilinear"
        ), "grid_sample only support 'bilinear' mode now"
        assert padding_mode in (
            "zeros",
            "border",
        ), "grid_sample only support 'zeros' and 'border' padding_mode now"
        assert isinstance(
            align_corners, (bool, type(None))
        ), "param 'align_corners' must be bool or None"

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        warnings.warn(
            "GridSample module is deprecated,"
            + " please use torch.nn.functional.grid_sample",
            DeprecationWarning,
        )

    def forward(self, x, grid):
        # type: (Tensor, Tensor) -> Tensor
        """
        The forward pass of GridSample.

        Args:
            x (Tensor[N, C, H, W]): Input data.
            grid (Tensor[N, H_out, W_out, (dx, dy)]): Flow-field. This param
                is different with torch.nn.functional.grid_sample. In this
                function, the sample point of output point (x, y) is computed
                by (x + dx, y + dy).
        """
        # convert grid format from 'delta' to 'norm'
        n = grid.size(0)
        h = grid.size(1)
        w = grid.size(2)
        base_coord_y = (
            torch.arange(h, dtype=grid.dtype, device=grid.device)
            .unsqueeze(-1)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        base_coord_x = (
            torch.arange(w, dtype=grid.dtype, device=grid.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        absolute_grid_x = grid[:, :, :, 0] + base_coord_x
        absolute_grid_y = grid[:, :, :, 1] + base_coord_y
        norm_grid_x = absolute_grid_x * 2 / (x.size(3) - 1) - 1
        norm_grid_y = absolute_grid_y * 2 / (x.size(2) - 1) - 1
        norm_grid = torch.stack((norm_grid_x, norm_grid_y), dim=-1)

        r = F.grid_sample(x, norm_grid, self.mode, self.padding_mode, True)
        return r
