from typing import Union

from torch import Tensor, nn

from .functional import point_pillars_scatter


class PointPillarsScatter(nn.Module):
    def __init__(self, output_shape=None):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image.
        This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.

        Args:
            output_shape: (List[int], optional).
                RExpected output shape. Defaults to None.
        """

        super(PointPillarsScatter, self).__init__()
        self.output_shape = (
            list(output_shape) if output_shape is not None else None
        )

    def forward(
        self,
        voxel_features: Tensor,
        coords: Tensor,
        output_shape: Union[Tensor, list, tuple] = None,
    ) -> Tensor:
        """
        Args:
            voxel_features (Tensor):
                [M, ...], dimention after M will be flattened.
            coords (Tensor):
                [M, (n, ..., y, x)], only indices on N, H and W are used.
            output_shape (Tensor, optional):
                Expected output shape. Defaults to None.

        Returns:
            Tensor: The NCHW pseudo image.
        """
        if output_shape is None:
            assert self.output_shape is not None, (
                "Please specify output shape"
                + " either during module init or forward."
            )
            output_shape = self.output_shape

        return point_pillars_scatter(voxel_features, coords, output_shape)
