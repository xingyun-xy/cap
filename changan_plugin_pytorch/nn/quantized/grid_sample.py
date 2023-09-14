import torch
from changan_plugin_pytorch.dtype import qint8
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor

from .functional import grid_sample


class GridSample(torch.nn.Module):
    """
    GridSample for bpu inference, See nn.GridSample.
    """

    _QAT_MODULE = qat.GridSample

    def __init__(
        self,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=None,
        out_dtype=qint8,
    ):
        super(GridSample, self).__init__()

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.out_dtype = out_dtype

    def forward(self, x, grid):
        # type: (QTensor, QTensor) -> QTensor
        """
        The forward pass of GridSample.

        Args:
            x (QTensor[N, C, H, W]): Input data.
            grid (QTensor[N, H_out, W_out, (dx, dy)]): Flow-field.
                This param is different with
                torch.nn.functional.grid_sample. In this function, the
                sample point of output point (x, y) is computed
                by (x + dx, y + dy).
        """
        r = grid_sample(
            x.int_repr(),
            grid.int_repr(),
            self.mode,
            self.padding_mode,
            self.align_corners,
            grid.q_scale(),
            grid.q_zero_point(),
            grid.dtype,
        )
        return QTensor(r, x.q_scale(), self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        assert mod.activation_post_process, (
            cls._QAT_MODULE.__name__ + " must have activation_post_process"
        )
        out_dtype = mod.activation_post_process.dtype

        quantized_mod = cls(
            mode=mod.mode,
            padding_mode=mod.padding_mode,
            align_corners=mod.align_corners,
            out_dtype=out_dtype,
        )
        return quantized_mod
