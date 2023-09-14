import torch
from changan_plugin_pytorch.nn import grid_sample as float_grid_sample
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input

from .functional import grid_sample


class GridSample(torch.nn.Module):
    """
    GridSample for quantized training, See nn.GridSample.
    """

    _FLOAT_MODULE = float_grid_sample.GridSample

    def __init__(
        self,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=None,
        qconfig=None,
    ):
        super(GridSample, self).__init__()

        assert (
            mode == "bilinear"
        ), "grid_sample only support 'bilinear' mode now"
        assert padding_mode in (
            "zeros",
            "border",
        ), "grid_sample only support 'zeros' and 'border' padding_mode now"

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        assert (
            self.qconfig.activation
        ), "activation_post_process must included in qconfig for qat.Resize2d"
        self.activation_post_process = self.qconfig.activation()
        self.activation_post_process.disable_observer()

    def forward(self, x, grid):
        # type: (QTensor, QTensor) -> QTensor
        """
        The forward pass of GridSample.

        Args:
            x (QTensor[N, C, H, W]): Input data.
            grid (QTensor[N, H_out, W_out, (dx, dy)]): Flow-field. This param
                is different with torch.nn.functional.grid_sample. In this
                function, the sample point of output point (x, y) is computed
                by (x + dx, y + dy).
        """
        assert_qtensor_input((x, grid))

        self.activation_post_process.set_qparams(x.q_scale())
        r = grid_sample(
            x.as_subclass(torch.Tensor),
            grid.as_subclass(torch.Tensor),
            self.mode,
            self.padding_mode,
            self.align_corners,
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
        )
        r = self.activation_post_process(r)
        return r

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qat_mod = cls(
            mode=mod.mode,
            padding_mode=mod.padding_mode,
            align_corners=mod.align_corners,
            qconfig=mod.qconfig,
        )
        return qat_mod
