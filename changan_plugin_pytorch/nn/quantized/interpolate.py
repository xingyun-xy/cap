import torch
from changan_plugin_pytorch.dtype import qint8
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn.modules.utils import _pair

from .functional import interpolate


class Interpolate(torch.nn.Module):
    """Resize for bpu inference. Support bilinear and nearest
    interpolate method.

    Parameters
    ----------
    Same as float version.
    """

    _QAT_MODULE = qat.Interpolate

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=None,
        recompute_scale_factor=True,
        out_dtype=qint8,
    ):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.out_dtype = out_dtype

    def forward(self, data):
        out = interpolate(
            data.int_repr(),
            _pair(self.size) if self.size else None,
            _pair(self.scale_factor) if self.scale_factor else None,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
            data.q_scale(),
            data.q_zero_point(),
            data.dtype,
        )
        return QTensor(
            out, data.scale, self.out_dtype, data.q_per_channel_axis()
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        assert (
            mod.activation_post_process
        ), "qat.Interpolate must have activation_post_process"
        out_dtype = mod.activation_post_process.dtype

        quantized_mod = cls(
            size=mod.size,
            scale_factor=mod.scale_factor,
            mode=mod.mode,
            align_corners=mod.align_corners,
            recompute_scale_factor=mod.recompute_scale_factor,
            out_dtype=out_dtype,
        )
        return quantized_mod
