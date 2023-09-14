import torch
import torch.nn.functional as F
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import qat

from .lut import LookUpTable
from .multi_table_fit import MultiTableFit
from .segment_lut import SegmentLUT


class GELU(torch.nn.Module):
    _QAT_MODULE = qat.GELU

    def __init__(self, lut):
        super(GELU, self).__init__()
        self.lut = lut
        self.register_buffer("scale", torch.tensor([1], dtype=torch.float32))

    def forward(self, data):
        return (
            self.lut(data)
            if get_march() == March.BAYES
            else self.lut(data, self.scale)
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args:
            mod: a qat module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )

        march = get_march()

        if march == March.BAYES:
            lut = SegmentLUT.from_float(mod.lut)
        elif march == March.BERNOULLI2:
            lut = LookUpTable(
                func=F.gelu, out_type=mod.activation_post_process.dtype
            )
        else:
            lut = MultiTableFit(
                func=F.gelu,
                dense_xmin=-5.0,
                dense_xmax=0.0,
                sparse_xmin=0.0,
                sparse_xmax=5.0,
                left_line_xmin=-40.0,
                left_line_xmax=-5.0,
                right_line_xmin=5.0,
                right_line_xmax=40.0,
                out_type=mod.activation_post_process.dtype,
            )
        quantized_gelu = cls(lut)
        if mod.activation_post_process is not None:
            quantized_gelu.scale.copy_(mod.activation_post_process.scale)
        return quantized_gelu
