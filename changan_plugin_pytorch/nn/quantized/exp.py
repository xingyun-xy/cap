import torch
from changan_plugin_pytorch.march import March, get_march
from ..qat.exp import Exp as qat_exp

from .lut import LookUpTable
from .multi_table_fit import MultiTableFit


class Exp(torch.nn.Module):
    _QAT_MODULE = qat_exp

    def __init__(self, scale, out_dtype):
        super(Exp, self).__init__()
        if get_march() == March.BERNOULLI2:
            self.lut = LookUpTable(func=torch.exp)
        else:
            self.lut = MultiTableFit(
                func=torch.exp,
                dense_xmin=-6.0,
                dense_xmax=0.0,
                sparse_xmin=0.0,
                sparse_xmax=6.0,
                left_line_xmin=-20.0,
                left_line_xmax=-6.0,
                right_line_xmin=6.0,
                right_line_xmax=20.0,
                out_type=out_dtype,
            )
        self.register_buffer("scale", scale)

    def forward(self, data):
        return self.lut(data, self.scale)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module

        Args: `mod` a qat module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        quantized_exp = cls(
            scale=mod.activation_post_process.scale,
            out_dtype=mod.activation_post_process.dtype,
        )
        quantized_exp.scale.copy_(mod.activation_post_process.scale)
        return quantized_exp
