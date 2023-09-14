import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import qat

from .lut import LookUpTable
from .multi_table_fit import MultiTableFit


class Sigmoid(torch.nn.Module):
    _QAT_MODULE = qat.Sigmoid

    def __init__(self, lut):
        super(Sigmoid, self).__init__()
        self.lut = lut
        self.register_buffer("scale", torch.tensor([1], dtype=torch.float32))

    def forward(self, data):
        return self.lut(data, self.scale)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )

        activation_post_process = mod.activation_post_process
        if get_march() in (March.BERNOULLI, March.BERNOULLI2):
            lut = LookUpTable(func=torch.sigmoid)
        else:
            lut = MultiTableFit(
                func=torch.sigmoid,
                dense_xmin=-3.0,
                dense_xmax=3.0,
                sparse_xmin=-10.0,
                sparse_xmax=10.0,
                left_line_xmin=-20.0,
                left_line_xmax=-10.0,
                right_line_xmin=10.0,
                right_line_xmax=20.0,
                out_type=activation_post_process.dtype,
            )
        quantized_sigmoid = cls(lut)
        quantized_sigmoid.scale.copy_(activation_post_process.scale)
        return quantized_sigmoid
