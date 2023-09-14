import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import qat

from .lut import LookUpTable
from .multi_table_fit import MultiTableFit


class Tanh(torch.nn.Module):
    r"""
    x3/j3 quantized tanh accuracy is low in (-0.1, 0.1) interval
    j5 quantized tanh accuracy is low in (-0.01, 0.01) interval with
    recommeded usage: setting int16 input and output.
    """
    _QAT_MODULE = qat.Tanh

    def __init__(self, lut):
        super(Tanh, self).__init__()
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
            lut = LookUpTable(func=torch.tanh)
        else:
            # due to restrictive condition of int32 data range,
            # in linear fitting certain param will be clipped while slope
            # is near to zero, so here we set zero linear fitting interval.
            # use table and constant fit is enuogh to fit the tanh function
            lut = MultiTableFit(
                func=torch.tanh,
                dense_xmin=0.0,
                dense_xmax=2.0,
                sparse_xmin=2.0,
                sparse_xmax=5.0,
                left_line_xmin=0.0,
                left_line_xmax=0.0,
                right_line_xmin=5.0,
                right_line_xmax=5.0,
                out_type=activation_post_process.dtype,
                is_symmetric=True,
                symmetric_k=-1,
            )
        quantized_tanh = cls(lut)
        quantized_tanh.scale.copy_(activation_post_process.scale)
        return quantized_tanh
