import torch
import torch.nn.functional as F
from changan_plugin_pytorch.nn import qat

from .multi_table_fit import MultiTableFit


class SiLU(MultiTableFit):
    _QAT_MODULE = qat.SiLU

    def __init__(self, output_type="qint8"):
        super(SiLU, self).__init__(
            func=F.silu,
            dense_xmin=0.0,
            dense_xmax=20.0,
            sparse_xmin=-20.0,
            sparse_xmax=0.0,
            left_line_xmin=-40.0,
            left_line_xmax=-20.0,
            right_line_xmin=20.0,
            right_line_xmax=40.0,
            out_type=output_type,
        )
        self.register_buffer("scale", torch.tensor([1], dtype=torch.float32))

    def forward(self, data):
        return super().forward(data, self.scale)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args: `mod` a float module
        """
        activation_post_process = mod.activation_post_process
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        quantized_silu = cls(output_type=activation_post_process.dtype)
        quantized_silu.scale.copy_(activation_post_process.scale)
        return quantized_silu
