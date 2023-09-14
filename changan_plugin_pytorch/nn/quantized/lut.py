import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module

from .functional import lut
from .table_generator import SingleTableGenerator
from changan_plugin_pytorch.march import get_march


class LookUpTable(Module):
    r"""
    Only one table in LookUpTable.
    The table initialization takes place in class construction or forward.
    If in construction, float table and convert func must be given
    In forward, float table and qint table will be calculated by activation
    function
    """

    _QAT_MODULE = qat.LookUpTable

    def __init__(self, func=None, float_table=None, out_type="qint8"):
        super(LookUpTable, self).__init__()
        self.float_table = float_table
        self.out_type = out_type
        if float_table is not None:
            self.func = func
            self.float_table = float_table
            self.table = func(float_table)
            out_scale = torch.tensor([2 / 256])
            self.register_buffer("out_scale", out_scale)
        else:
            self.table_generator = torch.jit.script(
                SingleTableGenerator(func, get_march())
            )

    def forward(self, data, scale=None):
        if self.float_table is None:
            self.out_scale = scale
            self.table = self.table_generator(
                data.q_scale(), scale, data.dtype, self.out_type
            )
        out = lut(
            data.int_repr(),
            data.q_scale(),
            data.q_zero_point(),
            data.dtype,
            self.table,
            self.out_scale,
            torch.zeros_like(self.out_scale).to(dtype=torch.long),
            self.out_type,
        )
        return QTensor(out, self.out_scale, self.out_type)

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
        quantized_lut = cls(
            func=lambda x: (x * 128).to(torch.int32),
            float_table=mod.table,
            out_type=mod.qconfig.activation().dtype,
        )
        return quantized_lut
