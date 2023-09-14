import torch
from changan_plugin_pytorch.nn.lut import LookUpTable
from changan_plugin_pytorch.qtensor import QTensor


class LookUpTable(torch.nn.Module):
    r"""apply look up table operator

    Args:
        table: tuple, table for looking up
    """

    _FLOAT_MODULE = LookUpTable

    def __init__(self, table, qconfig=None):
        super(LookUpTable, self).__init__()
        self.register_buffer("table", table)
        self.qconfig = qconfig

    def forward(self, index):
        return QTensor(
            torch.take(self.table, index.int_repr().to(torch.long) + 128),
            torch.tensor(
                [2 / 256], dtype=torch.float, device=self.table.device
            ),
            self.table.dtype,
        )

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
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_lut = cls(mod.table, qconfig=qconfig)
        return qat_lut
