""" quant dtype """
from typing import Dict, Tuple

import torch

__all__ = ["qinfo", "qint4", "qint8", "qint16", "quint4"]

_qtype_limit: Dict[str, Tuple[int, int]] = {
    "qint4": (-8, 7),
    "quint4": (0, 15),
    "qint8": (-128, 127),
    "qint16": (-32768, 32767),
    "qint32": (-2147483648, 2147483647),
    "qbool": (0, 1),
}
_storage_type_map = {
    "qint4": torch.int8,
    "quint4": torch.int8,
    "qint8": torch.int8,
    "qint16": torch.int16,
    "qint32": torch.int32,
    "qbool": torch.bool,
}


# wrap string dtype into this class to make it act like a torch dtype
class QuantDType(str):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if self in _qtype_limit:
            self.min, self.max = _qtype_limit[self]
            self.storage_type = _storage_type_map[self]
        self.is_complex = False
        self.is_floating_point = False
        self.is_signed = "uint" not in self


qint4 = QuantDType("qint4")
quint4 = QuantDType("quint4")
qint8 = QuantDType("qint8")
qint16 = QuantDType("qint16")
qint32 = QuantDType("qint32")
qbool = QuantDType("qbool")

horizon_quant_dtype = (qint4, quint4, qint8, qint16, qint32, qbool)


def get_horizon_quant_dtype(dtype):
    """replace torch quant dtype with changan quant dtype"""
    if dtype in horizon_quant_dtype:
        return dtype
    else:
        torch_q_dtype_map = {torch.qint8: qint8}
        assert (
            dtype in torch_q_dtype_map
        ), "only support torch.qint8 and changan dtype in (qint8, qint16, qint32), but receive {}".format(  # noqa
            dtype
        )
        return torch_q_dtype_map[dtype]


class qinfo(object):
    r"""limits for quant types."""

    def __init__(self, dtype: str):
        _qtype_limit = {
            "qint4": (-8, 7),
            "quint4": (0, 15),
            "qint8": (-128, 127),
            "qint16": (-32768, 32767),
            "qint32": (-2147483648, 2147483647),
            "qbool": (0, 1),
        }
        _storage_type_map = {
            "qint4": torch.int8,
            "quint4": torch.int8,
            "qint8": torch.int8,
            "qint16": torch.int16,
            "qint32": torch.int32,
            "qbool": torch.bool,
        }

        assert dtype in _qtype_limit, f"unsupported dtype: {dtype}"

        self.dtype = dtype
        self.min: float = _qtype_limit[self.dtype][0]
        self.max: float = _qtype_limit[self.dtype][1]
        self._storage_type = _storage_type_map[self.dtype]
