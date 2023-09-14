"""
Qat mode specify the way that ConvBN* handle the BN operation.
"""
import torch
import warnings
from changan_plugin_pytorch.utils import format_msg

__all__ = [
    "QATMode",
    "set_qat_mode",
    "get_qat_mode",
]


class QATMode(object):

    WithBN = "with_bn"
    FuseBN = "fuse_bn"
    WithBNReverseFold = "with_bn_reverse_fold"


_qat_mode = QATMode.FuseBN


def set_qat_mode(qat_mode):
    global _qat_mode
    assert qat_mode in [
        QATMode.FuseBN,
        QATMode.WithBN,
        QATMode.WithBNReverseFold,
    ]
    _qat_mode = qat_mode


def get_qat_mode():
    global _qat_mode
    return _qat_mode


class Tricks:
    def __init__(self, relu6=False):
        self.relu6 = relu6

    @property
    def relu6(self):
        return self._relu6

    @relu6.setter
    def relu6(self, relu6):
        if relu6:
            warnings.warn(
                format_msg("relu6 trick will be deprecated in v0.15.2", "r")
            )
        self._relu6 = relu6


tricks = Tricks()
