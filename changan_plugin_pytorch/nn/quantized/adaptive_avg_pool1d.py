import numpy as np
from changan_plugin_pytorch.nn import qat

from .adaptive_avg_pool2d import AdaptiveAvgPool2d


class AdaptiveAvgPool1d(AdaptiveAvgPool2d):
    r"""quantize version"""
    _QAT_MODULE = qat.AdaptiveAvgPool1d

    def __init__(
        self,
        output_size,
        out_dtype="qint8",
    ):
        output_size = np.array([1, output_size[0]])
        super(AdaptiveAvgPool1d, self).__init__(output_size, out_dtype)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2]))
        out = super(AdaptiveAvgPool1d, self).forward(x)
        return out.reshape((out.shape[0], out.shape[1], out.shape[3]))

    @classmethod
    def from_float(cls, mod):
        return super(AdaptiveAvgPool1d, cls).from_float(mod)
