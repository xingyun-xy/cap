import numbers
from typing import List, Union

import torch
from changan_plugin_pytorch.dtype import qint16
from changan_plugin_pytorch.qtensor import QTensor
from torch import Size, nn

from ..layer_norm import LayerNorm as HorizonLayerNorm
from .segment_lut import SegmentLUT
from .stubs import QuantStub


class MultiDimMean(nn.Module):
    def __init__(self, dims, qconfig):
        super(MultiDimMean, self).__init__()
        self.dims = dims

        from changan_plugin_pytorch.nn.quantized import FloatFunctional

        from .avg_pool2d import AvgPool2d

        if len(dims) == 1:
            self.pre_mean = FloatFunctional(qconfig)
            self.avg_pooling = None
            self.post_mean = None
        else:
            self.pre_mean = None
            self.avg_pooling = AvgPool2d(3, qconfig=qconfig)
            if len(dims) > 2:
                self.post_mean = FloatFunctional(qconfig)
            else:
                self.post_mean = None

    def forward(self, x):
        if self.pre_mean:
            x = self.pre_mean.mean(x, dim=self.dims[0])
        if self.avg_pooling:
            self.avg_pooling.kernel_size = x.shape[-2:]
            if x.ndim < 4:
                N, C, L = x.shape
                x = self.avg_pooling(x.reshape([N, 1, C, L])).reshape(N, 1, 1)
            else:
                x = self.avg_pooling(x)
        if self.post_mean:
            x = self.post_mean.mean(x, dim=1)
        return x


class LayerNorm(nn.LayerNorm):
    r"""qat version"""
    _FLOAT_MODULE = (nn.LayerNorm, HorizonLayerNorm)

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        dim=None,
        qconfig=None,
    ) -> None:
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        assert isinstance(
            normalized_shape, (list, tuple, Size)
        ), "normalized_shape muse be a list or intergral or tuple or torch.Size"  # noqa
        assert (
            len(normalized_shape) < 4
        ), "Only support layernorm on W or HW or CHW."
        for v in normalized_shape:
            assert isinstance(
                v, numbers.Integral
            ), "elements of normalized_shape must be integral"
        assert isinstance(eps, float), "param eps must be a float"
        assert isinstance(
            elementwise_affine, bool
        ), "param elementwise_affine must be a bool"
        assert isinstance(
            dim, (type(None), numbers.Integral)
        ), "param dim must be None or a integral"
        if dim is None:
            assert len(normalized_shape) in (
                1,
                2,
                3,
            ), "Only support layernorm on W or HW or CHW."

        assert qconfig is not None, "qconfig must be provided"
        assert (
            qconfig.activation is not None
        ), "qconfig.activation must be provided"
        assert (
            qconfig.weight is not None
        ), "qconfig.activation must be provided"

        super(LayerNorm, self).__init__(
            normalized_shape,
            eps,
            elementwise_affine,
            device,
            dtype,
        )

        self.dims = (
            tuple(reversed(range(-1, -len(normalized_shape) - 1, -1)))
            if dim is None
            else (dim,)
        )

        self.qconfig = qconfig

        from changan_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)

        from changan_plugin_pytorch.nn.quantized import FloatFunctional

        self.input_mean = MultiDimMean(self.dims, qconfig=int16_qconfig)
        self.sub = FloatFunctional(qconfig=int16_qconfig)
        self.mul = FloatFunctional(
            qconfig=qconfig if len(self.dims) > 1 else int16_qconfig
        )
        self.var_mean = MultiDimMean(self.dims, qconfig=int16_qconfig)
        self.rsqrt = SegmentLUT(
            lambda x: torch.rsqrt(x + self.eps),
            True,
            None,
            qconfig=int16_qconfig,
        )

        if self.elementwise_affine:
            self.out_mul = FloatFunctional(qconfig=int16_qconfig)
            self.weight_quant = QuantStub(qconfig=int16_qconfig)
            self.weight_mul = FloatFunctional(qconfig=int16_qconfig)
            self.bias_quant = QuantStub(qconfig=int16_qconfig)
            self.bias_add = FloatFunctional(qconfig=qconfig)
        else:
            self.out_mul = FloatFunctional(qconfig=qconfig)

    def propagate_qconfig(self, qconfig):
        from changan_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)
        self.qconfig = qconfig
        self.input_mean.qconfig = int16_qconfig
        self.sub.qconfig = int16_qconfig
        self.mul.qconfig = qconfig if len(self.dims) > 1 else int16_qconfig
        self.var_mean.qconfig = int16_qconfig
        self.rsqrt.qconfig = int16_qconfig

        if self.elementwise_affine:
            self.out_mul.qconfig = int16_qconfig
            self.weight_quant.qconfig = int16_qconfig
            self.weight_mul.qconfig = int16_qconfig
            self.bias_quant.qconfig = int16_qconfig
            self.bias_add.qconfig = qconfig
        else:
            self.out_mul.qconfig = qconfig

    def forward(self, input: QTensor) -> QTensor:
        mu = self.input_mean(input)
        diff = self.sub.sub(input, mu)
        diff_square = self.mul.mul(diff, diff)
        var = self.var_mean(diff_square)
        dev_rec = self.rsqrt(var)
        ret = self.out_mul.mul(diff, dev_rec)

        if self.elementwise_affine:
            ret = self.weight_mul.mul(ret, self.weight_quant(self.weight))
            ret = self.bias_add.add(ret, self.bias_quant(self.bias))

        return ret

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) in cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + (c.__name__ for c in cls._FLOAT_MODULE)
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_mod = cls(
            normalized_shape=mod.normalized_shape,
            eps=mod.eps,
            elementwise_affine=mod.elementwise_affine,
            device=mod.weight.device if mod.weight is not None else None,
            dtype=mod.weight.dtype if mod.weight is not None else None,
            dim=mod.dim if hasattr(mod, "dim") else None,
            qconfig=qconfig,
        )
        if mod.elementwise_affine:
            with torch.no_grad():
                qat_mod.weight.copy_(mod.weight)
                qat_mod.bias.copy_(mod.bias)
        return qat_mod
