import warnings
from numbers import Real

import torch
import torch.nn.quantized as nnq
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils import fx_helper
from changan_plugin_pytorch.utils.model_helper import call_with_hooks
from torch.jit.annotations import List
from torch.nn.quantized import FloatFunctional as TorchFloatFunctional

from .div import Div
from .exp import Exp
from .functional import add, cap, matmul, mean, mul, requantize, sub, sum


@fx_helper.wrap
class FloatFunctional(torch.nn.Module):
    r"""float and qat functionals"""

    _FLOAT_MODULE = nnq.FloatFunctional

    def __init__(self, qconfig=None):
        super(FloatFunctional, self).__init__()
        if qconfig is None:
            self.activation_post_process = torch.nn.Identity()
        else:
            self.qconfig = qconfig
            self.activation_post_process = qconfig.activation()

    def forward(self, x):
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    @call_with_hooks
    def add(self, x, y):
        r = torch.add(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def add_scalar(self, x: torch.Tensor, y: Real) -> torch.Tensor:
        assert isinstance(x, torch.Tensor) and isinstance(
            y, Real
        ), "add_scalar only support torch.Tensor + scalar!"
        if y == 0:
            if not isinstance(self.activation_post_process, torch.nn.Identity):
                self.activation_post_process.disable_observer()
                self.activation_post_process.set_qparams(x.q_scale())
            r = x.as_subclass(torch.Tensor)
        else:
            r = torch.add(x.as_subclass(torch.Tensor), y)
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def sub(self, x, y):
        r = torch.sub(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def cap(self, x, dim=0):
        same_scale = False
        if isinstance(x[0], QTensor):
            same_scale = True
            for rest in x:
                if (
                    rest.q_scale() != x[0].q_scale()
                    or rest.dtype != self.activation_post_process.dtype
                ):
                    same_scale = False
                    break

        r = torch.cat(
            [qt.as_subclass(torch.Tensor) for qt in x],
            dim=dim,
        )

        if same_scale:
            self.activation_post_process.disable_observer()
            self.activation_post_process.set_qparams(x[0].scale)
        else:
            if hasattr(self.activation_post_process, "enable_observer"):
                self.activation_post_process.enable_observer()

        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def matmul(self, x, y, x_trans=False, y_trans=False):
        if isinstance(x, QTensor):
            x = x.as_subclass(torch.Tensor)
            y = y.as_subclass(torch.Tensor)

        r = torch.matmul(
            torch.transpose(x, -1, -2) if x_trans else x,
            torch.transpose(y, -1, -2) if y_trans else y,
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def mul(self, x, y):
        r = torch.mul(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        if "qbool" in (x.dtype, y.dtype):
            if self.activation_post_process._observer_enabled:
                self.activation_post_process.disable_fake_quant()
                self.activation_post_process.disable_observer()
            if x.dtype == "qbool":
                oscale = y.q_scale()
                zero_point = y.q_zero_point()
            else:
                oscale = x.q_scale()
                zero_point = x.q_zero_point()
            self.activation_post_process.set_qparams(oscale, zero_point)
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def mul_scalar(self, x: torch.Tensor, y: Real) -> torch.Tensor:
        assert isinstance(x, torch.Tensor) and isinstance(
            y, Real
        ), "mul_scalar only support torch.Tensor * scalar!"
        r = torch.mul(x.as_subclass(torch.Tensor), y)
        if not isinstance(self.activation_post_process, torch.nn.Identity):
            self.activation_post_process.disable_observer()
            self.activation_post_process.set_qparams(x.q_scale() * abs(y))
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def div(self, x, y):
        warnings.warn(
            "\033[31mchangan_plugin_pytorch.nn.quantized.Floatfunctional.div "
            + "will be deprecated. Please use module "
            + "changan_plugin_pytorch.nn.Div!\033[0m",
            DeprecationWarning,
        )
        r = torch.div(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def sum(self, x, dim, keepdim):
        r = torch.sum(x.as_subclass(torch.Tensor), dim, keepdim)
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def exp(self, x):
        warnings.warn(
            "\033[31mchangan_plugin_pytorch.nn.quantized.Floatfunctional.exp "
            + "will be deprecated. Please use module "
            + "changan_plugin_pytorch.nn.Exp!\033[0m",
            DeprecationWarning,
        )
        r = torch.exp(x.as_subclass(torch.Tensor))
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def mean(self, x, dim):
        if get_march() == March.BERNOULLI and isinstance(x, QTensor):
            n = x.shape[dim]
            m, e = torch.frexp(
                torch.tensor(1 / n, device=x.as_subclass(torch.Tensor).device)
            )
            qm = torch.clamp(
                torch.floor(m * 128 + 0.5), -128, 127
            ) * torch.pow(2.0, e - 7)
            r = torch.sum(x.as_subclass(torch.Tensor), dim, True) * qm
        else:
            r = torch.mean(x.as_subclass(torch.Tensor), dim, True)
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def maximum(self, x, y):
        r = torch.maximum(
            x.as_subclass(torch.Tensor), y.as_subclass(torch.Tensor)
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    def minimum(self, x, y):
        r = torch.minimum(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        r = self.activation_post_process(r)
        return r

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) in (cls._FLOAT_MODULE, cls), (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + [modc.__name__ for modc in (cls._FLOAT_MODULE, cls)]
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qat_func = cls(mod.qconfig)

        return qat_func


@fx_helper.wrap
class QFunctional(torch.nn.Module):
    r"""quantized version"""
    _QAT_MODULE = FloatFunctional

    def __init__(self, out_dtype):
        super(QFunctional, self).__init__()
        # register scale
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.out_dtype = out_dtype
        self._div = Div(self.scale)
        self._exp = Exp(self.scale, out_dtype)

    def forward(self, x):
        raise RuntimeError(
            "QFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    @call_with_hooks
    def add(self, x: QTensor, y: QTensor):
        r = add(
            x.int_repr(),
            y.int_repr(),
            x.q_scale(),
            y.q_scale(),
            x.q_zero_point(),
            y.q_zero_point(),
            x.dtype,
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(
            r,
            self.scale,
            dtype=self.out_dtype,
            per_channel_axis=-1 if self.scale.numel() == 1 else 1,
        )

    @call_with_hooks
    def add_scalar(self, x: QTensor, y: Real) -> QTensor:
        if y == 0:
            r = x.int_repr()
        else:
            other_data = y / abs(y)
            other_scale = abs(y)

            # guarantee result precision after requantization
            if x.dtype == "qint16" and get_march() != March.BERNOULLI2:
                other_data = other_data * 32767
                other_scale = other_scale / 32767
            r = add(
                x.int_repr(),
                torch.tensor([[[[other_data]]]]).to(
                    x.as_subclass(torch.Tensor)
                ),
                x.q_scale(),
                torch.tensor([other_scale], dtype=torch.float32).to(x.device),
                x.q_zero_point(),
                x.q_zero_point(),
                x.dtype,
                x.dtype,
                self.scale,
                torch.zeros_like(self.scale).to(torch.long),
                self.out_dtype,
            )
        return QTensor(
            r,
            self.scale,
            dtype=self.out_dtype,
            per_channel_axis=-1 if self.scale.numel() == 1 else 1,
        )

    @call_with_hooks
    def sub(self, x: QTensor, y: QTensor):
        r = sub(
            x.int_repr(),
            y.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def cap(self, x, dim=0):
        # type: (List[QTensor], int) -> QTensor
        r = cap(
            [qt.int_repr() for qt in x],
            dim,
            [qt.q_scale() for qt in x],
            [qt.q_zero_point() for qt in x],
            [qt.dtype for qt in x],
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def matmul(self, x, y, x_trans=False, y_trans=False):
        r = matmul(
            x.int_repr(),
            y.int_repr(),
            x_trans,
            y_trans,
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def mul(self, x, y):
        r = mul(
            x.int_repr(),
            y.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def mul_scalar(self, x: QTensor, y: Real) -> QTensor:
        scalar = 0 if y == 0 else y / abs(y)
        # if positive, directly return QTensor, avoid extra 'mul' on HW
        if scalar == 1:
            r = x.int_repr()
        else:
            r = mul(
                x.int_repr(),
                torch.tensor([[[[scalar]]]]).to(x.as_subclass(torch.Tensor)),
                x.q_scale(),
                x.q_zero_point(),
                x.dtype,
                torch.tensor([abs(y)], dtype=torch.float32).to(x.device),
                x.q_zero_point(),
                x.dtype,
                self.scale,
                x.q_zero_point(),
                x.dtype,
            )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def div(self, x, y):
        return self._div(x, y)

    @call_with_hooks
    def sum(self, x, dim, keepdim):
        r = sum(
            x.int_repr(),
            dim,
            keepdim,
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def exp(self, x):
        return self._exp(x)

    @call_with_hooks
    def mean(self, x, dim):
        r = mean(
            x.int_repr(),
            dim,
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def maximum(self, x, y):
        x = requantize(
            x.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        y = requantize(
            y.int_repr(),
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        r = torch.maximum(x, y)
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    def minimum(self, x, y):
        x = requantize(
            x.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        y = requantize(
            y.int_repr(),
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        r = torch.minimum(x, y)
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        out_dtype = (
            mod.activation_post_process.dtype
            if mod.activation_post_process is not None
            else "qint32"
        )
        func = cls(out_dtype)
        func.scale.resize_as_(mod.activation_post_process.scale)
        func.scale.copy_(mod.activation_post_process.scale)
        return func
