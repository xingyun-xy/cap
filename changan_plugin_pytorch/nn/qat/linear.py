import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn
from changan_plugin_pytorch.nn import intrinsic
import torch.nn.intrinsic as nni
import torch.nn.functional as F


__all__ = [
    "Linear",
    "LinearReLU",
    "LinearReLU6",
    "LinearAdd",
    "LinearAddReLU",
    "LinearAddReLU6",
]


class Linear(nn.Linear):
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(Linear, self).__init__(in_features, out_features, bias)
        assert get_march() == March.BAYES
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight(channel_len=out_features)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation()

    def _get_weight_for_fake_quant(self):
        return self.weight.reshape(self.out_features, self.in_features, 1, 1)

    def _linear(self, input):
        out = torch.nn.functional.linear(
            input.as_subclass(torch.Tensor),
            self.weight_fake_quant(
                self.weight.reshape(self.out_features, self.in_features, 1, 1)
            )
            .reshape(self.weight.shape)
            .as_subclass(torch.Tensor),
            self.bias,
        )
        return out

    def forward(self, input: QTensor) -> QTensor:
        out = self._linear(input)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        return QTensor(out, scale=None, dtype="float32")

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        if type(mod) == nni.LinearReLU:
            mod = mod[0]
        elif (
            type(mod) == intrinsic.LinearAdd
            or type(mod) == intrinsic.LinearReLU6
            or type(mod) == intrinsic.LinearAddReLU
            or type(mod) == intrinsic.LinearAddReLU6
        ):
            mod = mod.linear
        qconfig = mod.qconfig
        qat_linear = cls(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            qconfig=qconfig,
        )
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        return qat_linear


class LinearReLU(Linear):
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearReLU, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def forward(self, input: QTensor) -> QTensor:
        out = self._linear(input)
        out = F.relu(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, scale=None, dtype="float32")

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU, cls).from_float(mod)


class LinearReLU6(Linear):
    _FLOAT_MODULE = intrinsic.LinearReLU6

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearReLU6, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def forward(self, input: QTensor) -> QTensor:
        out = self._linear(input)
        out = F.relu6(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, scale=None, dtype="float32")

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU6, cls).from_float(mod)


class LinearAdd(Linear):
    _FLOAT_MODULE = intrinsic.LinearAdd

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearAdd, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def forward(self, input1: QTensor, input2: QTensor) -> QTensor:
        out = self._linear(input1)
        out = out + input2.as_subclass(torch.Tensor)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, scale=None, dtype="float32")

    @classmethod
    def from_float(cls, mod):
        return super(LinearAdd, cls).from_float(mod)


class LinearAddReLU(LinearAdd):
    _FLOAT_MODULE = intrinsic.LinearAddReLU

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearAddReLU, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def forward(self, input1: QTensor, input2: QTensor) -> QTensor:
        out = self._linear(input1)
        out = out + input2.as_subclass(torch.Tensor)
        out = F.relu(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, scale=None, dtype="float32")

    @classmethod
    def from_float(cls, mod):
        return super(LinearAdd, cls).from_float(mod)


class LinearAddReLU6(LinearAdd):
    _FLOAT_MODULE = intrinsic.LinearAddReLU6

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearAddReLU6, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def forward(self, input1: QTensor, input2: QTensor) -> QTensor:
        out = self._linear(input1)
        out = out + input2.as_subclass(torch.Tensor)
        out = F.relu6(out, inplace=True)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return QTensor(out, scale=None, dtype="float32")

    @classmethod
    def from_float(cls, mod):
        return super(LinearAddReLU6, cls).from_float(mod)
