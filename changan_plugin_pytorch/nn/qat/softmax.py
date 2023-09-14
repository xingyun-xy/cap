import torch
from changan_plugin_pytorch.dtype import qint16
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch.nn import Module, functional


class QuantiSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, scale, input_type, output_type, max_softmax_value):
        types = {"qint8": torch.int8, "qint16": torch.int16}
        type_info = torch.iinfo(types[input_type])
        qxmin = type_info.min
        qxmax = type_info.max
        qtensor = QTensor(
            torch.clamp(torch.round(data / scale), qxmin, qxmax).to(
                torch.int8
            ),
            scale,
            input_type,
        )
        from changan_plugin_pytorch.nn.quantized.softmax import QuantSoftmax

        softmax = QuantSoftmax(output_type, max_softmax_value).to(data.device)
        q_res = softmax(qtensor)
        q_res = q_res.dequantize()
        ctx.save_for_backward(data, q_res)
        return q_res

    @staticmethod
    def backward(ctx, grad_out):
        data, softmax_res = ctx.saved_tensors
        in_grad = torch._softmax_backward_data(
            output=softmax_res, dim=1, grad_output=grad_out, input=data
        )
        return in_grad, None, None, None, None


class Softmax(torch.nn.Module):

    _FLOAT_MODULE = torch.nn.Softmax

    def __init__(self, qconfig=None):
        super(Softmax, self).__init__()
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.activation_post_process = self.qconfig.activation()
        self.register_buffer("max_softmax_value", torch.tensor([0.0]))

    def forward(self, input):
        assert_qtensor_input(input)
        assert input.dtype == "qint8", "Only support qint8 input"

        with torch.no_grad():
            max_softmax = torch.max(
                torch.softmax(
                    input.as_subclass(torch.Tensor).detach()
                    - torch.max(
                        input.as_subclass(torch.Tensor).detach(),
                        dim=1,
                        keepdims=True,
                    ).values,
                    dim=1,
                )
            )
        if self.max_softmax_value < max_softmax:
            self.max_softmax_value.copy_(max_softmax.clone().detach())
        return self.activation_post_process(
            QuantiSoftmaxFunction.apply(
                input.as_subclass(torch.Tensor),
                input.q_scale(),
                input.dtype,
                self.activation_post_process.dtype,
                self.max_softmax_value,
            )
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        if get_march() == March.BAYES:
            return SegmentLUTSoftmax.from_float(mod)
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        assert mod.dim in [None, 1], "Only support softmax along channel dim"
        qconfig = mod.qconfig
        qat_softmax = cls(qconfig=qconfig)
        return qat_softmax


class SegmentLUTSoftmax(Module):
    _FLOAT_MODULE = torch.nn.Softmax

    def __init__(
        self,
        dim=None,
        qconfig=None,
    ):
        super(SegmentLUTSoftmax, self).__init__()
        assert qconfig is not None, "qconfig must be provided"
        assert (
            qconfig.activation is not None
        ), "qconfig.activation must be provided"

        self.dim = dim
        self.qconfig = qconfig
        from changan_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)

        from changan_plugin_pytorch.nn.quantized.functional_modules import (
            FloatFunctional,
        )

        from .segment_lut import SegmentLUT

        self.sub = FloatFunctional(qconfig=int16_qconfig)
        self.exp = SegmentLUT(torch.exp, False, qconfig=int16_qconfig)
        self.sum = FloatFunctional(qconfig=int16_qconfig)
        self.reciprocal = SegmentLUT(
            torch.reciprocal, True, qconfig=int16_qconfig
        )
        self.mul = FloatFunctional(qconfig=qconfig)

    def propagate_qconfig(self, qconfig):
        from changan_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)
        self.qconfig = qconfig
        self.sub.qconfig = int16_qconfig
        self.exp.qconfig = int16_qconfig
        self.sum.qconfig = int16_qconfig
        self.reciprocal.qconfig = int16_qconfig
        self.mul.qconfig = qconfig

    def forward(self, input: QTensor):
        if self.dim is None:
            self.dim = functional._get_softmax_dim("softmax", input.dim(), 5)
        input = self.sub.sub(input, input.max(dim=self.dim, keepdim=True)[0])
        exp = self.exp(input)
        exp_sum = self.sum.sum(exp, dim=self.dim, keepdim=True)
        exp_sum_reciprocal = self.reciprocal(exp_sum)
        ret = self.mul.mul(exp, exp_sum_reciprocal)
        return ret

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
        qat_softmax = cls(mod.dim, qconfig=qconfig)
        return qat_softmax
