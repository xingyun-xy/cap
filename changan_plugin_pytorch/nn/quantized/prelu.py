import torch
import torch.nn.functional as F
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor

from .functional import prelu
from .segment_lut import SegmentLUT


class PReLU(torch.nn.Module):
    _QAT_MODULE = qat.PReLU

    def __init__(self, num_parameters=1, out_dtype="qint8"):
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.register_buffer(
            "weight", torch.empty(num_parameters, dtype=torch.int16)
        )
        self.register_buffer(
            "weight_scale", torch.ones([1], dtype=torch.float32)
        )
        self.register_buffer("scale", torch.ones([1], dtype=torch.float32))
        self.weight_dtype = "qint16"
        self.out_dtype = out_dtype

    def forward(self, input):
        out = prelu(
            input.int_repr(),
            self.weight,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
            self.weight_scale,
            torch.zeros_like(self.weight_scale).to(torch.long),
            self.weight_dtype,
            self.scale,
            input.q_zero_point(),
            self.out_dtype,
        )
        return QTensor(out, self.scale, self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized." + cls.__name__,
            +".from_float only works for " + cls._QAT_MODULE.__name__,
        )
        # if single parameter, hardward uses leakyrelu(which is lut on HW)
        # instead to speed up
        if mod.num_parameters == 1:
            return SegmentLUT(
                simulated_func=lambda x: F.leaky_relu(
                    x, mod.weight.item(), False
                ),
                is_centrosymmetric=False,
                dividing_points=[0, 0, 0, 0, 0, 0, 0, 0],
                output_scale=mod.activation_post_process.scale,
                output_dtype=mod.activation_post_process.dtype,
                device=mod.weight.device,
            )
        quantized_mod = cls(
            num_parameters=mod.num_parameters,
            out_dtype=mod.activation_post_process.dtype,
        )
        quantized_mod.scale.copy_(mod.activation_post_process.scale)

        # convert weight from qfloat to quantized
        quantized_mod.weight_scale.copy_(mod.weight_fake_quanti.scale)
        quantized_mod.weight_dtype = mod.weight_fake_quanti.dtype
        assert (
            mod.weight_fake_quanti.dtype == "qint16"
        ), "Only support qint16 dtype weight in PReLU now"
        info = qinfo(mod.weight_fake_quanti.dtype)
        qweight = torch.clamp(
            torch.floor(mod.weight / mod.weight_fake_quanti.scale + 0.5),
            info.min,
            info.max,
        ).to(info._storage_type)
        quantized_mod.weight.copy_(qweight)
        return quantized_mod
