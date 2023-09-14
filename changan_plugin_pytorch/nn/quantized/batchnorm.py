import torch
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor

from .functional import conv2d


class BatchNorm2d(torch.nn.Module):
    _QAT_MODULE = qat.BatchNorm2d

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        out_dtype="qint8",
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.out_dtype = out_dtype

        self.register_buffer(
            "weight", torch.ones(self.num_features, 1, 1, 1)
        )  # dummy
        self.register_buffer(
            "weight_scale", torch.ones((self.num_features,))
        )  # lambda / sigma
        self.register_buffer(
            "weight_zero_point",
            torch.zeros(self.num_features, dtype=torch.int64),
        )
        self.register_buffer(
            "zero_point", torch.zeros(self.num_features, dtype=torch.int64)
        )
        self.register_buffer(
            "bias", torch.zeros(num_features)
        )  # -lambda * mean / sigma + beta
        self.register_buffer("scale", torch.ones(1))  # out_scale
        self.register_buffer("bias_scale", torch.ones(num_features))
        self.register_buffer(
            "bias_zero_point", torch.zeros(num_features, dtype=torch.int64)
        )

    @torch.no_grad()
    def forward(self, input):
        out, dequant_out_scale = conv2d(
            input=input.int_repr(),
            weight=self.weight,
            bias=self.bias,
            sumin=None,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=self.num_features,
            padding_mode="zeros",
            activation="",
            input_scale=input.q_scale(),
            input_zero_point=input.q_zero_point(),
            input_dtype=input.dtype,
            weight_scale=self.weight_scale,
            weight_zero_point=self.weight_zero_point,
            weight_dtype="qint8",
            bias_scale=self.bias_scale,
            bias_zero_point=self.bias_zero_point,
            bias_dtype="qint32",
            sumin_scale=None,
            sumin_zero_point=None,
            sumin_dtype=None,
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=self.out_dtype,
        )
        return QTensor(out, self.scale, self.out_dtype, per_channel_axis=-1)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        activation_post_process = mod.activation_post_process
        out_dtype = (
            activation_post_process.dtype
            if activation_post_process is not None
            else None
        )
        assert out_dtype is not None
        qbn = cls(
            num_features=mod.bn.num_features,
            eps=mod.bn.eps,
            momentum=mod.bn.momentum,
            affine=mod.bn.affine,
            track_running_stats=True,
            out_dtype=out_dtype,
        )
        with torch.no_grad():
            running_std = torch.sqrt(mod.bn.running_var + mod.bn.eps)
            qbn.weight_scale.copy_(torch.abs(mod.bn.weight) / running_std)
            qbn.weight.copy_(
                (mod.bn.weight / running_std).reshape(-1, 1, 1, 1)
            )
            qbn.bias.copy_(
                -mod.bn.weight * mod.bn.running_mean / running_std
                + mod.bn.bias
            )
            qbn.scale.copy_(activation_post_process.scale)
            return qbn
