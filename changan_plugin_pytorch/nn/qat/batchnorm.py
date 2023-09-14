import torch
from changan_plugin_pytorch.qtensor import QTensor


class BatchNorm2d(torch.nn.Module):
    _FLOAT_MODULE = torch.nn.BatchNorm2d

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        qconfig=None,
    ):
        super(BatchNorm2d, self).__init__()
        self.bn = torch.nn.BatchNorm2d(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation()

    def forward(self, input: QTensor) -> QTensor:
        out = self.bn(input.as_subclass(torch.Tensor))
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
        qconfig = mod.qconfig
        qat_bn = cls(
            mod.num_features,
            mod.eps,
            mod.momentum,
            mod.affine,
            mod.track_running_stats,
            qconfig=qconfig,
        )
        qat_bn.bn.weight = mod.weight
        qat_bn.bn.bias = mod.bias
        qat_bn.bn.running_mean = mod.running_mean
        qat_bn.bn.running_var = mod.running_var

        return qat_bn
