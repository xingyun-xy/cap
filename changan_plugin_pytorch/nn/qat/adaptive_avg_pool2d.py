import numpy as np
import torch
import torch.nn.functional as F
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import nn


class AdaptiveAvgPool2d(nn.Module):
    r"""qat version"""
    _FLOAT_MODULE = nn.AdaptiveAvgPool2d

    def __init__(
        self,
        output_size,
        qconfig=None,
    ) -> None:
        super(AdaptiveAvgPool2d, self).__init__()
        assert qconfig, "qconfig must be provided for QAT module"
        assert isinstance(output_size, (int, tuple, list)), (
            "ouput size must be int, tuple and list, but get {}"
        ).format(type(output_size))
        self.output_size = (
            np.array([output_size, output_size])
            if isinstance(output_size, int)
            else np.array(output_size)
        )
        self.qconfig = qconfig
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation()

    def _check_output_size(self, input_size, dim):
        if self.output_size[dim - 2] is None:
            self.output_size[dim - 2] = input_size
        else:
            assert input_size % self.output_size[dim - 2] == 0, (
                "Only support the case that input size can be divided equally"
                + "by output size, but give input size {} and output size {}"
                + " in dim {}"
            ).format(input_size, self.output_size[dim - 2], dim)

    def forward(self, input: QTensor) -> QTensor:
        self._check_output_size(input.shape[2], 2)
        self._check_output_size(input.shape[3], 3)
        self.output_size = self.output_size.astype(np.int)

        if get_march() == March.BERNOULLI:
            assert_qtensor_input(input)

            input_size = np.array([input.shape[2], input.shape[3]])
            strides = np.floor(input_size / self.output_size).astype(np.int32)
            kernels = (input_size - (self.output_size - 1) * strides).astype(
                np.int32
            )
            hw_reciprocal = 1 / kernels[0] / kernels[1]
            # avg = accumulator * (int(hw_reciprocal * 2 ** 9) / 2 ** 9)
            divisor_shift = 9
            out = (
                F.adaptive_avg_pool2d(
                    input.as_subclass(torch.Tensor),
                    self.output_size,
                )
                / hw_reciprocal
            ) * (
                int(hw_reciprocal * 2 ** divisor_shift)
                * (1.0 / (2 ** divisor_shift))
            )
            if self.activation_post_process is not None:
                self.activation_post_process.scale.copy_(input.q_scale())
                self.activation_post_process.disable_observer()
        else:
            out = F.adaptive_avg_pool2d(
                input.as_subclass(torch.Tensor),
                self.output_size,
            )
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        return QTensor(out, scale=None, dtype="float32")

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
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_pool = cls(
            mod.output_size,
            qconfig=qconfig,
        )
        return qat_pool
