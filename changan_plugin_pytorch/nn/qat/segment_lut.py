import torch
from changan_plugin_pytorch.nn import segment_lut as float_segment_lut
from changan_plugin_pytorch.qtensor import QTensor
from torch import Tensor
from torch.nn import Module


class SegmentLUT(Module):
    _FLOAT_MODULE = float_segment_lut.SegmentLUT

    def __init__(
        self,
        simulated_func,
        is_centrosymmetric=False,
        dividing_points=None,
        input_range=None,
        auto_divide_strategy="evenly",
        inverse_func=None,
        qconfig=None,
    ):
        super(SegmentLUT, self).__init__()

        self.simulated_func = simulated_func
        self.is_centrosymmetric = is_centrosymmetric
        self.dividing_points = dividing_points
        self.input_range = input_range
        self.auto_divide_strategy = auto_divide_strategy
        self.inverse_func = inverse_func

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        assert self.qconfig.activation, (
            "activation_post_process must included "
            + "in qconfig for qat.SegmentLUT"
        )
        self.activation_post_process = self.qconfig.activation()

    def forward(self, input: QTensor):
        out = self.simulated_func(input.as_subclass(Tensor))
        return self.activation_post_process(out)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # we removed the input_scale buffer
        key = prefix + "input_scale"
        if key in state_dict:
            state_dict.pop(key)

        super(SegmentLUT, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

        qat_mod = cls(
            simulated_func=mod.simulated_func,
            is_centrosymmetric=mod.is_centrosymmetric,
            dividing_points=mod.dividing_points,
            input_range=mod.input_range,
            auto_divide_strategy=mod.auto_divide_strategy,
            inverse_func=mod.inverse_func,
            qconfig=mod.qconfig,
        )
        return qat_mod
