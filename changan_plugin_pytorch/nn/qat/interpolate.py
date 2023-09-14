import warnings

import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import interpolate as float_interpolate
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import nn
from torch.nn.modules.utils import _pair

from .functional import interpolate


class Interpolate(nn.Module):
    """Resize for quantized training. Support bilinear and nearest
    interpolate method.

    Parameters
    ----------
    Same as float version.
    """

    _FLOAT_MODULE = float_interpolate.Interpolate
    _version = 2

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=None,
        recompute_scale_factor=None,
        qconfig=None,
    ):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        assert self.qconfig.activation, (
            "activation_post_process must included "
            + "in qconfig for qat.Interpolate"
        )
        self.activation_post_process = self.qconfig.activation()
        self.activation_post_process.disable_observer()

        self.quantized_forward = True

        self._check_init()

    def _check_init(self):
        assert self.mode in (
            "bilinear",
            "nearest",
        ), "mode only support bilinear and nearest"
        if self.mode == "nearest":
            assert (
                self.align_corners is None
            ), "align_corners option can only be set with 'bilinear' mode"
        else:
            if self.align_corners is None:
                warnings.warn(
                    "default upsampling behavior when mode={} is changed "
                    "to align_corners=False since torch 0.4.0. Please specify "
                    "align_corners=True if the old behavior "
                    "is desired. ".format(self.mode)
                )
                self.align_corners = False
        # only support align_corners=True on BAYES
        if get_march() in (March.BERNOULLI2, March.BERNOULLI):
            assert (
                not self.align_corners
            ), "only support align_corners=True on BAYES"

        if self.scale_factor:
            assert self.recompute_scale_factor, (
                "only support recompute_scale_factor=True "
                + "when using scale_factor"
            )
        size_is_none = self.size is None
        scale_factor_is_none = self.scale_factor is None
        assert (
            size_is_none != scale_factor_is_none
        ), "either size or scale_factor should be set"
        if not size_is_none:
            out_height, out_width = _pair(self.size)
            assert (
                out_height > 0 and out_width > 0
            ), "size should not be negative"
        if not scale_factor_is_none:
            ratio_height, ratio_width = _pair(self.scale_factor)
            assert (
                ratio_height > 0 and ratio_width > 0
            ), "scale_factor should not be negative"

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version == 1:
            self.quantized_forward = False

        super(Interpolate, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, data):
        assert_qtensor_input(data)

        self.activation_post_process.set_qparams(data.q_scale())
        out = interpolate(
            data.as_subclass(torch.Tensor),
            _pair(self.size) if self.size else None,
            _pair(self.scale_factor) if self.scale_factor else None,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
            data.q_scale(),
            data.q_zero_point(),
            data.dtype,
            self.quantized_forward,
        )

        return self.activation_post_process(out)

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
            size=mod.size,
            scale_factor=mod.scale_factor,
            mode=mod.mode,
            align_corners=mod.align_corners,
            recompute_scale_factor=mod.recompute_scale_factor,
            qconfig=mod.qconfig,
        )
        return qat_mod
