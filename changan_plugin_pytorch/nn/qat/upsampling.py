import torch
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch import nn
from torch.nn.modules.utils import _pair

from .functional import interpolate


class Upsample(nn.Module):
    r"""Upsamples a given multi-channel data.
    Only upport bilinear and nearest mode now

    One can either give a attr `scale_factor` or the target output attr:`size`
    to calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (
            int or
            Tuple[int] or
            Tuple[int, int] or
            Tuple[int, int, int], optional
        ):
            output spatial sizes

        scale_factor (
            float or
            Tuple[float] or
            Tuple[float, float] or
            Tuple[float, float, float], optional
        ):
            multiplier for spatial size.
            Has to match input size if it is a tuple.

        mode (str, optional):
            the upsampling algorithm:
            only support 'bilinear' and 'nearest' now.

        align_corners (bool, optional):
            only has effect when mode = 'bilinear'. by default False
    """

    _FLOAT_MODULE = nn.Upsample
    _version = 2

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode: str = "nearest",
        align_corners=False,
        qconfig=None,
    ):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = True if scale_factor else None

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

        super(Upsample, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input):
        assert_qtensor_input(input)

        self.activation_post_process.set_qparams(input.q_scale())
        out = interpolate(
            input.as_subclass(torch.Tensor),
            _pair(self.size) if self.size else None,
            _pair(self.scale_factor) if self.scale_factor else None,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
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

        if cls.__name__ == "Upsample":
            qat_mod = cls(
                size=mod.size,
                scale_factor=mod.scale_factor,
                mode=mod.mode,
                align_corners=mod.align_corners,
                qconfig=mod.qconfig,
            )
        else:
            qat_mod = cls(
                size=mod.size,
                scale_factor=mod.scale_factor,
                qconfig=mod.qconfig,
            )
        return qat_mod


class UpsamplingNearest2d(Upsample):
    r"""Applies a 2D nearest neighbor upsampling to an input signal composed of
    several input channels.

    To specify the scale, it takes either the :attr:`size` or
    the :attr:`scale_factor` as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]]]])

        >>> m = nn.UpsamplingNearest2d(scale_factor=2)
        >>> m(input)
        tensor([[[[ 1.,  1.,  2.,  2.],
                  [ 1.,  1.,  2.,  2.],
                  [ 3.,  3.,  4.,  4.],
                  [ 3.,  3.,  4.,  4.]]]])
    """

    _FLOAT_MODULE = nn.UpsamplingNearest2d

    def __init__(self, size=None, scale_factor=None, qconfig=None):
        super(UpsamplingNearest2d, self).__init__(
            size,
            scale_factor,
            mode="nearest",
            align_corners=False,
            qconfig=qconfig,
        )

    @classmethod
    def from_float(cls, mod):
        return super(UpsamplingNearest2d, cls).from_float(mod)


class UpsamplingBilinear2d(Upsample):
    r"""Applies a 2D bilinear upsampling to an input signal composed of several
    input channels.

    To specify the scale, it takes either the :attr:`size` or
    the :attr:`scale_factor` as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]]]])

        >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
        >>> m(input)
        tensor([[[[ 1.0000,  1.3333,  1.6667,  2.0000],
                  [ 1.6667,  2.0000,  2.3333,  2.6667],
                  [ 2.3333,  2.6667,  3.0000,  3.3333],
                  [ 3.0000,  3.3333,  3.6667,  4.0000]]]])
    """

    _FLOAT_MODULE = nn.UpsamplingBilinear2d

    def __init__(self, size=None, scale_factor=None, qconfig=None):
        super(UpsamplingBilinear2d, self).__init__(
            size,
            scale_factor,
            mode="bilinear",
            align_corners=False,
            qconfig=qconfig,
        )

    @classmethod
    def from_float(cls, mod):
        return super(UpsamplingBilinear2d, cls).from_float(mod)
