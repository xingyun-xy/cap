import warnings
from numbers import Integral, Real

import torch
from changan_plugin_pytorch.march import March, get_march


class Interpolate(torch.nn.Module):
    r"""Resize for float training.
    Support bilinear and nearest interpolate method and NCHW input.
    The behaviour is same as torch.nn.functional.interpolate except the default
    mode is 'bilinear'

    Parameters
    ----------
    size : int or tuple of int, optional
        the output shape of resize: if int, the output shape is (size, size)
        else the output shape is (out_height, out_width), by default None
        size and scale_factor shouldn't be set at the same time
    scale_factor : float or tuple of float, optional
        the ratio of output shape to input shape, ie. out_shape / in_shape,
        or (out_height / in_height, out_width / in_width), by default None
        size and scale_factor shouldn't be set at the same time
    mode : str, optional
        the interpolate method, by default "bilinear",
        support "bilinear" and "nearest"
    align_corners : bool, optional
    recompute_scale_factor : bool, optional
        did not support, by default None
    """

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=None,
        recompute_scale_factor=None,
    ):
        super(Interpolate, self).__init__()
        assert isinstance(size, (Integral, type(None))) or (
            isinstance(size, (tuple, list))
            and len(size) == 2
            and isinstance(size[0], Integral)
            and isinstance(size[1], Integral)
        ), "param 'size' must be int or tuple of two int or None"
        assert isinstance(scale_factor, (Real, type(None))) or (
            isinstance(scale_factor, (tuple, list))
            and len(scale_factor) == 2
            and isinstance(scale_factor[0], Real)
            and isinstance(scale_factor[1], Real)
        ), "param 'scale_factor' must be real or tuple of two real or None"
        assert mode in (
            "bilinear",
            "nearest",
        ), "mode only support 'bilinear' and 'nearest'"
        if mode == "nearest":
            assert (
                align_corners is None
            ), "align_corners option can only be set with 'bilinear' mode"
        else:
            if align_corners is None:
                warnings.warn(
                    "default upsampling behavior when mode={} is changed "
                    "to align_corners=False since torch 0.4.0. Please specify "
                    "align_corners=True if the old behavior "
                    "is desired. ".format(mode)
                )
                align_corners = False
            assert isinstance(
                align_corners, bool
            ), "param 'align_corners' must be bool or None"

        # only support align_corners=True on BAYES

        if get_march() == March.BERNOULLI2:
            assert (
                not align_corners
            ), "only support align_corners = True on BAYES"

        assert isinstance(
            recompute_scale_factor, (bool, type(None))
        ), "param 'recompute_scale_factor' must be bool or None"
        if scale_factor:
            assert (
                size is None
            ), "only one of size or scale_factor should be defined"
            assert recompute_scale_factor, (
                "only support recompute_scale_factor=True "
                + "when using scale_factor"
            )

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, data):
        return torch.nn.functional.interpolate(
            data,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )
