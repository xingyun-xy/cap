import numbers
from typing import List, Union

import torch
from torch import Size, Tensor, nn


class LayerNorm(nn.LayerNorm):
    weight: Tensor
    bias: Tensor

    """
    Layer normalization with single dimention norm support.
    Only support NCHW input layout.

    Args:
        normalized_shape (int):
            Input shape from an expected input of size
            [* x normalized_shape[0] x … x normalized_shape[-1]].
        eps (float, optional):
            A value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool, optional):
            Whether use learnable per-element affine parameters.
            Defaults to True.
        dim (int, optional):
            If specified, the normalization will be done alone this dimention.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        dim=None,
    ):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        assert isinstance(
            normalized_shape, (list, tuple, Size)
        ), "normalized_shape muse be a intergral or list or tuple or torch.Size"  # noqa
        assert (
            len(normalized_shape) < 4
        ), "Only support layernorm on W or HW or CHW."
        for v in normalized_shape:
            assert isinstance(
                v, numbers.Integral
            ), "elements of normalized_shape must be integral"
        assert isinstance(eps, float), "param eps must be a float"
        assert isinstance(
            elementwise_affine, bool
        ), "param elementwise_affine must be a bool"
        assert isinstance(
            dim, (type(None), numbers.Integral)
        ), "param dim must be None or a integral"
        assert dim in (
            None,
            1,
            2,
            3,
            -1,
            -2,
            -3,
        ), "Only support layernorm on W or HW or CHW."

        if dim is not None:
            if dim < 0:
                assert (
                    len(normalized_shape) == -dim
                ), "normalized_shape should not include dim before the dim to be normalized"  # noqa
            normalized_shape = [normalized_shape[0]] + (
                [1] * (len(normalized_shape) - 1)
            )

        super(LayerNorm, self).__init__(
            normalized_shape, eps, elementwise_affine, device, dtype
        )

        self.dim = dim

    def single_dim_norm(self, input: torch.Tensor):
        if self.dim >= 0:
            assert (
                self.dim + len(self.normalized_shape) == input.ndim
            ), "normalized_shape should not include dim before the dim to be normalized"  # noqa

        var, mean = torch.var_mean(
            input, dim=self.dim, unbiased=False, keepdim=True
        )
        x = (input - mean) * torch.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

    def forward(self, input):
        if self.dim is not None:
            return self.single_dim_norm(input)
        else:
            return super(LayerNorm, self).forward(input)
