import math
from numbers import Integral

import torch
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn,
)
from torch import nn


class Correlation(nn.Module):
    r"""Correlation performs multiplicative patch comparisons
    between two feature maps. Given two multi-channel feature maps
    :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c`
    being their width, height, and number of channels, the correlation
    layer lets the network compare each patch from :math:`f_{1}`
    with each patch from :math:`f_{2}`.

    For now we consider only a single comparison of two patches.
    The 'correlation' of two patches centered at :math:`x_{1}`
    in the first map and :math:`x_{2}` in the second map is then defined as:
    .. math::
        c(x_{1}, x_{2}) =
        \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>
    for a square patch of size :math:`K:=2k+1`.

    Note that the equation above is identical to one step of a convolution in
    neural networks, but instead of convolving data with a filter, it convolves
    data with other data. For this reason, it has no training weights.

    Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}`
    multiplications. Comparing all patch combinations involves
    :math:`w^{2}*h^{2}` such computations.

    Given a maximum displacement :math:`d`, for each location :math:`x_{1}`
    it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood
    of size :math:`D:=2d+1`, by limiting the range of :math:`x_{2}`.
    We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally
    and to quantize :math:`x_{2}` within the neighborhood centered
    around :math:`x_{1}`.

    The final output is defined by the following expression:
        .. math::
            out[n, q, i, j] = c(x_{i, j}, x_{q})
    where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`,
    and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.

    Args:
        kernel_size(int(non-negative), optional, default=1):
            kernel size for Correlation must be an odd number
        max_displacement(int(non-negative), optional, default=1):
            Max displacement of Correlation
        stride1(int (non-negative), optional, default=1):
            stride1 quantize data1 globally
        stride2(int (non-negative), optional, default=1):
            stride2 quantize data2 within neighborhood centered around data1
        pad_size(int (non-negative), optional, default=0): pad for Correlation
        is_multiply(boolean, optional, default=1):
            operation type is either multiplication or subduction
            only support True now
    """

    def __init__(
        self,
        kernel_size: int = 1,
        max_displacement: int = 1,
        stride1: int = 1,
        stride2: int = 1,
        pad_size: int = 0,
        is_multiply: bool = True,
    ):
        super(Correlation, self).__init__()
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.pad_size = pad_size
        self.is_multiply = is_multiply

        assert isinstance(
            kernel_size, Integral
        ), "param 'kernel_size' must be int"
        assert isinstance(
            max_displacement, Integral
        ), "param 'max_displacement' must be int"
        assert isinstance(stride1, Integral), "param 'stride1' must be int"
        assert isinstance(stride2, Integral), "param 'stride2' must be int"
        assert isinstance(pad_size, Integral), "param 'pad_size' must be int"
        assert isinstance(
            is_multiply, bool
        ), "param 'is_multiply' must be bool"

        assert (
            kernel_size > 0 and kernel_size % 2
        ), "Only support positive odd kernel_size"
        assert (
            max_displacement >= 0
        ), "Only support non-negative max_displacement"
        assert stride1 > 0, "Only support positive stride1"
        assert stride2 > 0, "Only support positive stride2"
        assert pad_size >= 0, "Only support non-negative pad_size"
        assert is_multiply, "Only support multiplication now"

        self.kernel_radius = (self.kernel_size - 1) / 2
        self.border_size = self.kernel_radius + self.max_displacement

        self.neighborhood_grid_radius = self.max_displacement // self.stride2
        self.neighborhood_grid_width = self.neighborhood_grid_radius * 2 + 1
        self.top_channels = int(
            self.neighborhood_grid_width * self.neighborhood_grid_width
        )

    def _fake_quanti_inter_out(self, interout, s):
        return interout

    @script_quantized_fn
    def forward(self, data1, data2):
        """
        Args:
            data1: Tensor/QTensor[N,C,H,W]
            data2: Tensor/QTensor[N,C,H,W]

        Return:
            out: Tensor
        """
        if isinstance(data1, QTensor) and isinstance(data2, QTensor):
            s = data1.q_scale() * data2.q_scale()
            data1 = data1.as_subclass(torch.Tensor)
            data2 = data2.as_subclass(torch.Tensor)
        else:
            s = None

        data1 = data1.contiguous()
        data2 = data2.contiguous()
        assert data1.ndim == 4 and data2.ndim == 4, "data must be a 4D tensor"
        assert data1.shape == data2.shape, (
            "Data with different shapes lead to unexpected results!"
            + "Please check your net config!"
        )

        num = data1.shape[0]
        channels = data1.shape[1]
        height = data1.shape[2]
        width = data1.shape[3]

        top_height = math.ceil(
            float(height + 2 * self.pad_size - self.border_size * 2)
            / float(self.stride1)
        )
        top_width = math.ceil(
            float(width + 2 * self.pad_size - self.border_size * 2)
            / float(self.stride1)
        )
        assert top_height >= 1 and top_width >= 1, (
            "Correlation cannot be done with current settings and data shape!"
            "Please check your configs."
        )
        inter_height = (top_height - 1) * self.stride1 + self.kernel_size
        inter_width = (top_width - 1) * self.stride1 + self.kernel_size

        tmp1 = nn.functional.pad(data1, (self.pad_size,) * 4).to(data1.device)
        tmp2 = torch.zeros_like(tmp1, device=data1.device)
        tmp2[
            :,
            :,
            self.pad_size : self.pad_size + data2.shape[2],
            self.pad_size : self.pad_size + data2.shape[3],
        ] = data2

        out = torch.zeros((num, self.top_channels, top_height, top_width)).to(
            data1.device
        )
        inter_out = torch.zeros(
            (num, self.top_channels, inter_height, inter_width)
        ).to(data1.device)

        self.sumelems = self.kernel_size * self.kernel_size * channels

        # computer inter_out
        for c in range(self.top_channels):
            x1 = self.max_displacement
            y1 = self.max_displacement
            s2o = int(
                (
                    c % self.neighborhood_grid_width
                    - self.neighborhood_grid_radius
                )
                * self.stride2
            )
            s2p = int(
                (
                    c // self.neighborhood_grid_width
                    - self.neighborhood_grid_radius
                )
                * self.stride2
            )
            x2 = self.max_displacement + s2o
            y2 = self.max_displacement + s2p
            if self.is_multiply:
                inter_out[:, c, :, :] = torch.sum(
                    tmp1[
                        :, :, y1 : y1 + inter_height, x1 : x1 + inter_width
                    ].mul(
                        tmp2[
                            :, :, y2 : y2 + inter_height, x2 : x2 + inter_width
                        ]
                    ),
                    (1),
                )
            else:
                inter_out[:, c, :, :] = torch.sum(
                    torch.abs(
                        tmp1[
                            :, :, y1 : y1 + inter_height, x1 : x1 + inter_width
                        ]
                        - tmp2[
                            :, :, y2 : y2 + inter_height, x2 : x2 + inter_width
                        ]
                    ),
                    (1),
                )

        # Actually take affect in qat
        inter_out = self._fake_quanti_inter_out(inter_out, s)

        # computer output
        for i in range(top_height):
            for j in range(top_width):
                y1 = i * self.stride1
                x1 = j * self.stride1
                out[:, :, i, j] = torch.sum(
                    inter_out[
                        :,
                        :,
                        y1 : y1 + self.kernel_size,
                        x1 : x1 + self.kernel_size,
                    ],
                    (2, 3),
                )
        out /= self.sumelems
        return out
