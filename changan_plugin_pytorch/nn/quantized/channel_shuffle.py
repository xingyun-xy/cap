from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch import nn


class ChannelShuffle(nn.Module):
    r"""Divide the channels in a tensor of shape :math:`(*, C , H, W)`
    into g groups and rearrange them as :math:`(*, C \frac g, g, H, W)`,
    while keeping the original tensor shape.

    Args:
        groups (int): number of groups to divide channels in.

    Examples::

        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> input = torch.randn(1, 4, 2, 2)
        >>> print(input)
        [[[[1, 2],
           [3, 4]],
          [[5, 6],
           [7, 8]],
          [[9, 10],
           [11, 12]],
          [[13, 14],
           [15, 16]],
         ]]
        >>> output = channel_shuffle(input)
        >>> print(output)
        [[[[1, 2],
           [3, 4]],
          [[9, 10],
           [11, 12]],
          [[5, 6],
           [7, 8]],
          [[13, 14],
           [15, 16]],
         ]]
    """

    _QAT_MODULE = qat.ChannelShuffle

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, input):
        from ..functional import channel_shuffle

        return QTensor(
            channel_shuffle(input.int_repr(), self.groups),
            input.q_scale(),
            input.dtype,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        quantized_channel_shuffle = cls(groups=mod.groups)
        return quantized_channel_shuffle
