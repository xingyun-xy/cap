import torch


class Identity(torch.nn.Module):
    r"""
    A placeholder identity operator that is argument-insensitive.
    Return multi input as a Tuple.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        else:
            return input

    # implement from_float because we swap Dropout with Identity in convert
    @classmethod
    def from_float(cls, mod):
        return cls()
