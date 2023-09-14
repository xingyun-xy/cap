from torch import nn


class QuantStub(nn.Module):
    r"""
    Same as torch.nn.QuantStub, with an additional param to
    specify a fixed scale.

    Args:
        scale (float, optional): Pass a number to use as fixed scale.
            If set to None, scale will be computed by observer during forward.
            Defaults to 1.0/128.
        zero_point (int, optional): Pass a number to use as fixed zero_point.
            Defaults to 0.
        qconfig (optional): Quantization configuration for the tensor, if
            qconfig is not provided, we will get qconfig from parent modules.
            Defaults to None.
    """

    def __init__(
        self, scale: float = 1.0 / 128, zero_point: int = 0, qconfig=None
    ):
        super(QuantStub, self).__init__()
        if scale is not None:
            assert (
                zero_point is not None
            ), "zero_point must be provided while scale is fixed"
        self.scale = scale
        self.zero_point = zero_point
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x
