from numbers import Real

from torch.nn import Module


class SegmentLUT(Module):
    """
    Simulate any elementwise function by:
        Segment Look Up Table for int16 input.
        Look Up Table for int8 input.

    Args:
        simulated_func (function): Simulated function.
        is_centrosymmetric (bool): Whether -y=F(-x).
        dividing_points (List[Real], optional):
            Manually set the max input value of each segment.
            Defaults to None.
        input_range (Tuple[Real, Real], optional):
            Manually set the valid input range.
        auto_divide_strategy (str):
            Strategy used to generate dividing points when
            dividing_points is None, only support 'evenly' and 'curvature'.
        inverse_func (function): The inverse function of the simulated function
            used to compute the input range in int-infer stage.
            !!!Note: Can only be used in monotonically decreasing function!!!
            Otherwise, the result of int-infer may be unexpected.
            Default to None
    """

    def __init__(
        self,
        simulated_func,
        is_centrosymmetric=False,
        dividing_points=None,
        input_range=None,
        auto_divide_strategy="evenly",
        inverse_func=None,
    ):
        super(SegmentLUT, self).__init__()

        assert isinstance(
            is_centrosymmetric, bool
        ), "param is_centrosymmetric must be bool"

        msg = "param dividing_points must be None or list or tuple of 8 float"
        assert isinstance(dividing_points, type(None)) or (
            isinstance(dividing_points, (list, tuple))
            and len(dividing_points) == 8
        ), msg
        if isinstance(dividing_points, (list, tuple)):
            for v in dividing_points:
                assert isinstance(v, Real), msg
        msg = "param input_range must be None or list or tuple of 2 float"
        assert isinstance(input_range, type(None)) or (
            isinstance(input_range, (list, tuple)) and len(input_range) == 2
        ), msg
        if isinstance(input_range, (list, tuple)):
            for v in input_range:
                assert isinstance(v, Real), msg
        assert auto_divide_strategy in (
            "evenly",
            "curvature",
        ), "Unsupported strategy"

        self.simulated_func = simulated_func
        self.is_centrosymmetric = is_centrosymmetric
        self.dividing_points = dividing_points
        self.input_range = input_range
        self.auto_divide_strategy = auto_divide_strategy
        self.inverse_func = inverse_func

    def forward(self, input):
        return self.simulated_func(input)
