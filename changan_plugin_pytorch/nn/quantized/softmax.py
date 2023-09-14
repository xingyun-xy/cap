import torch
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module

from .functional import softmax
from .table_generator import MultiTableGenerator


class SoftmaxTableGenerator(Module):
    def __init__(
        self,
        func,
        dense_xmin,
        dense_xmax,
        sparse_xmin,
        sparse_xmax,
        left_line_xmin,
        left_line_xmax,
        right_line_xmin,
        right_line_xmax,
        out_type,
        get_exp_params,
    ):
        super(SoftmaxTableGenerator, self).__init__()
        self.multi_table_generator = MultiTableGenerator(
            func,
            dense_xmin,
            dense_xmax,
            sparse_xmin,
            sparse_xmax,
            left_line_xmin,
            left_line_xmax,
            right_line_xmin,
            right_line_xmax,
            out_type,
            False,
        )
        self.get_exp_params = get_exp_params

    def forward(
        self,
        data_scale,
        out_scale,
        data_type: str,
        out_type: str,
        in_channels: int,
    ):
        assert data_type in ["qint8", "qint16"]
        # exp and reciprocal result are always int16 in softmax
        assert out_type == "qint16"
        type_min = qinfo(out_type).min
        type_max = qinfo(out_type).max
        min_boundary = type_min
        max_boundary = type_max
        div_data_scale = torch.tensor([0], device=out_scale.device)
        shift = torch.tensor([0])
        if self.get_exp_params:
            assert data_type == "qint8"
            shift = torch.ceil(
                torch.log2(
                    torch.tensor(float(in_channels), device=out_scale.device)
                )
            )
            min_boundary = qinfo("qint16").min
            max_boundary = qinfo("qint16").max
        if self.get_exp_params:
            div_data_scale = out_scale * 2 ** shift
        return (
            self.multi_table_generator(
                data_scale,
                out_scale,
                "qint16",
                min_boundary,
                max_boundary,
            ),
            shift,
            div_data_scale,
        )


class QuantSoftmax(Module):

    _QAT_MODULE = qat.Softmax

    def __init__(self, out_type="qint8", max_softmax_value=None):
        super(QuantSoftmax, self).__init__()
        self.input_type = "qint8"
        self.out_type = out_type
        if self.out_type == "qint8":
            self.iinfo = torch.iinfo(torch.int8)
        elif self.out_type == "qint16":
            self.iinfo = torch.iinfo(torch.int16)
        else:
            raise ValueError(self.out_type)
        assert max_softmax_value is not None
        # Softmax calculation must less than 1.
        # But use 1 or real max softmax calculation in scale calculation
        # has a significant impact on the accuracy, use real max softmax
        # is much better.
        scale = max_softmax_value * 2 / (self.iinfo.max - self.iinfo.min)
        self.exp_left_line_xmin = -40.0
        self.exp_left_line_xmax = -20.0
        self.exp_sparse_xmin = -20.0
        self.exp_sparse_xmax = -10.0
        self.exp_dense_xmin = -10.0
        self.exp_dense_xmax = 0.0
        self.exp_right_line_xmin = 0.0
        self.exp_right_line_xmax = 1.0
        self.exp_data_type = "qint8"
        self.exp_out_type = "qint16"

        self.reciprocal_left_line_xmin = 0.001
        self.reciprocal_left_line_xmax = 0.1
        # If necessary, increase right boundary of reciprocal dense table,
        # sparse table, linear fit interval to cover bigger sum of exp result.
        # The cost of this is the accuracy may decrease
        self.reciprocal_dense_xmin = 0.1
        self.reciprocal_dense_xmax = 250.0
        self.reciprocal_sparse_xmin = 250.0
        self.reciprocal_sparse_xmax = 500.0
        self.reciprocal_right_line_xmin = 500.0
        self.reciprocal_right_line_xmax = 700.0

        self.reciprocal_data_type = "qint16"
        self.reciprocal_out_type = "qint16"
        self.int16_info = torch.iinfo(torch.int16)
        exp_out_scale = torch.tensor([2]) / (
            self.int16_info.max - self.int16_info.min
        )
        # To achieve higher accuracy, we can increase the floating-point left
        # boundary of the reciprocal scale.
        # The cost of this is that the precision between the left boundary and
        # 0 will be very low.
        # If necessary, increase reciprocal_left_boundary to achieve higher
        # accuracy
        reciprocal_left_boundary = 0.1
        reciprocal_out_scale = (
            (torch.tensor([1]) / reciprocal_left_boundary)
            * 2
            / (self.int16_info.max - self.int16_info.min)
        )
        self.register_buffer("exp_out_scale", exp_out_scale)
        self.register_buffer("reciprocal_out_scale", reciprocal_out_scale)
        self.register_buffer("scale", scale)

        self.exp_table_generator = torch.jit.script(
            SoftmaxTableGenerator(
                torch.exp,
                self.exp_dense_xmin,
                self.exp_dense_xmax,
                self.exp_sparse_xmin,
                self.exp_sparse_xmax,
                self.exp_left_line_xmin,
                self.exp_left_line_xmax,
                self.exp_right_line_xmin,
                self.exp_right_line_xmax,
                self.exp_out_type,
                True,
            )
        )

        self.reciprocal_table_generator = torch.jit.script(
            SoftmaxTableGenerator(
                torch.reciprocal,
                self.reciprocal_dense_xmin,
                self.reciprocal_dense_xmax,
                self.reciprocal_sparse_xmin,
                self.reciprocal_sparse_xmax,
                self.reciprocal_left_line_xmin,
                self.reciprocal_left_line_xmax,
                self.reciprocal_right_line_xmin,
                self.reciprocal_right_line_xmax,
                self.reciprocal_out_type,
                False,
            )
        )

    def forward(self, data):
        (
            (
                exp_dense_table,
                exp_qint_dense_xmin,
                exp_qint_dense_xmax,
                exp_sparse_table,
                exp_qint_sparse_xmin,
                exp_qint_sparse_xmax,
                exp_left_line_xmin,
                exp_left_line_ymin,
                exp_left_line_xmax,
                exp_left_line_ymax,
                exp_right_line_xmin,
                exp_right_line_ymin,
                exp_right_line_xmax,
                exp_right_line_ymax,
                exp_qint_left_constant_xmin,
                exp_qint_left_constant_xmax,
                exp_qint_right_constant_xmin,
                exp_qint_right_constant_xmax,
                exp_left_constant_fit_y,
                exp_right_constant_fit_y,
                _,
                _,
            ),
            rescale_shift,
            div_data_scale,
        ) = self.exp_table_generator(
            data.q_scale(),
            self.exp_out_scale,
            self.exp_data_type,
            self.exp_out_type,
            data.size()[1],
        )

        (
            (
                reciprocal_dense_table,
                reciprocal_qint_dense_xmin,
                reciprocal_qint_dense_xmax,
                reciprocal_sparse_table,
                reciprocal_qint_sparse_xmin,
                reciprocal_qint_sparse_xmax,
                reciprocal_left_line_xmin,
                reciprocal_left_line_ymin,
                reciprocal_left_line_xmax,
                reciprocal_left_line_ymax,
                reciprocal_right_line_xmin,
                reciprocal_right_line_ymin,
                reciprocal_right_line_xmax,
                reciprocal_right_line_ymax,
                reciprocal_qint_left_constant_xmin,
                reciprocal_qint_left_constant_xmax,
                reciprocal_qint_right_constant_xmin,
                reciprocal_qint_right_constant_xmax,
                reciprocal_left_constant_fit_y,
                reciprocal_right_constant_fit_y,
                _,
                _,
            ),
            _,
            _,
        ) = self.reciprocal_table_generator(
            div_data_scale,
            self.reciprocal_out_scale,
            self.reciprocal_data_type,
            self.reciprocal_out_type,
            data.size()[1],
        )
        out = softmax(
            data.int_repr(),
            data.q_scale(),
            data.q_zero_point(),
            self.exp_data_type,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_type,
            self.exp_out_scale,
            torch.zeros_like(self.exp_out_scale).to(dtype=torch.long),
            self.exp_out_type,
            self.reciprocal_out_scale,
            torch.zeros_like(self.reciprocal_out_scale).to(dtype=torch.long),
            self.reciprocal_out_type,
            exp_dense_table,
            exp_qint_dense_xmin,
            exp_qint_dense_xmax,
            exp_sparse_table,
            exp_qint_sparse_xmin,
            exp_qint_sparse_xmax,
            exp_left_line_xmin,
            exp_left_line_ymin,
            exp_left_line_xmax,
            exp_left_line_ymax,
            exp_right_line_xmin,
            exp_right_line_ymin,
            exp_right_line_xmax,
            exp_right_line_ymax,
            exp_qint_left_constant_xmin,
            exp_qint_left_constant_xmax,
            exp_qint_right_constant_xmin,
            exp_qint_right_constant_xmax,
            exp_left_constant_fit_y,
            exp_right_constant_fit_y,
            rescale_shift,
            reciprocal_dense_table,
            reciprocal_qint_dense_xmin,
            reciprocal_qint_dense_xmax,
            reciprocal_sparse_table,
            reciprocal_qint_sparse_xmin,
            reciprocal_qint_sparse_xmax,
            reciprocal_left_line_xmin,
            reciprocal_left_line_ymin,
            reciprocal_left_line_xmax,
            reciprocal_left_line_ymax,
            reciprocal_right_line_xmin,
            reciprocal_right_line_ymin,
            reciprocal_right_line_xmax,
            reciprocal_right_line_ymax,
            reciprocal_qint_left_constant_xmin,
            reciprocal_qint_left_constant_xmax,
            reciprocal_qint_right_constant_xmin,
            reciprocal_qint_right_constant_xmax,
            reciprocal_left_constant_fit_y,
            reciprocal_right_constant_fit_y,
        )
        return QTensor(out, self.scale, self.out_type)

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
        quantized_softmax = cls(
            mod.activation_post_process.dtype, mod.max_softmax_value
        )
        return quantized_softmax
