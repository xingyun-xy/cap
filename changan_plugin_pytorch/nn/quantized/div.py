import torch
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.qtensor import QTensor

from ..qat.div import Div as qat_div
from .functional import mul, multi_table_fit
from .table_generator import MultiTableGenerator


class Div(torch.nn.Module):
    """
    div is only used for int16 input and output
    """

    _QAT_MODULE = qat_div

    def __init__(self, scale):
        super(Div, self).__init__()
        self.dense_xmin = 0.01
        self.dense_xmax = 1.0
        self.sparse_xmin = 1.0
        self.sparse_xmax = 7.0
        self.left_line_xmin = 0.00001
        self.left_line_xmax = 0.01
        self.right_line_xmin = 7.0
        self.right_line_xmax = 20.0
        self.out_type = "qint16"
        # take 0.01 as the min absoulute value of bound for table scale
        # calculation because too small value of left bound result in too big
        # scale of reciprocal table, result in accuracy loss because
        # quantization result will be smaller, and the clipped proportion
        # increase compared to the quantization result.
        # the price of doing so is the Number with absolute value less than 1
        # will be clipped to min of qint or max of qint.
        # wee have to compromise like this
        table_scale = (
            torch.tensor([2])
            / 0.01
            / (qinfo("qint16").max - qinfo("qint16").min)
        )
        self.register_buffer("table_scale", table_scale)
        self.register_buffer("scale", scale)
        self.multi_table_generate = torch.jit.script(
            MultiTableGenerator(
                torch.reciprocal,
                self.dense_xmin,
                self.dense_xmax,
                self.sparse_xmin,
                self.sparse_xmax,
                self.left_line_xmin,
                self.left_line_xmax,
                self.right_line_xmin,
                self.right_line_xmax,
                out_type=self.out_type,
                is_symmetric=True,
            )
        )

    def forward(self, data1, data2):
        (
            dense_table,
            qint_dense_xmin,
            qint_dense_xmax,
            sparse_table,
            qint_sparse_xmin,
            qint_sparse_xmax,
            left_line_xmin,
            left_line_ymin,
            left_line_xmax,
            left_line_ymax,
            right_line_xmin,
            right_line_ymin,
            right_line_xmax,
            right_line_ymax,
            qint_left_constant_xmin,
            qint_left_constant_xmax,
            qint_right_constant_xmin,
            qint_right_constant_xmax,
            left_constant_fit_y,
            right_constant_fit_y,
            qint_symmetric_b,
            _,
        ) = self.multi_table_generate(
            data2.q_scale(),
            self.table_scale,
            data2.dtype,
            qinfo(data2.dtype).min,
            qinfo(data2.dtype).max,
            0.0,
            1.0,
        )

        reciprocal_res = multi_table_fit(
            data2.int_repr(),
            data2.q_scale(),
            data2.q_zero_point(),
            data2.dtype,
            dense_table,
            qint_dense_xmin,
            qint_dense_xmax,
            sparse_table,
            qint_sparse_xmin,
            qint_sparse_xmax,
            left_line_xmin,
            left_line_ymin,
            left_line_xmax,
            left_line_ymax,
            right_line_xmin,
            right_line_ymin,
            right_line_xmax,
            right_line_ymax,
            qint_left_constant_xmin,
            qint_left_constant_xmax,
            qint_right_constant_xmin,
            qint_right_constant_xmax,
            left_constant_fit_y,
            right_constant_fit_y,
            self.table_scale,
            torch.zeros_like(self.table_scale).to(dtype=torch.long),
            "qint16",
            True,
            -1,
            qint_symmetric_b,
        )
        mul_res = mul(
            data1.int_repr(),
            reciprocal_res,
            data1.q_scale(),
            data1.q_zero_point(),
            "qint16",
            self.table_scale,
            torch.zeros_like(self.table_scale).to(dtype=torch.long),
            "qint16",
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            "qint16",
        )
        return QTensor(mul_res, self.scale, self.out_type)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module

        Args: `mod` a qat module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        quantized_div = cls(
            scale=mod.activation_post_process.scale,
            out_dtype=mod.activation_post_process.dtype,
        )
        quantized_div.scale.copy_(mod.activation_post_process.scale)
        return quantized_div
