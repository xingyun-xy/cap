import torch
from changan_plugin_pytorch.dtype import qinfo
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn import Module

from .functional import lut, multi_table_fit
from .table_generator import MultiTableGenerator, SingleTableGenerator
from changan_plugin_pytorch.march import get_march


class MultiTableFit(Module):
    def __init__(
        self,
        func,
        dense_xmin=-3.0,
        dense_xmax=3.0,
        sparse_xmin=-6.0,
        sparse_xmax=6.0,
        left_line_xmin=-10.0,
        left_line_xmax=-6.0,
        right_line_xmin=6.0,
        right_line_xmax=10.0,
        out_type="qint8",
        is_symmetric=False,
        symmetric_k=1,
        symmetric_b=0,
    ):
        super(MultiTableFit, self).__init__()
        assert out_type == "qint8" or out_type == "qint16", (
            "Unsupport out_type " + out_type
        )
        self.out_type = out_type
        self.out_scale_coefficient = 1 if out_type == "qint16" else 1 / 256
        self.is_symmetric = is_symmetric
        self.symmetric_k = symmetric_k
        self.symmetric_b = symmetric_b
        self.multi_table_generator = torch.jit.script(
            MultiTableGenerator(
                func,
                dense_xmin,
                dense_xmax,
                sparse_xmin,
                sparse_xmax,
                left_line_xmin,
                left_line_xmax,
                right_line_xmin,
                right_line_xmax,
                out_type=self.out_type,
                is_symmetric=is_symmetric,
            )
        )
        self.single_table_generator = torch.jit.script(
            SingleTableGenerator(func, get_march())
        )

    def forward(self, data, scale):
        if data.dtype == "qint8" and self.out_type == "qint8":
            table = self.single_table_generator(
                data.q_scale(), scale, data.dtype, self.out_type
            )
            out = lut(
                data.int_repr(),
                data.q_scale(),
                data.q_zero_point(),
                data.dtype,
                table,
                scale,
                torch.zeros_like(scale).to(dtype=torch.long),
                self.out_type,
            )
        else:
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
                final_out_scale,
            ) = self.multi_table_generator(
                data.q_scale(),
                scale,
                data.dtype,
                qinfo(data.dtype).min,
                qinfo(data.dtype).max,
                self.symmetric_b,
                self.out_scale_coefficient,
            )
            out = multi_table_fit(
                data=data.int_repr(),
                data_scale=data.q_scale(),
                data_zero_point=data.q_zero_point(),
                data_type=data.dtype,
                dense_table=dense_table,
                qint_dense_xmin=qint_dense_xmin,
                qint_dense_xmax=qint_dense_xmax,
                sparse_table=sparse_table,
                qint_sparse_xmin=qint_sparse_xmin,
                qint_sparse_xmax=qint_sparse_xmax,
                left_line_xmin=left_line_xmin,
                left_line_ymin=left_line_ymin,
                left_line_xmax=left_line_xmax,
                left_line_ymax=left_line_ymax,
                right_line_xmin=right_line_xmin,
                right_line_ymin=right_line_ymin,
                right_line_xmax=right_line_xmax,
                right_line_ymax=right_line_ymax,
                qint_left_constant_xmin=qint_left_constant_xmin,
                qint_left_constant_xmax=qint_left_constant_xmax,
                qint_right_constant_xmin=qint_right_constant_xmin,
                qint_right_constant_xmax=qint_right_constant_xmax,
                left_constant_fit_y=left_constant_fit_y,
                right_constant_fit_y=right_constant_fit_y,
                scale=final_out_scale,
                zero_point=torch.zeros_like(scale).to(dtype=torch.long),
                dtype=self.out_type,
                is_symmetric=self.is_symmetric,
                symmetric_k=self.symmetric_k,
                symmetric_b=qint_symmetric_b,
            )
        return QTensor(out, scale, self.out_type)
