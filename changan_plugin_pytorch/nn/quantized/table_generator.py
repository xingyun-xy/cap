import torch
from changan_plugin_pytorch.dtype import qinfo

from .activation_function_fit_utils import clamp_boundary
from changan_plugin_pytorch.march import March


class SingleTableGenerator(torch.nn.Module):
    r"""
    Only generate one table
    """

    def __init__(self, func, march):
        super(SingleTableGenerator, self).__init__()
        self.func = func
        self.march = march

    def forward(self, data_scale, scale, input_type: str, out_type: str):
        assert input_type == "qint8"
        assert out_type == "qint8"
        xmin = qinfo(input_type).min * data_scale
        xmax = qinfo(input_type).max * data_scale
        float_table = self.func(
            torch.linspace(
                xmin.item(), xmax.item(), 256, device=data_scale.device
            )
        )
        if self.march == "bernoulli":
            table = torch.clamp(
                torch.floor(float_table / scale),
                qinfo(out_type).min,
                qinfo(out_type).max,
            ).to(torch.int8)
        else:
            table = torch.clamp(
                torch.floor(float_table / scale + 0.5),
                qinfo(out_type).min,
                qinfo(out_type).max,
            ).to(torch.int8)
        return table


class MultiTableGenerator(torch.nn.Module):
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
        is_symmetric,
    ):
        super(MultiTableGenerator, self).__init__()
        self.func = func
        self.out_type = out_type
        self.iinfo = qinfo(out_type)
        self.dense_xmin = dense_xmin
        self.dense_xmax = dense_xmax
        self.sparse_xmin = sparse_xmin
        self.sparse_xmax = sparse_xmax
        self.left_line_xmin = left_line_xmin
        self.left_line_xmax = left_line_xmax
        self.right_line_xmin = right_line_xmin
        self.right_line_xmax = right_line_xmax
        self.is_symmetric = is_symmetric

    def forward(
        self,
        data_scale,
        out_scale,
        data_type: str,
        min_qint_boundary: int,
        max_qint_boundary: int,
        symmetric_b: float = 0.0,
        out_scale_coefficient: float = 1.0,
    ):
        # if use symmetric mode, only fit on positive half axis
        if self.is_symmetric:
            min_qint_boundary = 0
        # clamp original boundaries
        (
            qint_dense_xmin,
            qint_dense_xmax,
            qint_sparse_xmin,
            qint_sparse_xmax,
            qint_left_line_xmin,
            qint_left_line_xmax,
            qint_right_line_xmin,
            qint_right_line_xmax,
            qint_left_constant_xmin,
            qint_left_constant_xmax,
            qint_right_constant_xmin,
            qint_right_constant_xmax,
        ) = clamp_boundary(
            self.dense_xmin,
            self.dense_xmax,
            self.sparse_xmin,
            self.sparse_xmax,
            self.left_line_xmin,
            self.left_line_xmax,
            self.right_line_xmin,
            self.right_line_xmax,
            data_scale.item(),
            min_qint_boundary,
            max_qint_boundary,
        )
        if data_type == "qint8":
            qint_dense_xmin = qinfo("qint8").min
            qint_dense_xmax = qinfo("qint8").max
        if qint_left_constant_xmax < qint_left_constant_xmin:
            qint_left_constant_xmax = qint_left_constant_xmin
        if qint_right_constant_xmin > qint_right_constant_xmax:
            qint_right_constant_xmin = qint_right_constant_xmax

        # get float adjusted boundaries
        dense_xmin = qint_dense_xmin * data_scale
        dense_xmax = qint_dense_xmax * data_scale
        sparse_xmin = qint_sparse_xmin * data_scale
        sparse_xmax = qint_sparse_xmax * data_scale
        left_line_xmin = qint_left_line_xmin * data_scale
        left_line_xmax = qint_left_line_xmax * data_scale
        right_line_xmin = qint_right_line_xmin * data_scale
        right_line_xmax = qint_right_line_xmax * data_scale
        # calculate dense table and sparse table
        dense_table = self._generate_table(dense_xmin, dense_xmax, out_scale)
        sparse_table = self._generate_table(
            sparse_xmin, sparse_xmax, out_scale
        )

        # calculate y for line fit
        left_line_ymin = self.func(left_line_xmin)
        left_line_ymax = self.func(left_line_xmax)
        right_line_ymin = self.func(right_line_xmin)
        right_line_ymax = self.func(right_line_xmax)
        left_constant_fit_y = (
            self.func(min_qint_boundary * data_scale)
            + self.func(left_line_xmin)
        ) / 2
        right_constant_fit_y = (
            self.func(max_qint_boundary * data_scale)
            + self.func(right_line_xmax)
        ) / 2

        qint_symmetric_b = torch.clamp(
            torch.round(symmetric_b / (out_scale * out_scale_coefficient)),
            qinfo("qint16").min,
            qinfo("qint16").max,
        ).to(torch.int16)
        final_out_scale = out_scale * out_scale_coefficient
        return (
            dense_table,
            torch.tensor([qint_dense_xmin], device=data_scale.device),
            torch.tensor([qint_dense_xmax], device=data_scale.device),
            sparse_table,
            torch.tensor([qint_sparse_xmin], device=data_scale.device),
            torch.tensor([qint_sparse_xmax], device=data_scale.device),
            left_line_xmin,
            left_line_ymin,
            left_line_xmax,
            left_line_ymax,
            right_line_xmin,
            right_line_ymin,
            right_line_xmax,
            right_line_ymax,
            torch.tensor([qint_left_constant_xmin], device=data_scale.device),
            torch.tensor([qint_left_constant_xmax], device=data_scale.device),
            torch.tensor([qint_right_constant_xmin], device=data_scale.device),
            torch.tensor([qint_right_constant_xmax], device=data_scale.device),
            left_constant_fit_y,
            right_constant_fit_y,
            qint_symmetric_b,
            final_out_scale,
        )

    def _generate_table(self, xmin, xmax, out_scale):
        if self.out_type == "qint8":
            scale = out_scale / 256
        else:
            scale = out_scale
        float_table = self.func(
            torch.linspace(
                xmin.item(), xmax.item(), 256, device=out_scale.device
            )
        )
        table = torch.clamp(
            torch.floor(float_table / scale + 0.5),
            qinfo("qint16").min,
            qinfo("qint16").max,
        )
        return table
