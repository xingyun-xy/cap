import torch


class LookUpTable(torch.nn.Module):
    r"""apply look up table operator
        This operator input range is [-128, 127],
        -128 corresponds to the table index is 0.
        So add 128 to input is the final table index

    Args:
        table: tuple, table for looking up
    """

    def __init__(self, table):
        super(LookUpTable, self).__init__()
        assert isinstance(table, tuple), "Table should be tuple"
        assert len(table) <= 256, "Table size should less than 256"
        table_max_value = 127 / 128
        assert max(table) <= table_max_value and min(table) >= -1, (
            "Table element range must be: [-1,  %f]" % table_max_value
        )
        table = torch.tensor(table).to(torch.float)
        self.register_buffer("table", table)

    def forward(self, index):
        index = index.to(torch.long).contiguous() + 128
        return torch.take(self.table, index)
