import torch


def exponent_assign_and_check(mod, exponent):
    if not isinstance(exponent, int):
        assert (
            exponent.numel() == 1
        ), "Only support power which exponent is scalar"
    if mod.exponent is None:
        mod.exponent = exponent
    else:
        assert mod.exponent == exponent, (
            f"This Pow is only used for exponent {mod.exponent}, "
            f"but get {exponent}"
        )


class Pow(torch.nn.Module):
    "Module implementation of torch.pow"

    def __init__(self):
        super(Pow, self).__init__()
        self.exponent = None

    def forward(self, data, exponent):
        exponent_assign_and_check(self, exponent)
        return torch.pow(data, exponent)
