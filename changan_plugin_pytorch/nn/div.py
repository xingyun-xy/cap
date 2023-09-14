import torch


class Div(torch.nn.Module):
    """Module implementation of torch.div"""

    def __init__(self):
        super(Div, self).__init__()

    def forward(self, input, other):
        return torch.div(input, other)
